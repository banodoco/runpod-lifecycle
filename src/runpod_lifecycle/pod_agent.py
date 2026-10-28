"""Single-file, standard-library-only pod agent. Run with system python3.

RunPod terminates TLS. Every route, including health, requires the claim token.
By default the token grants pod-user command/file authority. Trusted startup
may restrict it to health and curated read-only files with ObservationPolicy.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
import hmac
import http.client
import json
import os
import re
import socket
import signal
import stat
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

CHUNK = 1024 * 1024
OUTPUT_CHUNK = 256 * 1024
ID = re.compile(r"[a-f0-9]{32,64}\Z")
OBSERVATION_ENV = "ASTRID_POD_OBSERVATION_B64"
OBSERVATION_LIMIT = 16384  # ASCII bytes of the complete base64 env value.
OBSERVATION_SCHEMA = "astrid.pod-observation.v1"
ENGINE_ENV = "ASTRID_POD_ENGINE"
# Startup selection with a private state directory travels as base64 JSON so
# provider env serialization (the SDK interpolates values into GraphQL
# strings) never sees quotes, backslashes or path text.
ENGINE_B64_ENV = "ASTRID_POD_ENGINE_B64"
ENGINE_B64_LIMIT = 16384
ENGINE_BODY_LIMIT = 1024 * 1024 * 1024
ENGINE_TRANSFER_SECONDS = 15


class EngineProxy:
    """One trusted startup slot. The upstream host is deliberately not an input."""

    def __init__(self, endpoint_id, port, *, input_files=(), output_roots=()):
        if (not isinstance(endpoint_id, str) or not re.fullmatch(r"[a-z][a-z0-9-]*", endpoint_id)
                or type(port) is not int or not 1 <= port <= 65535):
            raise ValueError("invalid engine proxy slot")
        self.endpoint_id, self.port = endpoint_id, port
        self.prefix = "/engines/" + endpoint_id
        if (not isinstance(input_files, (list, tuple)) or len(input_files) > 32
                or not isinstance(output_roots, (list, tuple)) or len(output_roots) > 8):
            raise ValueError("invalid engine transfer configuration")
        for path in (*input_files, *output_roots):
            observation_path(path)
        if len(set(input_files)) != len(input_files) or len(set(output_roots)) != len(output_roots):
            raise ValueError("duplicate engine transfer path")
        self.input_files = tuple(input_files)
        self.output_roots = tuple(output_roots)
        self.output_policy = (ObservationPolicy({"schema": OBSERVATION_SCHEMA,
                              "roots": list(output_roots), "files": []}) if output_roots else None)

    def configuration(self):
        data = {"endpoint_id": self.endpoint_id, "port": self.port}
        if self.input_files:
            data["input_files"] = list(self.input_files)
        if self.output_roots:
            data["output_roots"] = list(self.output_roots)
        return data

    @contextmanager
    def input_parent(self, path):
        # Exact destinations only, from trusted startup. Open all ancestors
        # without symlinks and keep the final directory fd through the IO.
        if path not in self.input_files:
            raise ObservationDenied()
        directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            for part in path.split("/")[1:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                dir_fd=directory)
                os.close(directory)
                directory = child
            info = os.fstat(directory)
            if info.st_uid != os.getuid() or info.st_mode & 0o022:
                raise ObservationDenied()
            name = path.rsplit("/", 1)[-1]
            try:
                info = os.stat(name, dir_fd=directory, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid():
                    raise ObservationDenied()
            yield directory
        finally:
            os.close(directory)


def engine_operation(method, target):
    """Closed ComfyUI surface, also used by the single-slot client adapter."""
    if (not isinstance(target, str) or not target.startswith("/")
            or any(ord(c) <= 32 or ord(c) >= 127 for c in target)
            or any(c in target for c in "\\#") or len(target) > 16384):
        return False
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or "%" in parsed.path:
        return False
    path = parsed.path
    if path.startswith("/api/"):
        path = path[4:]
    if method == "GET":
        return (path in {"/object_info", "/system_stats", "/queue", "/history"}
                or re.fullmatch(r"/(?:history|object_info)/[A-Za-z0-9_-]+", path) is not None)
    return method == "POST" and path in {
        "/prompt", "/queue", "/interrupt", "/free"}


def relay_engine(handler, connection, target, *, bearer=None, timeout=None):
    """One short HTTP exchange, no redirects, upgrades, ranges or retries.

    A lost prompt reply is acceptance-unknown. The timer bounds the entire
    outbound exchange, including connect/headers and slow streamed bodies.
    """
    handler.close_connection = True
    started = time.monotonic()
    budget = min(ENGINE_TRANSFER_SECONDS, timeout if timeout is not None else ENGINE_TRANSFER_SECONDS)
    if budget <= 0:
        connection.close()
        return handler.reply(504, {"error": "engine deadline exceeded"})
    deadline = started + budget
    connection.timeout = budget
    def expire():
        sock = connection.sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
    timer = threading.Timer(budget, expire)
    timer.daemon = True
    timer.start()
    headers = {}
    response = None
    sent = False
    try:
        if (handler.headers.get("Upgrade") is not None or handler.headers.get("Range") is not None
                or "upgrade" in {v.strip().lower() for v in handler.headers.get("Connection", "").split(",")}):
            return handler.reply(403, {"error": "HTTP-only engine surface"})
        if (handler.headers.get("Transfer-Encoding")
                or len(handler.headers.get_all("Content-Length", [])) > 1):
            raise ValueError("explicit request framing required")
        length = int(handler.headers.get("Content-Length", "0"))
        if not 0 <= length <= ENGINE_BODY_LIMIT:
            raise ValueError("engine body exceeds limit")
        headers["Connection"] = "close"
        # Deliberate allowlist: no client Authorization, Cookie, Host, proxy
        # headers, forwarding destination or hop-by-hop headers reach ComfyUI.
        for name in ("Content-Type", "Accept"):
            if name in handler.headers:
                headers[name] = handler.headers[name]
        if bearer is not None:
            headers["Authorization"] = "Bearer " + bearer
        if length or handler.command == "POST":
            headers["Content-Length"] = str(length)
        connection.putrequest(handler.command, target, skip_accept_encoding=True)
        for name, value in headers.items():
            connection.putheader(name, value)
        connection.endheaders()
        remaining = length
        while remaining:
            budget = min(ENGINE_TRANSFER_SECONDS, deadline - time.monotonic())
            if budget <= 0:
                raise TimeoutError()
            handler.connection.settimeout(budget)
            connection.sock.settimeout(budget)
            block = handler.rfile.read(min(CHUNK, remaining))
            if not block:
                raise ValueError("short engine upload")
            connection.send(block)
            remaining -= len(block)
        connection.sock.settimeout(min(ENGINE_TRANSFER_SECONDS, max(0.001, deadline - time.monotonic())))
        upstream_socket = connection.sock
        response = connection.getresponse()
        if 300 <= response.status < 400:
            return handler.reply(502, {"error": "engine redirect refused"})
        if response.status == 101:
            return handler.reply(502, {"error": "unexpected engine upgrade"})
        declared = response.getheader("Content-Length")
        size = None if declared is None else int(declared)
        if size is not None and not 0 <= size <= ENGINE_BODY_LIMIT:
            raise ValueError("engine response exceeds limit")
        first = b""
        if response.status == 404 and size != 0:
            upstream_socket.settimeout(min(ENGINE_TRANSFER_SECONDS, max(0.001, deadline - time.monotonic())))
            first = response.read1(min(CHUNK, ENGINE_BODY_LIMIT))
        if response.status == 404 and not first:
            # An empty-bodied 404 is the provider's no-route signature, which
            # clients may retry unchanged. This agent never emits one, so an
            # empty engine 404 becomes a final JSON 404 here.
            return handler.reply(404, {"error": "engine resource not found"})
        sent = True
        handler.send_response(response.status)
        handler.send_header("Connection", "close")
        handler.send_header("Cache-Control", "no-store")
        for name in ("Content-Type", "Content-Range", "ETag"):
            value = response.getheader(name)
            if value is not None:
                handler.send_header(name, value)
        if size is not None:
            handler.send_header("Content-Length", str(size))
        handler.end_headers()
        total = len(first)
        if first:
            handler.wfile.write(first)
        while not response.isclosed():
            budget = min(ENGINE_TRANSFER_SECONDS, deadline - time.monotonic())
            if budget <= 0:
                raise TimeoutError()
            upstream_socket.settimeout(budget)
            handler.connection.settimeout(budget)
            block = response.read1(min(CHUNK, ENGINE_BODY_LIMIT - total + 1))
            if not block:
                if size is not None and total != size:
                    raise OSError("short engine response")
                break
            total += len(block)
            if total > ENGINE_BODY_LIMIT:
                raise ValueError("engine response exceeds limit")
            handler.wfile.write(block)
        if size is not None and total != size:
            raise OSError("short engine response")
    except (OSError, http.client.HTTPException):
        if not sent:
            handler.reply(502, {"error": "engine transport unavailable"})
    except (ValueError, TypeError):
        if not sent:
            handler.reply(400 if response is None else 502, {"error": "invalid engine exchange"})
    finally:
        if response is not None:
            response.close()
        timer.cancel()
        connection.close()


class ObservationDenied(ValueError):
    pass


def observation_path(value):
    # Inspect the original spelling: Path would silently remove dot/empty parts.
    if (not isinstance(value, str) or not value.startswith("/")
            or "\x00" in value or "\\" in value
            or any(part in ("", ".", "..") for part in value.split("/")[1:])):
        raise ValueError("invalid observation path")
    return value


class ObservationPolicy:
    """Startup-only public allowlist; never inferred from a download request.

    Roots contain only curated, closed public files. The publisher is responsible
    for sanitizing/closing them; this policy supplies no publication or exec API.
    """

    def __init__(self, data):
        error = "invalid observation policy"
        if (not isinstance(data, dict) or set(data) != {"schema", "roots", "files"}
                or data["schema"] != OBSERVATION_SCHEMA):
            raise ValueError(error)
        seen = set()
        path_characters = 0
        for field, limit in (("roots", 8), ("files", 32)):
            values = data[field]
            if not isinstance(values, list) or len(values) > limit:
                raise ValueError(error)
            for value in values:
                # Bound work before splitting paths or serializing escaped JSON.
                if isinstance(value, str):
                    path_characters += len(value)
                    if path_characters > OBSERVATION_LIMIT:
                        raise ValueError(error)
                try:
                    observation_path(value)
                except ValueError:
                    raise ValueError(error) from None
                if value in seen:
                    raise ValueError(error)
                seen.add(value)
        if not seen:
            raise ValueError(error)
        # Snapshot the caller's lists; later mutation cannot expand authority.
        self.roots = tuple(data["roots"])
        self.files = tuple(data["files"])
        self.encoded = base64.b64encode(json.dumps(
            {"schema": OBSERVATION_SCHEMA, "roots": list(self.roots), "files": list(self.files)},
            separators=(",", ":"), ensure_ascii=True).encode()).decode("ascii")
        if len(self.encoded) > OBSERVATION_LIMIT:
            raise ValueError(error)

    @classmethod
    def from_environment(cls, value):
        def unique_object(pairs):
            result = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("duplicate field")
                result[key] = item
            return result

        try:
            if not isinstance(value, str) or not 0 < len(value) <= OBSERVATION_LIMIT:
                raise ValueError("invalid size")
            raw = base64.b64decode(value, validate=True)
            return cls(json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object))
        except (ValueError, TypeError, RecursionError):
            raise ValueError("invalid observation policy") from None

    def open(self, path):
        try:
            observation_path(path)
        except ValueError:
            raise ObservationDenied() from None
        if path not in self.files and not any(path.startswith(root + "/") for root in self.roots):
            raise ObservationDenied()
        # Resolve every component through an already open directory descriptor.
        # O_NONBLOCK prevents a hostile FIFO from blocking before fstat rejects it.
        # Never check a pathname and then reopen it for hashing/reading.
        directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            parts = path.split("/")[1:]
            for part in parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                dir_fd=directory)
                os.close(directory)
                directory = child
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                         dir_fd=directory)
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ObservationDenied()
                stream = os.fdopen(fd, "rb")
            except BaseException:
                os.close(fd)
                raise
            return stream
        finally:
            os.close(directory)


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_signature(path):
    info = os.stat(path, follow_symlinks=False)
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def save_state(path, state):
    temporary = path.with_suffix(".new")
    with temporary.open("w") as stream:
        json.dump(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_engine_bearer_file(state_dir, token):
    """Project the original startup bearer; an attachment never owns this file."""
    if (not isinstance(token, str) or not 32 <= len(token) < 256
            or any(ord(c) <= 32 or ord(c) >= 127 for c in token)):
        raise ValueError("invalid engine bearer")
    path = Path(state_dir)
    raw = str(state_dir)
    if (not path.is_absolute() or str(path) != raw
            or any(p in ("", ".", "..") for p in raw.split("/")[1:])):
        raise ValueError("unsafe engine state directory")
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        parts = path.parts[1:]
        for part in parts:
            try:
                os.mkdir(part, 0o700, dir_fd=directory)
            except FileExistsError:
                pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=directory)
            os.close(directory)
            directory = child
        info = os.fstat(directory)
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise ValueError("unsafe engine state directory")
        flags = os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
        try:
            fd = os.open("engine-bearer.token", os.O_WRONLY | os.O_CREAT | os.O_EXCL | flags,
                         0o600, dir_fd=directory)
        except FileExistsError:
            fd = os.open("engine-bearer.token", os.O_RDONLY | flags, dir_fd=directory)
            try:
                info = os.fstat(fd)
                if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                        or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600
                        or info.st_size != len(token)
                        or not hmac.compare_digest(os.read(fd, 256), token.encode("ascii"))):
                    raise ValueError("engine bearer restart differs or is unsafe")
            finally:
                os.close(fd)
        else:
            try:
                body = token.encode("ascii")
                while body:
                    written = os.write(fd, body)
                    if written <= 0:
                        raise OSError("short private write")
                    body = body[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
            os.fsync(directory)
        return path / "engine-bearer.token"
    except OSError:
        raise ValueError("engine bearer file is unavailable or unsafe") from None
    finally:
        os.close(directory)


class AgentServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, token, state_dir=None, *, observation_policy=None, engine_proxy=None):
        if len(token) < 32 or not token.isascii():
            raise ValueError("a random token of at least 32 ASCII characters is required")
        self.token = token
        # This configuration comes only from trusted startup, never a route.
        self.engine_proxy = None if engine_proxy is None else EngineProxy(**engine_proxy)
        self.observation_policy = (None if observation_policy is None
                                   else ObservationPolicy(observation_policy))
        self.jobs = {}
        self.lock = threading.RLock()
        self.state_dir = Path(state_dir or tempfile.mkdtemp(prefix="astrid-agent-"))
        self._engine_state_dir = self.state_dir
        if self.engine_proxy is not None:
            _write_engine_bearer_file(self.state_dir, self.token)
        if self.observation_policy is not None:
            if self.engine_proxy is not None and self.engine_proxy.input_files:
                self._enable_uploads()
            # No upload journal recovery or other control-state IO in this mode.
            super().__init__(address, Handler)
            return
        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.state_dir.chmod(0o700)
        # Incomplete chunks are committed by offset; restart SHA verification
        # if the agent exited while hashing, before any target promotion.
        for meta in self.state_dir.glob("*.json"):
            if meta.name == "engine-readiness.json":
                continue
            state = json.loads(meta.read_text())
            if state.get("verifying"):
                if not Path(state["staging"]).exists():
                    # Promotion may have completed before its metadata write.
                    state["error"] = "promotion outcome unknown; inspect destination before retry"
                state["verifying"] = False
                save_state(meta, state)
        super().__init__(address, Handler)

    def _enable_uploads(self):
        self.state_dir = Path(self._engine_state_dir or tempfile.mkdtemp(prefix="astrid-engine-"))
        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.state_dir.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("engine upload state must be a private owned directory")

    def enable_engine(self, endpoint_id, port, *, input_files=(), output_roots=()):
        """Trusted installed-session handoff. No HTTP configuration endpoint.

        Observation reads stay curated; only exact configured input writes and
        output reads are added. Job/command authority is never granted.
        """
        selected = EngineProxy(endpoint_id, port, input_files=input_files, output_roots=output_roots)
        with self.lock:
            current = self.engine_proxy
            if current is not None and current.configuration() != selected.configuration():
                raise ValueError("the engine slot is already bound")
            if current is None and self.observation_policy is not None and selected.input_files:
                self._enable_uploads()
            _write_engine_bearer_file(self._engine_state_dir, self.token)
            self.engine_proxy = selected

    @contextmanager
    def upload_stream(self, state, mode):
        if self.observation_policy is None:
            with open(state["staging"], mode) as stream:
                yield stream
            return
        slot = self.engine_proxy
        if slot is None or state.get("reserved"):
            raise ObservationDenied()
        with slot.input_parent(state["path"]) as directory:
            staging = Path(state["staging"])
            if (str(staging.parent) != str(Path(state["path"]).parent)
                    or not re.fullmatch(r"\.astrid-upload-[a-f0-9]{32}", staging.name)):
                raise ObservationDenied()
            fd = os.open(staging.name, (os.O_RDWR if mode == "r+b" else os.O_RDONLY)
                         | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid():
                os.close(fd)
                raise ObservationDenied()
            with os.fdopen(fd, mode) as stream:
                yield stream

    def cancel(self, job):
        process = job["process"]
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            def reap():
                time.sleep(2)
                # Kill descendants even if the shell has exited on SIGTERM.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            threading.Thread(target=reap, daemon=True).start()

    def finish_upload(self, meta, state):
        try:
            with self.upload_stream(state, "rb") as stream:
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(CHUNK), b""):
                    digest.update(block)
                if digest.hexdigest() != state["sha256"]:
                    raise ValueError("sha256 mismatch")
            target = Path(state["path"])
            if state.get("reserved"):
                # Preserve inode/device/mode established by worker reserve_secret.
                fd = os.open(target, os.O_RDWR | os.O_NOFOLLOW)
                try:
                    info = os.fstat(fd)
                    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid()
                            or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size != 0):
                        raise ValueError("reserved target is not an owned empty 0600 file")
                    with os.fdopen(os.dup(fd), "wb") as dst, open(state["staging"], "rb") as src:
                        for block in iter(lambda: src.read(CHUNK), b""):
                            dst.write(block)
                        dst.flush()
                        os.fsync(dst.fileno())
                finally:
                    os.close(fd)
                os.unlink(state["staging"])
            elif self.observation_policy is not None:
                with self.engine_proxy.input_parent(state["path"]) as directory:
                    os.replace(Path(state["staging"]).name, target.name,
                               src_dir_fd=directory, dst_dir_fd=directory)
            else:
                os.replace(state["staging"], target)
            state["file_signature"] = file_signature(target)
            state["complete"] = True
        except (OSError, ValueError):
            state["error"] = "upload verification or promotion failed"
        finally:
            state["verifying"] = False
            with self.lock:
                save_state(meta, state)

    def server_close(self):
        for job in list(self.jobs.values()):
            self.cancel(job)
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    server_version = "AstridPodAgent/1"
    protocol_version = "HTTP/1.1"
    rbufsize = 0

    def log_message(self, *_args):
        pass  # No request paths, headers, command output or bearer tokens in logs.

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def __getattr__(self, name):
        # BaseHTTPRequestHandler otherwise sends a pre-auth 501 for unknown
        # methods, including HEAD/OPTIONS/PATCH. Route them through auth too.
        if name.startswith("do_") and self.server.observation_policy is not None:
            return self.handle_request
        raise AttributeError(name)

    def open_download(self, path):
        if self.server.observation_policy is not None:
            try:
                return self.server.observation_policy.open(path)
            except ObservationDenied:
                slot = self.server.engine_proxy
                if (self.command == "POST" and urlsplit(self.path).path == "/download-chunk"
                        and slot is not None and slot.output_policy is not None):
                    return slot.output_policy.open(path)
                raise
        target = Path(path)
        if not target.is_absolute():
            raise ValueError("absolute path required")
        return target.open("rb")

    def download_info(self, stream):
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("download requires a regular file")
        if self.server.observation_policy is not None and info.st_nlink != 1:
            raise ObservationDenied()
        return info

    def reply(self, status, data=None, *, body=None, headers=None):
        self.close_connection = True
        if body is None:
            body = json.dumps(data or {}).encode()
        self.send_response(status)
        self.send_header("Connection", "close")
        self.send_header("Content-Type", "application/json" if data is not None else "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, str(value))
        self.end_headers()
        self.wfile.write(body)

    def body(self, limit=CHUNK):
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("use bounded Content-Length chunks")
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 <= length <= limit:
            raise ValueError("request body exceeds limit")
        blocks, remaining = [], length
        while remaining:
            block = self.rfile.read(min(CHUNK, remaining))
            if not block:
                raise ValueError("short body")
            blocks.append(block)
            remaining -= len(block)
        return b"".join(blocks)

    def json_body(self):
        data = json.loads(self.body())
        if not isinstance(data, dict):
            raise ValueError("JSON object required")
        return data

    def dispatch(self):
        authorization = self.headers.get("Authorization", "")
        if (len(self.headers.get_all("Authorization", [])) != 1
                or not hmac.compare_digest(authorization.encode(), ("Bearer " + self.server.token).encode())):
            self.close_connection = True
            return self.reply(401, {"error": "unauthorized"})
        if self.server.engine_proxy is not None and (
                self.headers.get("Upgrade") is not None or self.headers.get("Range") is not None
                or "upgrade" in {v.strip().lower() for v in self.headers.get("Connection", "").split(",")}):
            return self.reply(403, {"error": "HTTP-only engine surface"})
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        path = parsed.path
        slot = self.server.engine_proxy
        engine_allowed = slot is not None and (
            path.startswith(slot.prefix + "/") or
            (slot.input_files and (path == "/uploads" or path.startswith("/uploads/"))))
        if (self.server.observation_policy is not None
                and (self.command, path) not in {("GET", "/health"),
                    ("POST", "/download-chunk"), ("GET", "/files")}
                and not engine_allowed):
            self.close_connection = True
            return self.reply(403, {"error": "forbidden"})
        if path.startswith("/engines/"):
            slot = self.server.engine_proxy
            if slot is None or not path.startswith(slot.prefix + "/"):
                return self.reply(404, {"error": "unknown engine slot"})
            target = self.path[len(slot.prefix):]
            if not engine_operation(self.command, target):
                return self.reply(403, {"error": "engine operation denied"})
            try:
                return relay_engine(self, http.client.HTTPConnection("127.0.0.1", slot.port, timeout=ENGINE_TRANSFER_SECONDS), target)
            except (OSError, http.client.HTTPException):
                # No response body, headers, token or upstream exception text.
                self.close_connection = True
                return
        if self.command == "GET" and path == "/health":
            return self.reply(200, {"ok": True, "protocol": 1})
        if path == "/jobs" and self.command == "POST":
            data = self.json_body()
            job_id, command = data["id"], data["command"]
            if not ID.fullmatch(job_id) or not isinstance(command, str) or not command:
                raise ValueError("invalid job")
            with self.server.lock:
                old = self.server.jobs.get(job_id)
                if old:
                    return self.reply(200 if old["command"] == command else 409, {"id": job_id})
                if len(self.server.jobs) >= 128:
                    return self.reply(429, {"error": "job limit reached; rotate claim"})
                directory = self.server.state_dir / job_id
                directory.mkdir(mode=0o700)
                environment = {k: v for k, v in os.environ.items()
                               if k not in {"ASTRID_POD_TOKEN", "ASTRID_POD_AGENT_B64"}}
                with (directory / "stdout").open("wb") as stdout, (directory / "stderr").open("wb") as stderr:
                    process = subprocess.Popen(command, shell=True, executable="/bin/bash", stdin=subprocess.DEVNULL,
                                               stdout=stdout, stderr=stderr, env=environment, start_new_session=True)
                self.server.jobs[job_id] = {"process": process, "directory": directory, "command": command}
            return self.reply(202, {"id": job_id})
        if path.startswith("/jobs/"):
            job_id = path.removeprefix("/jobs/")
            job = self.server.jobs.get(job_id)
            if not job:
                return self.reply(404, {"error": "unknown job"})
            if self.command == "DELETE":
                self.server.cancel(job)
                return self.reply(202, {"id": job_id})
            if self.command == "GET":
                code = job["process"].poll()
                result = {"id": job_id, "status": "running" if code is None else "exited", "exit_code": code}
                for name in ("stdout", "stderr"):
                    offset = int(query.get(name, ["0"])[0])
                    if offset < 0:
                        raise ValueError("negative offset")
                    with (job["directory"] / name).open("rb") as stream:
                        stream.seek(offset)
                        chunk = stream.read(OUTPUT_CHUNK)
                        result[name] = base64.b64encode(chunk).decode()
                        result[name + "_offset"] = offset + len(chunk)
                        result[name + "_eof"] = stream.tell() == os.fstat(stream.fileno()).st_size
                return self.reply(200, result)
        if path == "/uploads" and self.command == "POST":
            data = self.json_body()
            if self.server.observation_policy is not None:
                try:
                    observation_path(data["path"])
                except ValueError:
                    raise ObservationDenied() from None
            target = Path(data["path"])
            size, sha = data["size"], data["sha256"]
            if not target.is_absolute() or type(size) is not int or size < 0 or not re.fullmatch(r"[a-f0-9]{64}", sha):
                raise ValueError("invalid upload metadata")
            if self.server.observation_policy is not None:
                slot = self.server.engine_proxy
                if slot is None or data.get("reserved"):
                    raise ObservationDenied()
                with slot.input_parent(str(target)):
                    pass
            # File metadata and committed offsets survive server restarts.
            upload_id = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
            meta = self.server.state_dir / (upload_id + ".json")
            with self.server.lock:
                reusable = meta.exists()
                if reusable:
                    old = json.loads(meta.read_text())
                    if old["complete"]:
                        try:
                            reusable = file_signature(target) == old.get("file_signature")
                        except FileNotFoundError:
                            reusable = False
                if not reusable:
                    if self.server.observation_policy is not None:
                        with slot.input_parent(str(target)) as directory:
                            name = ".astrid-upload-" + os.urandom(16).hex()
                            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                         0o600, dir_fd=directory)
                            staging = str(target.parent / name)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        fd, staging = tempfile.mkstemp(prefix=".astrid-upload-", dir=target.parent)
                    os.close(fd)
                    data.update({"staging": staging, "offset": 0, "complete": False})
                    save_state(meta, data)
                state = json.loads(meta.read_text())
            return self.reply(200, {"id": upload_id, "offset": state["offset"], "complete": state["complete"], "error": state.get("error")})
        if path.startswith("/uploads/"):
            upload_id = path.removeprefix("/uploads/")
            if not ID.fullmatch(upload_id):
                raise ValueError("invalid upload ID")
            meta = self.server.state_dir / (upload_id + ".json")
            with self.server.lock:
                if not meta.exists():
                    return self.reply(404, {"error": "unknown upload"})
                state = json.loads(meta.read_text())
                if self.server.observation_policy is not None:
                    with self.server.engine_proxy.input_parent(state["path"]):
                        pass
                if self.command == "PUT":
                    chunk = self.body()
                    offset = int(query.get("offset", ["-1"])[0])
                    if state["complete"] or state.get("verifying") or offset != state["offset"] or offset + len(chunk) > state["size"]:
                        return self.reply(409, {"offset": state["offset"]})
                    with self.server.upload_stream(state, "r+b") as stream:
                        stream.seek(offset)
                        stream.write(chunk)
                        stream.flush()
                        os.fsync(stream.fileno())
                    state["offset"] += len(chunk)
                elif self.command == "POST":
                    self.body()
                    if not state["complete"] and not state.get("verifying"):
                        if state["offset"] != state["size"]:
                            return self.reply(422, {"error": "size mismatch"})
                        state["verifying"] = True
                        save_state(meta, state)
                        threading.Thread(target=self.server.finish_upload, args=(meta, state.copy()), daemon=True).start()

                elif self.command != "GET":
                    return self.reply(405, {"error": "method not allowed"})
                save_state(meta, state)
            return self.reply(200, {"id": upload_id, "offset": state["offset"], "complete": state["complete"], "error": state.get("error")})
        if path == "/download-chunk" and self.command == "POST":
            # Some HTTP intermediaries normalize Range/Content-Range. A JSON
            # envelope provides the same bounded byte offsets without relying
            # on hop-by-hop handling of those headers. /files still serves 206.
            data = self.json_body()
            offset = data["offset"]
            if type(offset) is not int or offset < 0:
                raise ValueError("invalid download range")
            with self.open_download(data["path"]) as stream:
                info = self.download_info(stream)
                if offset > info.st_size:
                    raise ValueError("invalid download file or offset")
                etag = "%s-%s-%s-%s" % (info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                if data.get("etag", etag) != etag:
                    return self.reply(412, {"error": "file changed"})
                stream.seek(offset)
                chunk = stream.read(CHUNK)
                after = self.download_info(stream)
                if (info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    return self.reply(412, {"error": "file changed"})
            return self.reply(200, {"offset": offset, "total": info.st_size, "etag": etag,
                                    "sha256": hashlib.sha256(chunk).hexdigest(),
                                    "data": base64.b64encode(chunk).decode()})
        if path == "/files" and self.command == "GET":
            with self.open_download(query["path"][0]) as stream:
                info = self.download_info(stream)
                size = info.st_size
                header = self.headers.get("Range", "")
                match = re.fullmatch(r"bytes=(\d+)-(\d*)", header)
                if not match:
                    raise ValueError("bounded byte Range required")
                start = int(match[1])
                if start >= size and size:
                    return self.reply(416, {"error": "range outside file"}, headers={"Content-Range": f"bytes */{size}"})
                end = min(int(match[2]) if match[2] else size - 1, size - 1, start + CHUNK - 1)
                if start < 0 or (size and end < start):
                    raise ValueError("invalid range")
                stream.seek(start)
                body = stream.read(max(0, end - start + 1))
                # Pin the representation for all chunks. Caller verifies full SHA
                # against the upload digest when a round trip is required.
                after = self.download_info(stream)
                if (info.st_size, info.st_mtime_ns, info.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    return self.reply(412, {"error": "file changed"})
                etag = '"%s-%s-%s-%s"' % (info.st_ino, size, info.st_mtime_ns, info.st_ctime_ns)
                if self.headers.get("If-Match", etag) != etag:
                    return self.reply(412, {"error": "file changed"})
            return self.reply(206, body=body, headers={"Content-Range": f"bytes {start}-{end}/{size}", "ETag": etag})
        return self.reply(404, {"error": "unknown route"})

    def handle_request(self):
        try:
            self.dispatch()
        except ObservationDenied:
            self.close_connection = True
            try:
                self.reply(403, {"error": "forbidden"})
            except OSError:
                pass
        except (ValueError, KeyError, TypeError, OSError):
            self.close_connection = True
            try:
                self.reply(400, {"error": "invalid request or file operation failed"})
            except OSError:
                pass

    do_GET = do_POST = do_PUT = do_DELETE = handle_request


def main():
    token = os.environ.pop("ASTRID_POD_TOKEN", "")
    os.environ.pop("ASTRID_POD_AGENT_B64", None)
    encoded = os.environ.pop(OBSERVATION_ENV, None)
    policy = None
    if encoded is not None:
        validated = ObservationPolicy.from_environment(encoded)
        policy = {"schema": OBSERVATION_SCHEMA, "roots": list(validated.roots),
                  "files": list(validated.files)}
    engine_options = {}
    encoded_engine = os.environ.pop(ENGINE_B64_ENV, None)
    if ENGINE_ENV in os.environ:
        if encoded_engine is not None:
            raise ValueError("conflicting engine startup selections")
        engine_options["engine_proxy"] = json.loads(os.environ.pop(ENGINE_ENV))
    elif encoded_engine is not None:
        if len(encoded_engine) > ENGINE_B64_LIMIT:
            raise ValueError("engine startup selection is too large")
        engine_options["engine_proxy"] = json.loads(base64.b64decode(encoded_engine, validate=True))
    server = AgentServer(("0.0.0.0", int(os.environ.get("ASTRID_POD_PORT", "8000"))), token,
                         os.environ.get("ASTRID_POD_STATE", "/tmp/astrid-pod-agent"),
                         observation_policy=policy, **engine_options)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
