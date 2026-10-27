"""Single-file, standard-library-only pod agent. Run with system python3.

RunPod terminates TLS. Every route, including health, requires the claim token.
By default the token grants pod-user command/file authority. Trusted startup
may restrict it to health and curated read-only files with ObservationPolicy.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
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


class AgentServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, token, state_dir=None, *, observation_policy=None):
        if len(token) < 32 or not token.isascii():
            raise ValueError("a random token of at least 32 ASCII characters is required")
        self.token = token
        self.observation_policy = (None if observation_policy is None
                                   else ObservationPolicy(observation_policy))
        self.jobs = {}
        self.lock = threading.RLock()
        if self.observation_policy is not None:
            # No upload journal recovery or other control-state IO in this mode.
            super().__init__(address, Handler)
            return
        self.state_dir = Path(state_dir or tempfile.mkdtemp(prefix="astrid-agent-"))
        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.state_dir.chmod(0o700)
        # Incomplete chunks are committed by offset; restart SHA verification
        # if the agent exited while hashing, before any target promotion.
        for meta in self.state_dir.glob("*.json"):
            state = json.loads(meta.read_text())
            if state.get("verifying"):
                if not Path(state["staging"]).exists():
                    # Promotion may have completed before its metadata write.
                    state["error"] = "promotion outcome unknown; inspect destination before retry"
                state["verifying"] = False
                save_state(meta, state)
        super().__init__(address, Handler)

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
            if digest_file(state["staging"]) != state["sha256"]:
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
            return self.server.observation_policy.open(path)
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
        if body is None:
            body = json.dumps(data or {}).encode()
        self.send_response(status)
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
        data = self.rfile.read(length)
        if len(data) != length:
            raise ValueError("short body")
        return data

    def json_body(self):
        data = json.loads(self.body())
        if not isinstance(data, dict):
            raise ValueError("JSON object required")
        return data

    def dispatch(self):
        authorization = self.headers.get("Authorization", "")
        if not hmac.compare_digest(authorization.encode(), ("Bearer " + self.server.token).encode()):
            self.close_connection = True
            return self.reply(401, {"error": "unauthorized"})
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        path = parsed.path
        if (self.server.observation_policy is not None
                and (self.command, path) not in {("GET", "/health"),
                    ("POST", "/download-chunk"), ("GET", "/files")}):
            self.close_connection = True
            return self.reply(403, {"error": "forbidden"})
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
            target = Path(data["path"])
            size, sha = data["size"], data["sha256"]
            if not target.is_absolute() or type(size) is not int or size < 0 or not re.fullmatch(r"[a-f0-9]{64}", sha):
                raise ValueError("invalid upload metadata")
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
                if self.command == "PUT":
                    chunk = self.body()
                    offset = int(query.get("offset", ["-1"])[0])
                    if state["complete"] or state.get("verifying") or offset != state["offset"] or offset + len(chunk) > state["size"]:
                        return self.reply(409, {"offset": state["offset"]})
                    with open(state["staging"], "r+b") as stream:
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
    server = AgentServer(("0.0.0.0", int(os.environ.get("ASTRID_POD_PORT", "8000"))), token,
                         os.environ.get("ASTRID_POD_STATE", "/tmp/astrid-pod-agent"),
                         observation_policy=policy)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
