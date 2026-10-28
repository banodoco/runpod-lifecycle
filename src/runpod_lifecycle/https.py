"""HTTPS transport over RunPod's public pod proxy (no SSH or pip on the pod)."""
from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import logging
import re
import shlex
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from . import pod_agent

TOKEN_ENV = "ASTRID_POD_TOKEN"
PORT = 8000
CHUNK = pod_agent.CHUNK
# RunPod's public proxy answers an EMPTY-bodied 404 when it has no route to the
# pod port. Right after a pod starts, that route appears and can still drop out
# intermittently for several seconds (observed live: 200, 200, 404, 200, 404...).
# The agent never sends an empty 404 (its 404s are JSON), so an empty 404 means
# the request did not reach the agent and is safe to retry within this bound.
ROUTE_GRACE_SECONDS = 90
READY_CONSECUTIVE = 3
# Detached jobs keep all state on the volume, so they outlive the client. The
# supervisor runs in its own session and process group, so cancelling
# the short launcher agent job never reaches the work.
DETACHED_ROOT = "/workspace/.astrid-jobs"
_DETACHED_ID = re.compile(r"[a-f0-9]{32}")
_DETACHED_SUPERVISOR = """\
d="$1"
echo "$$" > "$d/pid.tmp" && mv "$d/pid.tmp" "$d/pid"
if [ -e "$d/cancel" ]; then echo cancelled > "$d/status.tmp" && mv "$d/status.tmp" "$d/status"; exit 0; fi
echo running > "$d/status.tmp" && mv "$d/status.tmp" "$d/status"
rc=0
bash "$d/command.sh" > "$d/stdout" 2> "$d/stderr" < /dev/null || rc=$?
echo "$rc" > "$d/exit.tmp" && mv "$d/exit.tmp" "$d/exit"
echo exited > "$d/status.tmp" && mv "$d/status.tmp" "$d/status"
"""
logger = logging.getLogger(__name__)


class PodTransportError(RuntimeError):
    """Deliberately contains no request bodies, headers or token material."""


def agent_bootstrap(token: str, port: int = PORT, *, observation_policy=None, engine_proxy=None,
                    state_dir=None) -> dict:
    """Create SDK/REST-independent boot material. The token is only in env.

    Inlining the compressed single-file agent avoids mutable download URLs and
    any package installs on stock runpod/base images.

    observation_policy is an optional public astrid.pod-observation.v1 mapping
    with roots (up to 8 curated directories) and files (up to 32 exact paths).
    It restricts the agent to authenticated health/downloads. Its base64 JSON
    env value is capped at 16384 ASCII bytes and consumed at agent startup.
    None preserves the general command/file agent.

    engine_proxy is a trusted-startup {endpoint_id, port} slot, enabled only
    after the original installation/session readiness handoff. Its upstream
    is always 127.0.0.1; the provider exposes only this agent's port. It cannot
    be configured by an HTTP call. With an observation policy it preserves
    curated reads and denies ordinary commands. Optional input_files grants
    only exact absolute destinations to the existing chunk upload; optional
    output_roots grants JSON-chunk reads beneath configured directories.
    Both are installed-session startup configuration, never request inputs.
    Omitting the slot stays observation-only. Engine calls are HTTP-only.

    state_dir selects the agent's private ASTRID_POD_STATE directory (absolute,
    canonical, mode 0700 on the pod); it is accepted only with engine_proxy so
    the original bearer and engine-readiness.json share one private parent.
    With it the engine selection is sent base64-encoded (ASTRID_POD_ENGINE_B64),
    so every env value is plain safe text for the SDK's GraphQL interpolation.
    """
    import zlib
    if len(token) < 32 or not token.isascii():
        raise ValueError("claim token must have at least 32 ASCII characters")
    policy = (None if observation_policy is None
              else pod_agent.ObservationPolicy(observation_policy))
    engine = None if engine_proxy is None else pod_agent.EngineProxy(**engine_proxy)
    if state_dir is not None:
        from .engine_startup import _state_path
        if engine is None:
            raise ValueError("a state directory selection requires an engine proxy slot")
        state_dir = _state_path(state_dir, None)
    source = Path(pod_agent.__file__).read_bytes()
    payload = base64.b64encode(zlib.compress(source, 9)).decode()
    loader = ("import os,base64,zlib; "
              "exec(compile(zlib.decompress(base64.b64decode(os.environ.pop('ASTRID_POD_AGENT_B64'))),"
              "'<astrid-pod-agent>','exec'))")
    command = "python3 -u -c " + shlex.quote(loader)
    environment = {TOKEN_ENV: token, "ASTRID_POD_AGENT_B64": payload, "ASTRID_POD_PORT": str(port)}
    if engine is not None:
        encoded = json.dumps(engine.configuration(), separators=(",", ":"))
        if state_dir is None:
            environment[pod_agent.ENGINE_ENV] = encoded
        else:
            encoded_b64 = base64.b64encode(encoded.encode()).decode()
            # The agent refuses a larger value after the pod is paid for:
            # refuse it here, before any create call.
            if len(encoded_b64) > pod_agent.ENGINE_B64_LIMIT:
                raise ValueError("engine startup selection is too large")
            environment[pod_agent.ENGINE_B64_ENV] = encoded_b64
            environment["ASTRID_POD_STATE"] = state_dir
    if policy is not None:
        environment[pod_agent.OBSERVATION_ENV] = policy.encoded
    return {"env": environment,
            "ports": f"{port}/http", "docker_args": "bash -lc " + shlex.quote("exec " + command),
            "docker_entrypoint": ["/bin/bash", "-lc"], "docker_start_cmd": ["exec " + command]}


def pod_url(pod_id: str, port: int = PORT) -> str:
    if not re.fullmatch(r"[a-z0-9]+", pod_id) or not 1 <= port <= 65535:
        raise ValueError("invalid pod ID or port")
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def _is_loopback(url: str) -> bool:
    host = urlsplit(url).hostname or ""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def _detached_directory(root: str, job_id: str) -> str:
    if not isinstance(root, str) or not root.startswith("/") or any(c in root for c in "\x00\r\n") \
            or ".." in root.split("/"):
        raise ValueError("detached root must be an absolute path without '..'")
    if not isinstance(job_id, str) or not _DETACHED_ID.fullmatch(job_id):
        raise ValueError("invalid detached job ID")
    return f"{root.rstrip('/')}/{job_id}"


def _detached_record(job_id: str, directory: str, fields: dict) -> dict:
    exit_code, pid = fields.get("exit_code", ""), fields.get("pid", "")
    return {"id": job_id, "dir": directory, "state": fields.get("state", "missing"),
            "exit_code": int(exit_code) if exit_code.isdigit() else None,
            "pid": int(pid) if pid.isdigit() else None}


def _transport_url(pod_id: str, port: int, override: str | None) -> str:
    expected = pod_url(pod_id, port)
    url = expected if override is None else override
    error = "HTTPS required at the exact pod proxy origin (HTTP allowed only for local tests)"
    # Reject syntax that URL parsers may silently normalize, including empty
    # query/fragment delimiters. The agent has no base-path/prefix contract.
    if not isinstance(url, str) or any(ord(c) <= 32 or ord(c) == 127 for c in url) or any(c in url for c in "\\?#"):
        raise ValueError(error)
    try:
        parsed = urlsplit(url)
        host, url_port = parsed.hostname, parsed.port
    except ValueError:
        raise ValueError(error) from None
    try:
        loopback = ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        loopback = host == "localhost"
    if (parsed.username is not None or parsed.password is not None
            or parsed.path not in ("", "/") or parsed.netloc.endswith(":")
            or parsed.scheme not in ("http", "https")
            or (not loopback and (parsed.scheme != "https"
                or host != urlsplit(expected).hostname or url_port not in (None, 443)))):
        raise ValueError(error)
    return url


class HttpsPodTransport:
    """Command, job, file and readiness operations with bounded HTTP requests."""

    def __init__(self, pod_id: str, token: str, *, port: int = PORT, base_url: str | None = None,
                 request_timeout: float = 20, poll_interval: float = 0.5,
                 route_grace: float = ROUTE_GRACE_SECONDS, deadline: float | None = None):
        url = _transport_url(pod_id, port, base_url)
        if len(token) < 32 or not token.isascii():
            raise ValueError("missing random pod token")
        self.poll_interval = poll_interval
        self.route_grace = route_grace
        # Optional absolute local time.monotonic() caller deadline. It bounds
        # every actual request wait and every retry/route-grace sleep.
        self.deadline = deadline
        self.request_timeout = min(request_timeout, 30)
        # The exact pod origin keeps the supported controller egress
        # (environment proxy/CA configuration), as in the live-qualified
        # transport. Only a loopback base (a route-local gateway or test agent,
        # already restricted by _transport_url) must never leave the host.
        self.http = httpx.Client(base_url=url, headers={"Authorization": "Bearer " + token},
                                 timeout=self.request_timeout, follow_redirects=False,
                                 trust_env=not _is_loopback(url))

    def _remaining(self):
        if self.deadline is None:
            return None
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise PodTransportError("pod HTTPS caller deadline exceeded")
        return remaining

    def _pause(self, seconds):
        remaining = self._remaining()
        time.sleep(seconds if remaining is None else min(seconds, remaining))

    @staticmethod
    def route_unavailable(response) -> bool:
        """True for the provider proxy's no-route reply, never for an agent reply."""
        return response.status_code == 404 and not response.content

    def _request(self, method, path, *, _download=None, _route_grace=None, _absent=(), **kwargs):
        # All operations are idempotent: command IDs and upload offsets make
        # retry after a lost response safe. Never follow redirects with auth.
        # Statuses listed in _absent are returned to the caller, not raised.
        grace = self.route_grace if _route_grace is None else _route_grace
        route_deadline = None
        attempt = 0
        while True:
            remaining = self._remaining()
            if remaining is not None:
                kwargs["timeout"] = min(self.request_timeout, remaining)
            try:
                response = self.http.request(method, path, **kwargs)
            except httpx.RequestError:
                if attempt < 2:
                    attempt += 1
                    self._pause(self.poll_interval)
                    continue
                message = "pod HTTPS request failed"
                if _download is not None:
                    framing, offset = _download
                    message += f" (operation=download framing={framing} offset={offset} HTTP unavailable)"
                self._log_upload_failure(method, path)
                raise PodTransportError(message) from None
            if self.route_unavailable(response) and grace > 0:
                # The request never reached the agent; wait for the route.
                now = time.monotonic()
                if route_deadline is None:
                    route_deadline = now + grace
                    logger.warning("pod HTTPS route unavailable (HTTP 404 without agent body); retrying")
                if now < route_deadline:
                    self._pause(min(max(self.poll_interval, 0.05), max(0, route_deadline - now)))
                    continue
            elif response.status_code in {502, 503, 504} and attempt < 2:
                attempt += 1
                self._pause(self.poll_interval)
                continue
            if not response.is_success and response.status_code not in _absent:
                self._log_upload_failure(method, path, response.status_code)
                if _download is not None:
                    framing, offset = _download
                    raise self._download_error("request rejected", response, framing, offset)
                detail = self._rate_limit_detail(response)
                raise PodTransportError(f"pod HTTPS request rejected (HTTP {response.status_code}{detail})")
            return response

    def _rate_limit_detail(self, response) -> str:
        """Sanitized body and Retry-After for HTTP 429 only; empty otherwise.

        The bearer token is redacted before the body is shortened, so no part of
        it can survive truncation. No retry or backoff happens here.
        """
        if response.status_code != 429:
            return ""
        secret = self.http.headers.get("Authorization", "").removeprefix("Bearer ")

        def clean(text: str, limit: int) -> str:
            if secret:
                text = text.replace(secret, "[redacted]")
            text = "".join(char if char.isprintable() else " " for char in text)
            return " ".join(text.split())[:limit]

        parts = []
        body = clean(response.content.decode("utf-8", "replace"), 200)
        if body:
            parts.append(f"body: {body}")
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            parts.append(f"Retry-After: {clean(retry_after, 40)}")
        return "; " + "; ".join(parts) if parts else ""

    @staticmethod
    def _log_upload_failure(method, path, status=None):
        # Fixed local labels only: never log paths/IDs, metadata, exception
        # text or response data. Keep the v1 exception/receipt formats intact.
        stage = None
        if method == "POST" and path == "/uploads":
            stage = "initiate"
        elif isinstance(path, str) and re.fullmatch(r"/uploads/[a-f0-9]{32,64}", path):
            stage = {"PUT": "chunk", "POST": "verify", "GET": "status"}.get(method)
        if stage:
            logger.warning("pod HTTPS upload failed (stage=%s HTTP %s)", stage,
                           status if type(status) is int and 100 <= status <= 599 else "unavailable")

    @staticmethod
    def _download_error(reason, response, framing, offset, *, received=None):
        # Only local offsets, status and byte counts are diagnostic. Never
        # interpolate remote paths, response values, headers, bodies or ETags.
        detail = f"operation=download framing={framing} offset={offset} HTTP {response.status_code}"
        if received is not None:
            detail += f" received_bytes={received}"
        return PodTransportError(f"{reason} ({detail})")

    def connect(self, *, _route_grace=None):
        data = self._request("GET", "/health", _route_grace=_route_grace).json()
        if data != {"ok": True, "protocol": 1}:
            raise PodTransportError("incompatible pod agent")

    def wait_ready(self, timeout: float = 600):
        """Return after READY_CONSECUTIVE successive authenticated health replies.

        A single 200 does not mean the provider route is settled; later requests
        still ride out brief no-route gaps via route_grace in _request.
        """
        deadline = time.monotonic() + timeout
        healthy = 0
        while True:
            try:
                self.connect(_route_grace=0)
                healthy += 1
                if healthy >= READY_CONSECUTIVE:
                    return
            except PodTransportError as exc:
                if "HTTP 401" in str(exc) or "HTTP 403" in str(exc):
                    raise
                healthy = 0
                if time.monotonic() >= deadline:
                    # Fixed labels only: was the provider route missing, the
                    # connection failing, or did something else answer?
                    status = re.search(r"\(HTTP ([0-9]{3})\)", str(exc))
                    last = ("route_unavailable" if status and status[1] == "404"
                            else f"HTTP {status[1]}" if status
                            else "request_failed" if "request failed" in str(exc) else "incompatible")
                    logger.warning("pod HTTPS readiness timed out (last=%s)", last)
                    raise PodTransportError("pod HTTPS readiness timed out") from None
            time.sleep(min(self.poll_interval, max(0, deadline - time.monotonic())))

    def start(self, command: str, *, job_id: str | None = None) -> str:
        job_id = job_id or uuid.uuid4().hex
        return self._request("POST", "/jobs", json={"id": job_id, "command": command}).json()["id"]

    def poll(self, job_id: str, stdout_offset: int = 0, stderr_offset: int = 0) -> dict:
        return self._request("GET", "/jobs/" + job_id,
                             params={"stdout": stdout_offset, "stderr": stderr_offset}).json()

    def cancel(self, job_id: str):
        self._request("DELETE", "/jobs/" + job_id)

    def execute_command(self, command: str, timeout: float = 600) -> tuple[int, str, str]:
        job_id = self.start(command)
        deadline = time.monotonic() + timeout
        stdout, stderr = bytearray(), bytearray()
        out_offset = err_offset = 0
        finished = False
        try:
            while True:
                result = self.poll(job_id, out_offset, err_offset)
                stdout.extend(base64.b64decode(result["stdout"]))
                stderr.extend(base64.b64decode(result["stderr"]))
                out_offset, err_offset = result["stdout_offset"], result["stderr_offset"]
                if result["status"] == "exited" and result["stdout_eof"] and result["stderr_eof"]:
                    finished = True
                    return result["exit_code"], stdout.decode(errors="replace"), stderr.decode(errors="replace")
                if time.monotonic() >= deadline:
                    return -1, stdout.decode(errors="replace"), stderr.decode(errors="replace") + f"\nCommand timed out after {timeout} seconds"
                # Drain buffered output immediately; sleep only at the live tail.
                if result["stdout_eof"] and result["stderr_eof"]:
                    time.sleep(min(self.poll_interval, max(0, deadline - time.monotonic())))
        finally:
            if not finished:
                self.cancel(job_id)

    run = exec = execute_command

    def _detached_script(self, script: str, timeout: float) -> dict:
        code, stdout, _stderr = self.execute_command(script, timeout=timeout)
        if code != 0:
            raise PodTransportError(f"detached job control failed (exit {code})")
        return dict(line.split("=", 1) for line in stdout.splitlines() if "=" in line)

    def detached_start(self, command: str, *, root: str = DETACHED_ROOT, job_id: str | None = None,
                       launch_timeout: float = 60) -> dict:
        """Run command detached on the volume; return {"id", "dir"}.

        The agent job used here only spawns the supervisor and exits. It is never
        cancelled, so a client timeout while waiting for it leaves the work running.
        """
        job_id = uuid.uuid4().hex if job_id is None else job_id
        directory = _detached_directory(root, job_id)
        if not isinstance(command, str) or not command.strip() or "\x00" in command:
            raise ValueError("detached command must be a non-empty string")
        encoded = base64.b64encode(command.encode()).decode()
        launcher = (
            "set -eu\n"
            f"d={shlex.quote(directory)}\n"
            'if [ -e "$d" ]; then exit 3; fi\n'
            "umask 077\n"
            'mkdir -p -- "$d"\n'
            f'printf %s {shlex.quote(encoded)} | base64 -d > "$d/command.sh"\n'
            'printf starting > "$d/status"\n'
            f'python3 -c {shlex.quote("import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])")} '
            f'bash -c {shlex.quote(_DETACHED_SUPERVISOR)} detached-job "$d" </dev/null >/dev/null 2>&1 &\n'
        )
        launch = self.start(launcher)
        deadline = time.monotonic() + launch_timeout
        while True:
            result = self.poll(launch)
            if result["status"] == "exited":
                break
            if time.monotonic() >= deadline:
                raise PodTransportError("detached job launch did not finish")
            time.sleep(self.poll_interval)
        if result["exit_code"] != 0:
            raise PodTransportError(f"detached job launch failed (exit {result['exit_code']})")
        return {"id": job_id, "dir": directory}

    def _detached_file(self, path: str) -> str | None:
        """Text of one job-state file, or None when the agent cannot open it (HTTP 400).

        Reads the JSON chunk envelope, not a Range GET: engine-enabled agents
        refuse every Range request. State files are a few bytes, so one chunk
        must hold the whole file; the envelope's total and SHA256 confirm it.
        """
        response = self._request("POST", "/download-chunk", json={"path": path, "offset": 0},
                                 _absent=(400,))
        if response.status_code == 400:
            return None
        try:
            data = response.json()
            chunk = base64.b64decode(data["data"], validate=True)
            valid = (type(data["total"]) is int and data["total"] == len(chunk)
                     and hashlib.sha256(chunk).hexdigest() == data["sha256"])
        except (ValueError, KeyError, TypeError):
            valid = False
        if not valid:
            raise PodTransportError("pod HTTPS job state read failed")
        return chunk.decode(errors="replace")

    def detached_poll(self, job_id: str, *, root: str = DETACHED_ROOT) -> dict:
        """Read a detached job's state without creating an agent job record.

        Status is read before exit on purpose: the supervisor writes exit before
        its final "exited" status, so an "exited" read is always followed by an
        exit file. Same fields and precedence as the former shell read (exit wins).
        """
        directory = _detached_directory(root, job_id)
        status = self._detached_file(f"{directory}/status")
        exit_text = self._detached_file(f"{directory}/exit")
        pid = self._detached_file(f"{directory}/pid")
        if exit_text is not None:
            state, code = "exited", exit_text.rstrip("\n")
        else:
            state, code = (status.rstrip("\n") if status is not None else "missing"), ""
        return _detached_record(job_id, directory,
                                {"state": state, "exit_code": code, "pid": (pid or "").rstrip("\n")})

    def detached_cancel(self, job_id: str, *, root: str = DETACHED_ROOT) -> dict:
        """Signal the recorded process group; a finished job keeps its exit code."""
        directory = _detached_directory(root, job_id)
        script = (
            f"d={shlex.quote(directory)}\n"
            'if [ ! -d "$d" ] || [ -f "$d/exit" ]; then exit 0; fi\n'
            ': > "$d/cancel"\n'
            'pid=$(cat "$d/pid" 2>/dev/null || true)\n'
            'case "$pid" in ""|*[!0-9]*|0|1) pid= ;; esac\n'
            'if [ -n "$pid" ]; then\n'
            '  kill -TERM -- "-$pid" 2>/dev/null || true\n'
            "  for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 -- \"-$pid\" 2>/dev/null || break; sleep 0.2; done\n"
            '  kill -KILL -- "-$pid" 2>/dev/null || true\n'
            "fi\n"
            'if [ ! -f "$d/exit" ] && [ -d "$d" ]; then echo cancelled > "$d/status.tmp" && mv "$d/status.tmp" "$d/status"; fi\n'
        )
        self._detached_script(script, timeout=30)
        return self.detached_poll(job_id, root=root)

    def put(self, local: str | Path, remote: str, *, reserved: bool = False):
        local = Path(local)
        with local.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else pod_agent.digest_file(local)
        metadata = {"path": remote, "size": local.stat().st_size, "sha256": digest, "reserved": reserved}
        state = self._request("POST", "/uploads", json=metadata).json()
        upload_id = state["id"]
        with local.open("rb") as stream:
            while not state["complete"] and state["offset"] < metadata["size"]:
                stream.seek(state["offset"])
                chunk = stream.read(CHUNK)
                try:
                    state = self._request("PUT", "/uploads/" + upload_id,
                                          params={"offset": state["offset"]}, content=chunk).json()
                except PodTransportError as chunk_error:
                    # A committed chunk may have lost its reply. Reconcile its
                    # durable offset before sending any bytes again.
                    try:
                        state = self._request("GET", "/uploads/" + upload_id).json()
                    except PodTransportError:
                        # Keep the initiating rejection when reconciliation
                        # also fails. Neither failure permits another write.
                        raise chunk_error from None
                    if state["offset"] == stream.tell() - len(chunk):
                        raise
        if not state["complete"]:
            state = self._request("POST", "/uploads/" + upload_id, content=b"").json()
            deadline = time.monotonic() + 3600
            while not state["complete"]:
                if time.monotonic() >= deadline:
                    raise PodTransportError("upload verification timed out; resume the same file")
                self._pause(self.poll_interval)
                state = self._request("GET", "/uploads/" + upload_id).json()
                if state.get("error"):
                    raise PodTransportError("upload verification failed")

    def get(self, remote: str, local: str | Path, *, use_ranges: bool = False):
        """Download bounded chunks; use_ranges opts into HTTP 206 framing.

        JSON envelopes are the default through public proxies so byte offsets,
        representation identity and SHA256 remain in the authenticated body.
        """
        local = Path(local)
        staging = local.with_name(local.name + ".https-part")
        offset, etag = 0, None
        with staging.open("wb") as stream:
            while True:
                if not use_ranges:
                    request = {"path": remote, "offset": offset}
                    if etag:
                        request["etag"] = etag
                    response = self._request("POST", "/download-chunk", json=request,
                                             _download=("json", offset))
                    try:
                        data = response.json()
                        chunk = base64.b64decode(data["data"], validate=True)
                        valid = (type(data["offset"]) is int and data["offset"] == offset
                                 and type(data["total"]) is int and offset <= data["total"]
                                 and len(chunk) <= CHUNK and isinstance(data["etag"], str) and data["etag"]
                                 and (not etag or data["etag"] == etag)
                                 and hashlib.sha256(chunk).hexdigest() == data["sha256"]
                                 and offset + len(chunk) <= data["total"]
                                 and (chunk or offset == data["total"]))
                    except (ValueError, KeyError, TypeError):
                        valid = False
                    if not valid:
                        raise self._download_error("invalid download chunk or changed file", response,
                                                   "json", offset) from None
                    stream.write(chunk)
                    offset += len(chunk)
                    etag = data["etag"]
                    if offset == data["total"]:
                        break
                    continue
                headers = {"Range": f"bytes={offset}-{offset + CHUNK - 1}"}
                if etag:
                    headers["If-Match"] = etag
                response = self._request("GET", "/files", params={"path": remote}, headers=headers,
                                         _download=("range", offset))
                match = re.fullmatch(r"bytes (\d+)-(-?\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if (response.status_code != 206 or not match or int(match[1]) != offset
                        or not response.headers.get("ETag")
                        or (etag and response.headers.get("ETag") != etag)):
                    raise self._download_error("invalid download range or changed file", response,
                                               "range", offset, received=len(response.content))
                total = int(match[3])
                length = max(0, int(match[2]) - offset + 1)
                if (len(response.content) != length or length > CHUNK
                        or offset + length > total or (not response.content and total)
                        or (length == 0 and (offset != 0 or total != 0 or int(match[2]) != -1))):
                    raise self._download_error("short download or invalid range size", response,
                                               "range", offset, received=len(response.content))
                stream.write(response.content)
                offset += len(response.content)
                etag = response.headers["ETag"]
                if offset == total:
                    break
        staging.replace(local)

    def disconnect(self):
        self.http.close()

    close = disconnect

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_args):
        self.disconnect()


def read_token_file(path: str | Path) -> str:
    """Load a local bearer without following symlinks or accepting shared files."""
    import os
    import stat
    raw = str(path)
    selected = Path(raw)
    if (not selected.is_absolute() or str(selected) != raw
            or any(p in ("", ".", "..") for p in raw.split("/")[1:])):
        raise ValueError("pod token file path is unsafe")
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        for part in selected.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=directory)
            os.close(directory)
            directory = child
        fd = os.open(selected.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                     dir_fd=directory)
    finally:
        os.close(directory)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_uid != os.getuid() or info.st_nlink != 1):
            raise ValueError("pod token file must be an owned regular 0600 file")
        token = os.read(fd, 256).decode().strip()
        if len(token) < 32 or len(token) >= 256 or not token.isascii():
            raise ValueError("invalid pod token file")
        return token
    finally:
        os.close(fd)


def write_token_file(path: str | Path, token: str) -> None:
    import os
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(token)
        stream.flush()
        os.fsync(stream.fileno())
