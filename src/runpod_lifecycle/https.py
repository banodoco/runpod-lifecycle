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
logger = logging.getLogger(__name__)


class PodTransportError(RuntimeError):
    """Deliberately contains no request bodies, headers or token material."""


def agent_bootstrap(token: str, port: int = PORT, *, observation_policy=None) -> dict:
    """Create SDK/REST-independent boot material. The token is only in env.

    Inlining the compressed single-file agent avoids mutable download URLs and
    any package installs on stock runpod/base images.

    observation_policy is an optional public astrid.pod-observation.v1 mapping
    with roots (up to 8 curated directories) and files (up to 32 exact paths).
    It restricts the agent to authenticated health/downloads. Its base64 JSON
    env value is capped at 16384 ASCII bytes and consumed at agent startup.
    None preserves the general command/file agent.
    """
    import zlib
    if len(token) < 32 or not token.isascii():
        raise ValueError("claim token must have at least 32 ASCII characters")
    policy = (None if observation_policy is None
              else pod_agent.ObservationPolicy(observation_policy))
    source = Path(pod_agent.__file__).read_bytes()
    payload = base64.b64encode(zlib.compress(source, 9)).decode()
    loader = ("import os,base64,zlib; "
              "exec(compile(zlib.decompress(base64.b64decode(os.environ.pop('ASTRID_POD_AGENT_B64'))),"
              "'<astrid-pod-agent>','exec'))")
    command = "python3 -u -c " + shlex.quote(loader)
    environment = {TOKEN_ENV: token, "ASTRID_POD_AGENT_B64": payload, "ASTRID_POD_PORT": str(port)}
    if policy is not None:
        environment[pod_agent.OBSERVATION_ENV] = policy.encoded
    return {"env": environment,
            "ports": f"{port}/http", "docker_args": "bash -lc " + shlex.quote("exec " + command),
            "docker_entrypoint": ["/bin/bash", "-lc"], "docker_start_cmd": ["exec " + command]}


def pod_url(pod_id: str, port: int = PORT) -> str:
    if not re.fullmatch(r"[a-z0-9]+", pod_id) or not 1 <= port <= 65535:
        raise ValueError("invalid pod ID or port")
    return f"https://{pod_id}-{port}.proxy.runpod.net"


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
                 request_timeout: float = 20, poll_interval: float = 0.5):
        url = _transport_url(pod_id, port, base_url)
        if len(token) < 32 or not token.isascii():
            raise ValueError("missing random pod token")
        self.poll_interval = poll_interval
        self.http = httpx.Client(base_url=url, headers={"Authorization": "Bearer " + token},
                                 timeout=min(request_timeout, 30), follow_redirects=False)

    def _request(self, method, path, *, _download=None, **kwargs):
        # All operations are idempotent: command IDs and upload offsets make
        # retry after a lost response safe. Never follow redirects with auth.
        for attempt in range(3):
            try:
                response = self.http.request(method, path, **kwargs)
            except httpx.RequestError:
                if attempt < 2:
                    time.sleep(self.poll_interval)
                    continue
                message = "pod HTTPS request failed"
                if _download is not None:
                    framing, offset = _download
                    message += f" (operation=download framing={framing} offset={offset} HTTP unavailable)"
                self._log_upload_failure(method, path)
                raise PodTransportError(message) from None
            if response.status_code in {502, 503, 504} and attempt < 2:
                time.sleep(self.poll_interval)
                continue
            if not response.is_success:
                self._log_upload_failure(method, path, response.status_code)
                if _download is not None:
                    framing, offset = _download
                    raise self._download_error("request rejected", response, framing, offset)
                raise PodTransportError(f"pod HTTPS request rejected (HTTP {response.status_code})")
            return response
        raise AssertionError("unreachable")

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

    def connect(self):
        data = self._request("GET", "/health").json()
        if data != {"ok": True, "protocol": 1}:
            raise PodTransportError("incompatible pod agent")

    def wait_ready(self, timeout: float = 600):
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.connect()
                return
            except PodTransportError as exc:
                if "HTTP 401" in str(exc) or "HTTP 403" in str(exc):
                    raise
                if time.monotonic() >= deadline:
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
                time.sleep(self.poll_interval)
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
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
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
