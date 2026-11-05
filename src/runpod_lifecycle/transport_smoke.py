"""Durable, single-allocation qualification of the HTTPS pod transport.

Provider mutations belong to api.py; this command adds custody, evidence and an
independent exact-ID REST absence check. No credentials or remote output enter
its receipt. Only the owned token sidecar contains the agent bearer.
"""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import secrets
import signal
import stat
import sys
import tempfile
import threading
import time
import uuid

import httpx

from . import api
from .https import HttpsPodTransport, PodTransportError, write_token_file
from .preflight import preflight_transport

MIB = 1024 * 1024
IMAGE = "runpod/base:1.0.3-dev-fix-pytorch-version-verification-cuda1300-ubuntu2404"
LONG_SECONDS = 102
REQUEST_SECONDS = 10


class SmokeError(RuntimeError):
    """A fixed, secret-free operator diagnostic."""


class SmokeInterrupted(BaseException):
    pass


class SmokeTimeout(BaseException):
    """Cannot be swallowed by provider/client Exception retry handlers."""


def _stamp(clock):
    return datetime.fromtimestamp(clock.time(), timezone.utc).isoformat()


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _transport_failure(exc):
    # PodTransportError currently has no typed metadata. Match only complete
    # locally generated formats from https.py; never retain exception text or
    # accept an arbitrary reason, framing, header/body suffix or remote path.
    message = str(exc)
    context = r"\(operation=(?P<operation>download) framing="
    offset = r" offset=(?P<offset>[0-9]{1,19}) HTTP "
    status = r"(?P<http_status>[1-5][0-9]{2})"
    received = r" received_bytes=(?P<received_bytes>[0-9]{1,19})"
    formats = (
        (r"pod HTTPS request rejected \(HTTP " + status + r"\)", "http_rejected"),
        ("request rejected " + context + r"(?P<framing>json|range)" + offset + status + r"\)",
         "http_rejected"),
        ("invalid download chunk or changed file " + context + r"(?P<framing>json)"
         + offset + status + r"\)", "transport_error"),
        (r"(?:invalid download range or changed file|short download or invalid range size) "
         + context + r"(?P<framing>range)" + offset + status + received + r"\)", "transport_error"),
        ("pod HTTPS request failed " + context + r"(?P<framing>json|range)"
         + offset + r"unavailable\)", "transport_error"),
    )
    for pattern, failure in formats:
        match = re.fullmatch(pattern, message)
        if match:
            metadata = {"transport_failure": failure}
            for key, value in match.groupdict().items():
                if key in {"offset", "received_bytes", "http_status"}:
                    value = int(value)
                    if value > 2**63 - 1:
                        return {"transport_failure": "transport_error"}
                metadata["transport_" + key] = value
            return metadata
    return {"transport_failure": "transport_error"}


def _owned_fd(path, flags, mode=0o600):
    fd = os.open(path, flags | os.O_NOFOLLOW, mode)
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.getuid() or info.st_nlink != 1):
        os.close(fd)
        raise SmokeError("unsafe local custody file")
    return fd


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _save(path, record):
    # Atomic replacement and directory fsync retain both intent and identity
    # across crashes. The process lock lives in a separate, stable inode.
    fd, temporary = tempfile.mkstemp(prefix=".smoke-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def _lock(path):
    import fcntl
    fd = _owned_fd(path, os.O_CREAT | os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SmokeError("another transport smoke owns this lock") from None
        yield
    finally:
        os.close(fd)


@contextlib.contextmanager
def _quiet_provider():
    # Older lifecycle SDK fallbacks log arbitrary exception strings. Do not
    # let a provider response (possibly including env) escape this command.
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    finally:
        logging.disable(previous)


@contextlib.contextmanager
def _https_diagnostics_to_stderr():
    # Only the HTTPS client's fixed-label warnings (stage/status/route labels;
    # never paths, IDs, headers, bodies or tokens) are emitted, on stderr.
    https_logger = logging.getLogger("runpod_lifecycle.https")
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    previous = https_logger.propagate, https_logger.level
    https_logger.addHandler(handler)
    https_logger.propagate = False
    if https_logger.level == logging.NOTSET or https_logger.level > logging.WARNING:
        https_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        https_logger.removeHandler(handler)
        https_logger.propagate, level = previous
        https_logger.setLevel(level)


class Provider:
    """Existing lifecycle mutations plus strict read-only REST verification."""

    def __init__(self, api_key):
        self.api_key = api_key
        self.http = httpx.Client(headers={"Authorization": "Bearer " + api_key},
                                 timeout=REQUEST_SECONDS, follow_redirects=False)

    def pods(self):
        with _quiet_provider():
            return api.list_pods(self.api_key)

    def create(self, name, token):
        # The standard REST CPU path supports minimum vcpu_count=2 and its
        # default cpu3c/cpu5c flavors. Never invoke GPU or capacity fallbacks.
        with _quiet_provider():
            return api.create_pod(self.api_key, None, IMAGE, name=name,
                                  compute_type="CPU", cpu_api="rest", vcpu_count=2,
                                  cpu_flavor_ids=("cpu3c", "cpu5c"),
                                  cpu_flavor_priority="custom", network_volume_id=None,
                                  disk_in_gb=0, container_disk_in_gb=5,
                                  transport="https", https_token=token)

    def inspect(self, pod_id):
        response = self.http.get(api.PODS_URL + "/" + pod_id)
        if response.status_code == 404:
            return 404, None
        if response.status_code != 200:
            raise SmokeError("provider exact-ID lookup failed")
        data = response.json()
        if not isinstance(data, dict) or data.get("id") != pod_id:
            raise SmokeError("provider exact-ID response is invalid")
        return 200, data

    def terminate(self, pod_id):
        with _quiet_provider():
            api.terminate_pod(pod_id, self.api_key, verify_timeout_seconds=0,
                              poll_interval_seconds=0)

    def close(self):
        self.http.close()


class _Budget:
    """Bound even blocking put/get and provider calls, including retries.

    SIGALRM is deliberately required: per-request timeouts alone cannot bound
    repeated chunk transfers or a hung provider SDK. Cleanup has its own reserve.
    """

    def __init__(self, seconds, clock):
        self.clock = clock
        self.end = clock.monotonic() + seconds

    def remaining(self):
        remaining = self.end - self.clock.monotonic()
        if remaining <= 0:
            raise SmokeTimeout
        return remaining

    def arm(self):
        signal.setitimer(signal.ITIMER_REAL, self.remaining())


@contextlib.contextmanager
def _signals():
    if threading.current_thread() is not threading.main_thread():
        raise SmokeError("transport smoke must run in the main thread")
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise SmokeError("transport smoke requires an unused process alarm")
    def alarm(*_):
        raise SmokeTimeout
    def interrupt(*_):
        raise SmokeInterrupted
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM)}
    try:
        signal.signal(signal.SIGALRM, alarm)
        signal.signal(signal.SIGINT, interrupt)
        signal.signal(signal.SIGTERM, interrupt)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in old.items():
            signal.signal(sig, handler)


def _load(path):
    with os.fdopen(_owned_fd(path, os.O_RDONLY)) as stream:
        record = json.load(stream)
    if (record.get("schema") != "runpod-transport-smoke-v1"
            or not re.fullmatch(r"astrid-https-smoke-[0-9a-f]{32}", record.get("request_name", ""))
            or (record.get("pod_id") is not None
                and not re.fullmatch(r"[a-z0-9]+", record["pod_id"]))
            or not isinstance(record.get("create_attempts"), int)
            or record["create_attempts"] not in (0, 1)):
        raise SmokeError("invalid transport smoke handle")
    return record


def _identity(record, pod):
    pod_id = pod.get("id") if isinstance(pod, dict) else None
    if not isinstance(pod_id, str) or not re.fullmatch(r"[a-z0-9]+", pod_id):
        raise SmokeError("create returned no valid pod identity; reconciliation required")
    record["pod_id"] = pod_id


def _cleanup(provider, record, save, clock, budget):
    """Never convert an empty collection after ambiguous create into success."""
    record["cleanup"] = "pending"
    record["qualified"] = False
    for key in ("cleanup_get_status", "released_at", "lifetime_upper_bound_seconds",
                "estimated_cost_upper_bound", "cleanup_failure"):
        record.pop(key, None)
    record["cleanup_attempted_at"] = _stamp(clock)
    # An already durable allocation intent still authorizes release if local
    # storage fills after create. Do not let an evidence-write failure leak it.
    with contextlib.suppress(OSError):
        save()
    if not record.get("pod_id"):
        matches = [p for p in provider.pods() if p.get("name") == record["request_name"]]
        record["reconciliation_matches"] = len(matches)
        with contextlib.suppress(OSError):
            save()
        if len(matches) != 1:
            # A delayed create may appear after an empty list. Keep custody
            # pending and require cleanup-only again; never create a second pod.
            raise SmokeError("allocation unresolved; retry cleanup-only with this handle")
        _identity(record, matches[0])
        record["identity_recorded_at"] = _stamp(clock)
        with contextlib.suppress(OSError):
            save()
    pod_id = record["pod_id"]
    status, pod = provider.inspect(pod_id)
    if status != 404:
        if pod.get("name") != record["request_name"]:
            raise SmokeError("pod name differs from allocation intent; refusing deletion")
        try:
            provider.terminate(pod_id)
        except Exception:
            # A lost DELETE reply is not evidence of absence. GET below decides.
            pass
        while True:
            budget.remaining()
            status, _ = provider.inspect(pod_id)
            if status == 404:
                break
            clock.sleep(min(2, budget.remaining()))
    record.update(cleanup="verified_absent", cleanup_get_status=404,
                  released_at=_stamp(clock))
    if record.get("intent_epoch") is not None:
        elapsed = max(0, clock.time() - record["intent_epoch"],
                      record.get("lifetime_seconds_observed", 0))
        record["lifetime_upper_bound_seconds"] = math.ceil(elapsed)
        rate = record.get("observed_hourly")
        if _number(rate):
            record["estimated_cost_upper_bound"] = rate * math.ceil(elapsed) / 3600
    save()


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _roundtrip(transport, record, save, clock, budget, evidence, payload_bytes):
    record["phase"] = "health"
    save()
    transport.wait_ready(timeout=min(240, budget.remaining()))
    record.update(agent_ready_at=_stamp(clock), protocol=1,
                  request_timeout_seconds=REQUEST_SECONDS)
    save()
    record["phase"] = "uname"
    save()
    code, _, _ = transport.execute_command("uname -a", timeout=min(30, budget.remaining()))
    if code != 0:
        raise SmokeError("uname failed")
    record["uname"] = "passed"  # Remote stdout may include secrets; never retain it.
    record["phase"] = "payload"
    save()
    source, destination = evidence / "source.bin", evidence / "download.bin"
    with source.open("xb") as stream:
        remaining = payload_bytes
        while remaining:
            budget.remaining()
            chunk = secrets.token_bytes(min(MIB, remaining))
            stream.write(chunk)
            remaining -= len(chunk)
        stream.flush()
        os.fsync(stream.fileno())
    sha256 = _digest(source)
    record.update(payload_bytes=payload_bytes, source_sha256=sha256, phase="upload")
    save()
    started = clock.monotonic()
    remote = "/tmp/" + record["request_name"] + ".bin"
    transport.put(source, remote)
    record.update(upload_completed_at=_stamp(clock), phase="download")
    save()
    budget.remaining()
    transport.get(remote, destination)  # Default JSON envelopes; no Range override.
    if destination.stat().st_size != payload_bytes or _digest(destination) != sha256:
        raise SmokeError("round-trip size or SHA256 mismatch")
    record["transfer_receipt"] = {
        "bytes": payload_bytes, "sha256": sha256, "full_sha256_verified": True,
        "download_mode": "json-envelope", "seconds": clock.monotonic() - started,
        "completed_at": _stamp(clock),
    }
    record["phase"] = "long_job"
    save()
    if budget.remaining() < LONG_SECONDS + 15:
        record["long_job"] = {"status": "skipped", "reason": "insufficient_remaining_budget"}
    else:
        job = transport.start("sleep 102", job_id=uuid.uuid4().hex)
        started = clock.monotonic()
        record["long_job"] = {"status": "running", "id": job, "started_at": _stamp(clock)}
        save()
        # Each poll is a short HTTP request; no 102-second HTTP request.
        while True:
            budget.remaining()
            result = transport.poll(job)
            if result["status"] == "exited":
                elapsed = clock.monotonic() - started
                if result["exit_code"] != 0 or elapsed <= 100:
                    raise SmokeError("long async command did not pass the 100-second window")
                record["long_job"].update(status="passed", seconds=elapsed,
                                          completed_at=_stamp(clock))
                break
            clock.sleep(min(1, budget.remaining()))
    record.update(status="passed", phase="complete", completed_at=_stamp(clock))
    save()


def run_smoke(*, handle, api_key, max_hourly=None, max_seconds=600,
              payload_bytes=50 * MIB, cleanup_only=False, provider=None,
              transport_factory=HttpsPodTransport, clock=time):
    """Return a secret-free durable receipt. Existing handles never allocate.

    Dependencies permit offline fake-provider/clock and loopback-agent tests.
    Cleanup-only reads no bearer, and may be repeated until exact GET 404.
    """
    if type(max_seconds) is not int or not 1 <= max_seconds <= 900:
        raise SmokeError("max-seconds must be an integer from 1 to 900")
    if not cleanup_only and not _number(max_hourly):
        raise SmokeError("max-hourly must be a finite positive number")
    if type(payload_bytes) is not int or not 1 <= payload_bytes <= 1024 * MIB:
        raise SmokeError("payload-bytes must be from 1 to 1073741824")
    handle = Path(handle).absolute()
    # The caller chooses durable storage. Reject symlinked parents and paths
    # writable by another user so replacement cannot redirect custody writes.
    if handle.parent.resolve() != handle.parent:
        raise SmokeError("handle directory must not contain symlinks")
    info = handle.parent.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise SmokeError("handle directory must be owned and not writable by others")
    evidence = handle.with_name(handle.name + ".files")
    own_provider = provider is None
    # Serialize this user's local smoke invocations even across distinct handles.
    account_lock = Path(tempfile.gettempdir()) / f"runpod-transport-smoke-{os.getuid()}.lock"
    with _lock(account_lock), _lock(handle.with_name(handle.name + ".lock")), _signals():
        if cleanup_only:
            record = _load(handle)
        else:
            # O_EXCL is essential: a completed or unresolved handle is never fresh.
            fd = _owned_fd(handle, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            os.close(fd)
            record = {"schema": "runpod-transport-smoke-v1", "status": "preflight",
                      "request_name": "astrid-https-smoke-" + uuid.uuid4().hex,
                      "created_at": _stamp(clock), "create_attempts": 0, "pod_id": None,
                      "cleanup": "not_needed", "max_seconds": max_seconds,
                      "max_hourly": max_hourly, "payload_bytes": payload_bytes,
                      "token_file": str(evidence / "agent.token"), "qualified": False}
        def save():
            _save(handle, record)
        save()
        provider = provider if provider is not None else Provider(api_key)
        transport = None
        started = clock.monotonic()
        reserve = min(90, max_seconds / 3)
        work = _Budget(max_seconds if cleanup_only else max_seconds - reserve, clock)
        try:
            work.arm()
            if cleanup_only:
                return record  # finally performs provider-only cleanup
            preflight_transport("https", timeout=min(5, work.remaining()))
            pods = provider.pods()
            if not isinstance(pods, list) or pods:
                raise SmokeError("account already has pods or inventory is invalid; no allocation")
            evidence.mkdir(mode=0o700)
            token = secrets.token_urlsafe(32)
            write_token_file(evidence / "agent.token", token)
            _sync_dir(evidence)
            _sync_dir(evidence.parent)
            record.update(create_attempts=1, status="creating", phase="create",
                          intent_at=_stamp(clock), intent_epoch=clock.time(), cleanup="pending")
            save()  # Durable unique request name + intent BEFORE the single create.
            pod = provider.create(record["request_name"], token)
            _identity(record, pod)
            record.update(identity_recorded_at=_stamp(clock), status="running")
            save()  # fsync identity before pricing, transport, or any other I/O.
            status, observed = provider.inspect(record["pod_id"])
            rate = observed.get("costPerHr") if status == 200 else None
            if _number(rate):
                record["observed_hourly"] = rate
            record["price_observed_at"] = _stamp(clock)
            save()
            if not _number(rate) or rate > max_hourly:
                raise SmokeError("observed hourly price missing, invalid, or above max-hourly")
            transport = transport_factory(record["pod_id"], token, request_timeout=REQUEST_SECONDS,
                                          poll_interval=1)
            _roundtrip(transport, record, save, clock, work, evidence, payload_bytes)
        except BaseException as exc:
            record.update(status="failed", qualified=False, failed_at=_stamp(clock),
                          failure="interrupted" if isinstance(exc, (KeyboardInterrupt, SmokeInterrupted))
                          else "timeout" if isinstance(exc, (TimeoutError, SmokeTimeout)) else "operation_failed")
            # Only locally defined fixed diagnostics can enter the ordinary handle.
            if isinstance(exc, SmokeError):
                record["diagnostic"] = str(exc)
            if isinstance(exc, PodTransportError):
                record.update(_transport_failure(exc))
            save()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            # Repeated interrupts must not bypass release; cleanup's alarm still
            # bounds provider calls. SIGKILL/power loss require cleanup-only.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            if record["create_attempts"] and (cleanup_only or record.get("cleanup") != "verified_absent"):
                cleanup = _Budget(max_seconds if cleanup_only else reserve, clock)
                try:
                    cleanup.arm()
                    if not cleanup_only:
                        record["lifetime_seconds_observed"] = clock.monotonic() - started
                    _cleanup(provider, record, save, clock, cleanup)
                except BaseException:
                    record["cleanup"] = "pending"
                    record["cleanup_failure"] = "verification_incomplete; run cleanup-only"
                    save()
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
            if transport is not None:
                with contextlib.suppress(Exception):
                    transport.disconnect()
            if own_provider:
                with contextlib.suppress(Exception):
                    provider.close()
            record["qualified"] = (record["status"] == "passed"
                                   and record["cleanup"] == "verified_absent"
                                   and record.get("payload_bytes", 0) >= 50 * MIB
                                   and record.get("long_job", {}).get("status") == "passed")
            save()
            # Only remove generated evidence after BOTH digest success and
            # provider absence have been durably recorded. Failed runs retain it.
            if not cleanup_only and record["status"] == "passed" and record["cleanup"] == "verified_absent":
                for name in ("source.bin", "download.bin", "agent.token"):
                    with contextlib.suppress(FileNotFoundError):
                        (evidence / name).unlink()
                _sync_dir(evidence)
        return record


def register_parser(subparsers):
    parser = subparsers.add_parser("transport-smoke", help="Qualify HTTPS using one bounded CPU pod.")
    parser.add_argument("--handle", required=True, type=Path,
                        help="Durable new handle path; existing handles require --cleanup-only.")
    parser.add_argument("--max-seconds", type=int, default=600,
                        help="Total work plus cleanup budget, 1..900 seconds (default: 600).")
    parser.add_argument("--max-hourly", type=float,
                        help="Required positive observed hourly price cap for allocation.")
    parser.add_argument("--payload-bytes", type=int, default=50 * MIB,
                        help="Round-trip bytes, default 52428800 (50 MiB), maximum 1 GiB.")
    parser.add_argument("--cleanup-only", action="store_true",
                        help="Reconcile/release this handle only; no agent token needed.")


async def command(args):
    """CLI boundary: arbitrary provider exceptions must never reach main()."""
    try:
        key = getattr(args, "api_key", None) or os.environ.get("RUNPOD_API_KEY")
        if not key:
            raise SmokeError("RUNPOD_API_KEY is required")
        # Provider/SDK calls are individually quieted inside Provider. Do not
        # disable logging for the whole run: the HTTPS client's fixed-label
        # stage/route warnings are the only per-request failure evidence and
        # must reach stderr. Keep stdout reserved for the JSON receipt.
        with _https_diagnostics_to_stderr(), contextlib.redirect_stdout(io.StringIO()):
            record = run_smoke(handle=args.handle, api_key=key, max_hourly=args.max_hourly,
                               max_seconds=args.max_seconds, payload_bytes=args.payload_bytes,
                               cleanup_only=args.cleanup_only)
        # Print our allowlisted receipt, never args, provider data or exceptions.
        print(json.dumps(record, indent=2, sort_keys=True))
        if record["cleanup"] == "pending":
            return 3
        if args.cleanup_only:
            return 0
        return 0 if record["status"] == "passed" else 1
    except BaseException as exc:
        message = str(exc) if isinstance(exc, SmokeError) else "transport smoke stopped; inspect handle and use cleanup-only"
        print(json.dumps({"status": "failed", "diagnostic": message}), file=sys.stderr)
        return 130 if isinstance(exc, (KeyboardInterrupt, SmokeInterrupted)) else 2
