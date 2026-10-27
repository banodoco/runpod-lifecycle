"""Offline qualification-command custody, budgets and real agent transfer tests."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import threading

import httpx
import pytest

from runpod_lifecycle import cli
from runpod_lifecycle import transport_smoke as smoke
from runpod_lifecycle.https import HttpsPodTransport, read_token_file
from runpod_lifecycle.pod_agent import AgentServer

KEY = "test-provider-secret-must-not-escape"

# The product's transfer deadline owns SIGALRM; keep pytest's independent
# 60-second bound without consuming the same process alarm.
pytestmark = pytest.mark.timeout(60, method="thread")


class Clock:
    def __init__(self):
        self.now = 0.0

    def time(self):
        return 1800000000 + self.now

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Provider:
    def __init__(self, handle):
        self.handle = handle
        self.pod = None
        self.create_calls = []
        self.delete_calls = []
        self.inspections = []
        self.existing = []
        self.rate = 0.06
        self.create_error = None
        self.create_allocates = True
        self.delete_error = False
        self.delete_ack_only = False
        self.lookup_error = False
        self.token = None

    def pods(self):
        return self.existing + ([self.pod] if self.pod else [])

    def create(self, name, token):
        # The intent and token must already be durable BEFORE any allocation.
        intent = json.loads(self.handle.read_text())
        assert intent["request_name"] == name and intent["create_attempts"] == 1
        assert intent["cleanup"] == "pending" and intent["pod_id"] is None
        assert read_token_file(intent["token_file"]) == token
        assert token not in self.handle.read_text()
        self.token = token
        self.create_calls.append(name)
        if self.create_allocates:
            self.pod = {"id": "fakepod1", "name": name, "costPerHr": self.rate}
        if self.create_error:
            raise self.create_error
        return self.pod

    def inspect(self, pod_id):
        self.inspections.append(pod_id)
        assert pod_id == "fakepod1"
        # Identity has to be saved before price inspection or deletion.
        assert json.loads(self.handle.read_text())["pod_id"] == pod_id
        if self.lookup_error:
            raise RuntimeError(KEY + (self.token or ""))
        return (200, self.pod.copy()) if self.pod else (404, None)

    def terminate(self, pod_id):
        self.delete_calls.append(pod_id)
        if self.delete_error:
            raise RuntimeError(KEY)
        if not self.delete_ack_only:
            self.pod = None


class Transport:
    def __init__(self, clock, **kwargs):
        self.clock = clock
        self.kwargs = kwargs
        self.error = None
        self.data = None
        self.disconnected = False
        self.long_started = None
        self.health = False

    def wait_ready(self, **kwargs):
        self.health = True

    def execute_command(self, command, **kwargs):
        assert command == "uname -a"
        return 0, KEY, KEY  # Remote output deliberately includes a secret.

    def put(self, source, remote):
        assert self.health
        if self.error:
            if isinstance(self.error, int):
                os.kill(os.getpid(), self.error)
            raise self.error
        self.data = Path(source).read_bytes()

    def get(self, remote, destination, **kwargs):
        assert not kwargs  # Exercise transport's default JSON envelope mode.
        Path(destination).write_bytes(self.data)

    def start(self, command, *, job_id):
        assert command == "sleep 102"
        self.long_started = self.clock.monotonic()
        return job_id

    def poll(self, job_id):
        return {"status": "exited" if self.clock.monotonic() - self.long_started >= 102 else "running",
                "exit_code": 0, "stdout": KEY, "stderr": KEY}

    def disconnect(self):
        self.disconnected = True


@pytest.fixture
def rig(tmp_path, monkeypatch):
    # No code path in these tests may contact an external address.
    original_request = httpx.Client.request
    def loopback_only(client, method, url, **kwargs):
        target = client.base_url.join(url)
        assert target.host in {"127.0.0.1", "localhost"}, "external HTTP forbidden"
        return original_request(client, method, url, **kwargs)
    monkeypatch.setattr(httpx.Client, "request", loopback_only)
    monkeypatch.setattr(smoke, "preflight_transport", lambda *a, **kw: None)
    handle = tmp_path / "qualification.json"
    provider = Provider(handle)
    clock = Clock()
    transport = Transport(clock)
    def factory(pod_id, token, **kwargs):
        assert pod_id == "fakepod1" and token == provider.token
        transport.kwargs = kwargs
        return transport
    def run(**kwargs):
        return smoke.run_smoke(handle=handle, api_key=KEY, max_hourly=.1,
                               provider=provider, clock=clock, transport_factory=factory,
                               **dict({"payload_bytes": 4096}, **kwargs))
    return handle, provider, clock, transport, run


def test_success_receipt_and_fresh_handle_required(rig):
    handle, provider, clock, transport, run = rig
    result = run(payload_bytes=50 * smoke.MIB)
    assert result["status"] == "passed" and result["qualified"] is True
    assert result["protocol"] == 1 and result["uname"] == "passed"
    receipt = result["transfer_receipt"]
    assert receipt["bytes"] == 52428800 and receipt["download_mode"] == "json-envelope"
    assert receipt["sha256"] == hashlib.sha256(transport.data).hexdigest()
    assert receipt["full_sha256_verified"] is True
    assert result["long_job"]["seconds"] >= 102
    assert result["cleanup_get_status"] == 404 and result["cleanup"] == "verified_absent"
    assert result["lifetime_upper_bound_seconds"] >= 102
    assert result["estimated_cost_upper_bound"] >= .06 * 102 / 3600
    assert len(provider.create_calls) == 1 and provider.delete_calls == ["fakepod1"]
    assert transport.disconnected and transport.kwargs["request_timeout"] <= 30
    assert not list(handle.with_name(handle.name + ".files").iterdir())
    before = handle.read_bytes()
    with pytest.raises(FileExistsError):
        run()
    assert len(provider.create_calls) == 1 and handle.read_bytes() == before
    assert KEY not in before.decode() and provider.token not in before.decode()


def test_preflight_denied_before_create(rig, monkeypatch):
    handle, provider, _, _, run = rig
    def deny(*a, **kw):
        raise RuntimeError(KEY)
    monkeypatch.setattr(smoke, "preflight_transport", deny)
    result = run()
    assert result["status"] == "failed" and result["create_attempts"] == 0
    assert result["cleanup"] == "not_needed" and not provider.create_calls
    assert KEY not in handle.read_text()


def test_existing_account_pod_refused_including_stopped(rig):
    _, provider, _, _, run = rig
    provider.existing = [{"id": "someoneelse", "desiredStatus": "STOPPED"}]
    result = run()
    assert result["create_attempts"] == 0 and not provider.create_calls
    assert not provider.delete_calls and result["cleanup"] == "not_needed"


@pytest.mark.parametrize("error", [RuntimeError(KEY), TimeoutError(), KeyboardInterrupt()])
def test_ambiguous_create_reconciles_exact_name_never_retries(rig, error):
    handle, provider, _, _, run = rig
    provider.create_error = error
    # Unrelated pod appearing AFTER preflight must never be selected/deleted.
    original_create = provider.create
    def create(*args):
        try:
            return original_create(*args)
        finally:
            provider.existing = [{"id": "unowned", "name": "other"}]
    provider.create = create
    result = run()
    assert len(provider.create_calls) == 1
    assert result["pod_id"] == "fakepod1" and result["cleanup"] == "verified_absent"
    assert result["reconciliation_matches"] == 1
    assert provider.existing and provider.delete_calls == ["fakepod1"]
    assert provider.token not in handle.read_text() and KEY not in handle.read_text()
    with pytest.raises(FileExistsError):
        run()
    assert len(provider.create_calls) == 1


def test_lost_create_empty_inventory_remains_pending_then_recovers(rig):
    handle, provider, clock, _, run = rig
    provider.create_error = RuntimeError(KEY)
    provider.create_allocates = False
    result = run()
    assert result["cleanup"] == "pending" and result["reconciliation_matches"] == 0
    assert result["pod_id"] is None
    # Create can become visible much later. Cleanup-only needs no bearer file.
    Path(result["token_file"]).unlink()
    clock.sleep(100)
    provider.pod = {"id": "fakepod1", "name": result["request_name"], "costPerHr": .06}
    recovered = smoke.run_smoke(handle=handle, api_key=KEY, cleanup_only=True,
                                provider=provider, clock=clock, max_seconds=20)
    assert recovered["cleanup_get_status"] == 404
    assert recovered["lifetime_upper_bound_seconds"] >= 100
    assert len(provider.create_calls) == 1


def test_multiple_exact_name_matches_refuse_guessing(rig):
    _, provider, _, _, run = rig
    provider.create_error = RuntimeError(KEY)
    original = provider.create
    def create(*args):
        try:
            return original(*args)
        finally:
            provider.existing = [dict(provider.pod, id="secondpod")]
    provider.create = create
    result = run()
    assert result["cleanup"] == "pending" and result["reconciliation_matches"] == 2
    assert not provider.delete_calls


@pytest.mark.parametrize("rate", [1.2, None, 0, -1, float("nan"), float("inf"), "0.06", True])
def test_observed_rate_fails_closed_and_releases(rig, rate):
    _, provider, _, transport, run = rig
    provider.rate = rate
    result = run()
    assert result["status"] == "failed" and result["cleanup_get_status"] == 404
    assert not transport.health and provider.delete_calls == ["fakepod1"]


@pytest.mark.parametrize("error,category", [
    (RuntimeError(KEY), "operation_failed"), (KeyboardInterrupt(), "interrupted"),
    (TimeoutError(), "timeout"), (signal.SIGINT, "interrupted"),
    (signal.SIGTERM, "interrupted"), (signal.SIGALRM, "timeout"),
])
def test_transfer_failure_or_signal_always_cleans_preserves_evidence(rig, error, category):
    handle, provider, _, transport, run = rig
    transport.error = error
    result = run()
    assert result["failure"] == category and result["phase"] == "upload"
    assert result["status"] == "failed" and result["cleanup_get_status"] == 404
    assert provider.delete_calls == ["fakepod1"] and transport.disconnected
    assert (handle.with_name(handle.name + ".files") / "source.bin").exists()
    token_file = Path(result["token_file"])
    assert token_file.stat().st_uid == os.getuid()
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
    assert read_token_file(token_file) == provider.token
    assert provider.token not in handle.read_text() and KEY not in handle.read_text()


@pytest.mark.parametrize("mode", ["ack_only", "error", "lookup_error"])
def test_cleanup_failure_remains_durable_then_cleanup_only_recovers(rig, mode):
    handle, provider, clock, transport, run = rig
    provider.delete_ack_only = mode == "ack_only"
    provider.delete_error = mode == "error"
    # Trigger lookup failure during cleanup rather than initial price lookup.
    original_put = transport.put
    def put(*args):
        original_put(*args)
        provider.lookup_error = mode == "lookup_error"
        raise RuntimeError(KEY)
    transport.put = put
    result = run(max_seconds=30)
    assert result["cleanup"] == "pending"
    assert json.loads(handle.read_text())["cleanup"] == "pending"
    assert "released_at" not in result and "cleanup_get_status" not in result
    Path(result["token_file"]).unlink()
    provider.delete_ack_only = provider.delete_error = provider.lookup_error = False
    recovered = smoke.run_smoke(handle=handle, api_key=KEY, cleanup_only=True,
                                max_seconds=15, provider=provider, clock=clock)
    assert recovered["cleanup"] == "verified_absent" and recovered["cleanup_get_status"] == 404
    assert len(provider.create_calls) == 1


def test_lost_delete_reply_can_be_verified_by_independent_get(rig):
    _, provider, _, _, run = rig
    def terminate(pod_id):
        provider.delete_calls.append(pod_id)
        provider.pod = None
        raise RuntimeError(KEY)
    provider.terminate = terminate
    result = run()
    assert result["cleanup"] == "verified_absent" and result["cleanup_get_status"] == 404
    assert provider.inspections[-1] == "fakepod1"


def test_cleanup_target_name_mismatch_is_never_deleted(rig):
    handle, provider, clock, _, run = rig
    provider.delete_error = True
    result = run(max_seconds=15)
    provider.pod["name"] = "not-our-intent"
    provider.delete_calls.clear()
    recovered = smoke.run_smoke(handle=handle, api_key=KEY, cleanup_only=True,
                                provider=provider, clock=clock, max_seconds=15)
    assert recovered["cleanup"] == "pending" and not provider.delete_calls


def test_small_budget_records_long_job_skip(rig):
    _, _, _, _, run = rig
    result = run(max_seconds=30)
    assert result["status"] == "passed" and not result["qualified"]
    assert result["long_job"] == {"status": "skipped", "reason": "insufficient_remaining_budget"}


@pytest.mark.parametrize("kwargs", [
    {"max_seconds": 901}, {"max_seconds": 0}, {"max_seconds": -1},
    {"max_seconds": 1.5}, {"max_hourly": None}, {"max_hourly": float("nan")},
    {"max_hourly": float("inf")}, {"max_hourly": 0}, {"max_hourly": -1},
    {"payload_bytes": 0}, {"payload_bytes": 1024 * smoke.MIB + 1},
])
def test_invalid_options_before_allocation(rig, kwargs):
    handle, provider, clock, _, _ = rig
    with pytest.raises(smoke.SmokeError):
        smoke.run_smoke(handle=handle, api_key=KEY, provider=provider, clock=clock,
                        **dict({"max_hourly": .1}, **kwargs))
    assert not handle.exists() and not provider.create_calls


def test_real_loopback_50mib_json_roundtrip(rig, tmp_path, monkeypatch):
    handle, provider, clock, _, _ = rig
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    server = None
    thread = None
    remote = None
    requests = []
    class LocalTransport(HttpsPodTransport):
        def __init__(self, pod_id, token, **kwargs):
            nonlocal server, thread
            server = AgentServer(("127.0.0.1", 0), token, tmp_path / "agent-state")
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            super().__init__(pod_id, token, base_url=f"http://127.0.0.1:{server.server_port}",
                             request_timeout=.5, poll_interval=.01)
        def _request(self, method, path, **kwargs):
            requests.append((method, path))
            return super()._request(method, path, **kwargs)
        def put(self, local, remote_path, **kwargs):
            nonlocal remote
            remote = Path(remote_path)
            return super().put(local, remote_path, **kwargs)
    try:
        result = smoke.run_smoke(handle=handle, api_key=KEY, max_hourly=.1, max_seconds=90,
                                  provider=provider, clock=clock, transport_factory=LocalTransport)
        assert result["status"] == "passed" and result["cleanup_get_status"] == 404
        assert result["transfer_receipt"]["bytes"] == 50 * smoke.MIB
        assert result["transfer_receipt"]["sha256"] == hashlib.sha256(remote.read_bytes()).hexdigest()
        assert ("GET", "/health") in requests
        assert sum(path == "/download-chunk" for _, path in requests) == 50
        assert not any(path == "/files" for _, path in requests)
        assert provider.token not in handle.read_text() and KEY not in handle.read_text()
    finally:
        if remote:
            remote.unlink(missing_ok=True)
        if server:
            server.shutdown()
            server.server_close()
            thread.join(2)


@pytest.mark.parametrize("stage", ["initiate", "chunk", "verify", "status"])
def test_upload_404_stages_keep_same_receipt_and_log_safe_stage(rig, tmp_path, monkeypatch, caplog, stage):
    """Distinct local causes match QUAL0's retained evidence; none proves its cause."""
    from runpod_lifecycle.pod_agent import Handler

    handle, provider, clock, _, _ = rig
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    requests = []
    remote = None

    class RejectUpload(Handler):
        def dispatch(self):
            requests.append((self.command, self.path))
            matches = ((stage == "initiate" and self.command == "POST" and self.path == "/uploads")
                       or (stage == "chunk" and self.command == "PUT")
                       or (stage == "verify" and self.command == "POST" and self.path.startswith("/uploads/"))
                       or (stage == "status" and self.command == "GET" and self.path.startswith("/uploads/")))
            if matches:
                return self.reply(404, {"error": KEY + provider.token}, headers={"X-Secret": KEY})
            return super().dispatch()

    server = AgentServer(("127.0.0.1", 0), "fixture-only-" + "x" * 32, tmp_path / "agent-state")
    server.RequestHandlerClass = RejectUpload
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    class LocalTransport(HttpsPodTransport):
        def __init__(self, pod_id, token, **kwargs):
            server.token = token
            super().__init__(pod_id, token, base_url=f"http://127.0.0.1:{server.server_port}",
                             request_timeout=.5, poll_interval=.01)

        def put(self, local, remote_path, **kwargs):
            nonlocal remote
            remote = Path(remote_path)
            return super().put(local, remote_path, **kwargs)

    try:
        result = smoke.run_smoke(handle=handle, api_key=KEY, max_hourly=.1, max_seconds=90,
                                 payload_bytes=4096, provider=provider, clock=clock,
                                 transport_factory=LocalTransport)
        assert result["phase"] == "upload" and result["uname"] == "passed"
        assert result["status"] == "failed" and not result["qualified"]
        assert {k: v for k, v in result.items() if k.startswith("transport_")} == {
            "transport_failure": "http_rejected", "transport_http_status": 404}
        assert "transfer_receipt" not in result and "long_job" not in result
        assert result["cleanup"] == "verified_absent" and result["cleanup_get_status"] == 404
        assert len(provider.create_calls) == 1 and provider.delete_calls == ["fakepod1"]
        assert ("GET", "/health") in requests and any(path == "/jobs" for _, path in requests)
        messages = [r.getMessage() for r in caplog.records if r.name == "runpod_lifecycle.https"]
        assert f"pod HTTPS upload failed (stage={stage} HTTP 404)" in messages
        for secret in (KEY, provider.token, str(remote), "X-Secret"):
            assert secret not in "\n".join(messages) + handle.read_text()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
        if remote:
            remote.unlink(missing_ok=True)


def test_product_provider_uses_only_standard_minimum_cpu_api(monkeypatch):
    calls = []
    def create(*args, **kwargs):
        calls.append((args, kwargs))
        return {"id": "fakepod1"}
    monkeypatch.setattr(smoke.api, "create_pod", create)
    provider = smoke.Provider(KEY)
    try:
        assert provider.create("testname", "t" * 40) == {"id": "fakepod1"}
    finally:
        provider.close()
    args, kwargs = calls[0]
    assert args == (KEY, None, smoke.IMAGE)
    assert kwargs["compute_type"] == "CPU" and kwargs["vcpu_count"] == 2
    assert kwargs["cpu_api"] == "rest" and kwargs["cpu_flavor_ids"] == ("cpu3c", "cpu5c")
    assert kwargs["network_volume_id"] is None and kwargs["disk_in_gb"] == 0
    assert kwargs["transport"] == "https" and kwargs["https_token"] == "t" * 40


def test_product_provider_exact_get_404_only(monkeypatch):
    provider = smoke.Provider(KEY)
    provider.http.close()
    responses = iter([httpx.Response(404), httpx.Response(403),
                      httpx.Response(200, json={"id": "other"})])
    def handler(request):
        assert request.method == "GET" and request.url.path == "/v1/pods/fakepod1"
        assert request.headers["Authorization"] == "Bearer " + KEY
        return next(responses)
    provider.http = httpx.Client(transport=httpx.MockTransport(handler),
                                 headers={"Authorization": "Bearer " + KEY})
    try:
        assert provider.inspect("fakepod1") == (404, None)
        with pytest.raises(smoke.SmokeError):
            provider.inspect("fakepod1")
        with pytest.raises(smoke.SmokeError):
            provider.inspect("fakepod1")
    finally:
        provider.close()


def test_product_cli_help_and_required_handle(capsys):
    with pytest.raises(SystemExit) as caught:
        cli.build_parser().parse_args(["transport-smoke", "--help"])
    assert caught.value.code == 0
    output = capsys.readouterr().out
    for option in ("--handle", "--max-hourly", "--max-seconds", "--payload-bytes", "--cleanup-only"):
        assert option in output
    assert "600" in output and "900" in output and "50 MiB" in output
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["transport-smoke", "--max-hourly", ".1"])
    args = cli.build_parser().parse_args(["transport-smoke", "--handle", "receipt.json"])
    assert args.max_seconds == 600 and args.payload_bytes == 52428800
    assert args.max_hourly is None


def test_cli_errors_and_provider_logging_are_secret_free(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    monkeypatch.setattr(cli, "load_runpod_env", lambda: None)
    def fail(**kwargs):
        with smoke._quiet_provider():
            import logging
            print(KEY)
            logging.getLogger("runpod_lifecycle.api").warning(KEY)
        raise RuntimeError(KEY)
    monkeypatch.setattr(smoke, "run_smoke", fail)
    assert cli.main(["transport-smoke", "--handle", str(tmp_path / "h"), "--max-hourly", ".1"]) == 2
    out = capsys.readouterr()
    assert KEY not in out.out + out.err
    assert json.loads(out.err)["status"] == "failed"


def test_cleanup_only_cli_needs_no_agent_token(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_runpod_env", lambda: None)
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    monkeypatch.setenv("RUNPOD_HTTPS_TOKEN_FILE", "/missing-token-file")
    def cleanup(**kwargs):
        assert kwargs["cleanup_only"] and kwargs["max_hourly"] is None
        return {"status": "failed", "cleanup": "verified_absent", "cleanup_get_status": 404}
    monkeypatch.setattr(smoke, "run_smoke", cleanup)
    assert cli.main(["transport-smoke", "--handle", str(tmp_path / "h"), "--cleanup-only"]) == 0
    assert json.loads(capsys.readouterr().out)["cleanup_get_status"] == 404


def test_real_alarm_bounds_blocking_transfer_even_with_exception_retry(rig):
    import time
    _, provider, _, transport, run = rig
    def stuck(*args):
        # Simulate a library that retries ordinary exceptions. The deadline
        # must escape this handler to reach durable cleanup.
        try:
            time.sleep(30)
        except Exception:
            time.sleep(30)
    transport.put = stuck
    started = time.monotonic()
    result = run(max_seconds=1)
    assert time.monotonic() - started < 2
    assert result["failure"] == "timeout" and result["cleanup_get_status"] == 404
    assert provider.delete_calls == ["fakepod1"]


def test_local_lock_prevents_second_fresh_handle_allocation(rig):
    _, provider, _, _, run = rig
    lock = Path(smoke.tempfile.gettempdir()) / f"runpod-transport-smoke-{os.getuid()}.lock"
    with smoke._lock(lock):
        with pytest.raises(smoke.SmokeError, match="another transport smoke"):
            run()
    assert not provider.create_calls


def test_cleanup_only_without_token_rechecks_even_completed_handle(rig):
    handle, provider, clock, _, run = rig
    result = run()
    assert not Path(result["token_file"]).exists()
    provider.inspections.clear()
    recovered = smoke.run_smoke(handle=handle, api_key=KEY, cleanup_only=True,
                                provider=provider, clock=clock, max_seconds=15)
    assert recovered["cleanup_get_status"] == 404 and provider.inspections == ["fakepod1"]
    assert len(provider.create_calls) == 1


def test_wrong_handle_owner_mode_or_symlink_fails_before_provider(rig, tmp_path):
    handle, provider, clock, _, run = rig
    result = run()
    handle.chmod(0o644)
    with pytest.raises(smoke.SmokeError, match="unsafe local custody"):
        smoke.run_smoke(handle=handle, api_key=KEY, cleanup_only=True,
                        provider=provider, clock=clock)
    handle.chmod(0o600)
    link = tmp_path / "link.json"
    link.symlink_to(handle)
    with pytest.raises(OSError):
        smoke.run_smoke(handle=link, api_key=KEY, cleanup_only=True,
                        provider=provider, clock=clock)
    assert len(provider.create_calls) == 1


def test_mismatched_download_preserves_full_failure_evidence(rig):
    handle, provider, _, transport, run = rig
    def corrupt(remote, destination):
        Path(destination).write_bytes(b"changed")
    transport.get = corrupt
    result = run()
    assert result["status"] == "failed" and result["phase"] == "download"
    assert result["cleanup_get_status"] == 404 and not result["qualified"]
    assert (handle.with_name(handle.name + ".files") / "download.bin").read_bytes() == b"changed"


def test_long_job_running_at_deadline_releases_pod(rig):
    _, provider, clock, transport, run = rig
    transport.poll = lambda job: {"status": "running", "exit_code": None}
    result = run(max_seconds=200)
    assert result["failure"] == "timeout" and result["phase"] == "long_job"
    assert not result["qualified"] and result["cleanup_get_status"] == 404
    assert len(provider.create_calls) == 1


def test_transfer_http_status_retained_without_arbitrary_error_text(rig):
    handle, _, _, transport, run = rig
    transport.error = smoke.PodTransportError("pod HTTPS request rejected (HTTP 502)")
    result = run()
    assert result["transport_http_status"] == 502 and result["transport_failure"] == "http_rejected"
    assert "502" in handle.read_text() and KEY not in handle.read_text()


@pytest.mark.parametrize("message,metadata", [
    ("request rejected (operation=download framing=json offset=1048576 HTTP 403)",
     {"transport_framing": "json", "transport_offset": 1048576,
      "transport_http_status": 403, "transport_failure": "http_rejected"}),
    ("request rejected (operation=download framing=range offset=0 HTTP 503)",
     {"transport_framing": "range", "transport_offset": 0,
      "transport_http_status": 503, "transport_failure": "http_rejected"}),
    ("invalid download chunk or changed file (operation=download framing=json offset=7 HTTP 200)",
     {"transport_framing": "json", "transport_offset": 7,
      "transport_http_status": 200, "transport_failure": "transport_error"}),
    ("invalid download range or changed file (operation=download framing=range offset=8 HTTP 200 received_bytes=3)",
     {"transport_framing": "range", "transport_offset": 8, "transport_received_bytes": 3,
      "transport_http_status": 200, "transport_failure": "transport_error"}),
    ("short download or invalid range size (operation=download framing=range offset=9 HTTP 206 received_bytes=0)",
     {"transport_framing": "range", "transport_offset": 9, "transport_received_bytes": 0,
      "transport_http_status": 206, "transport_failure": "transport_error"}),
    ("pod HTTPS request failed (operation=download framing=json offset=10 HTTP unavailable)",
     {"transport_framing": "json", "transport_offset": 10, "transport_failure": "transport_error"}),
    ("pod HTTPS request failed (operation=download framing=range offset=0 HTTP unavailable)",
     {"transport_framing": "range", "transport_offset": 0, "transport_failure": "transport_error"}),
    ("pod HTTPS request rejected (HTTP 401)",
     {"transport_http_status": 401, "transport_failure": "http_rejected"}),
    ("pod HTTPS request rejected (HTTP 504)",
     {"transport_http_status": 504, "transport_failure": "http_rejected"}),
])
def test_download_error_formats_retain_only_safe_scalars(rig, message, metadata):
    handle, provider, _, transport, run = rig
    def fail(*args):
        raise smoke.PodTransportError(message)
    transport.get = fail
    result = run()
    if "transport_framing" in metadata:
        metadata = dict(metadata, transport_operation="download")
    assert {key: value for key, value in result.items() if key.startswith("transport_")} == metadata
    for key, value in metadata.items():
        assert type(result[key]) is type(value)
    assert result["status"] == "failed" and result["phase"] == "download"
    assert result["failure"] == "operation_failed" and not result["qualified"]
    assert result["cleanup_get_status"] == 404 and transport.disconnected
    assert provider.delete_calls == ["fakepod1"] and len(provider.create_calls) == 1
    assert json.loads(handle.read_text()) == result
    assert message not in handle.read_text() and "diagnostic" not in result
    assert KEY not in handle.read_text() and provider.token not in handle.read_text()


@pytest.mark.parametrize("message", [
    "{secret}", "body={secret} (operation=download framing=json offset=0 HTTP 403)",
    "request rejected (operation=download framing=json offset=0 HTTP 403) {secret}",
    "{secret}pod HTTPS request rejected (HTTP 403)",
    "pod HTTPS request rejected (HTTP 403)\n{secret}",
    "pod HTTPS request rejected (HTTP 099)", "pod HTTPS request rejected (HTTP 600)",
    "request rejected (operation={secret} framing=json offset=0 HTTP 403)",
    "request rejected (operation=download framing={secret} offset=0 HTTP 403)",
    "request rejected (operation=download framing=json offset={secret} HTTP 403)",
    "request rejected (operation=download framing=json offset=-1 HTTP 403)",
    "request rejected (operation=download framing=json offset=1.5 HTTP 403)",
    "request rejected (operation=download framing=json offset=True HTTP 403)",
    "request rejected (operation=download framing=json offset=١ HTTP 403)",
    "request rejected (operation=download framing=json offset=" + "9" * 5000 + " HTTP 403)",
    "request rejected (operation=download framing=json offset=9223372036854775808 HTTP 403)",
    "request rejected (operation=download framing=json offset=0 HTTP 999)",
    "request rejected (operation=download framing=json offset=0 HTTP ٤٠٣)",
    "request rejected (operation=download framing=json offset=0 HTTP unavailable)",
    "pod HTTPS request failed (operation=download framing=json offset=0 HTTP 403)",
    "invalid download chunk or changed file (operation=download framing=range offset=0 HTTP 200)",
    "invalid download range or changed file (operation=download framing=json offset=0 HTTP 206 received_bytes=3)",
    "invalid download range or changed file (operation=download framing=range offset=0 HTTP 206)",
    "short download or invalid range size (operation=download framing=range offset=0 HTTP 206 received_bytes=-1)",
    "short download or invalid range size (operation=download framing=range offset=0 HTTP 206 received_bytes=9223372036854775808)",
    "request rejected (operation=download framing=json offset=0 HTTP 403 received_bytes=3)",
    "request rejected (operation=download framing=json offset=0 HTTP 403 header={secret})",
])
def test_arbitrary_transport_error_text_never_enters_cli_or_handle(rig, monkeypatch, capsys, message):
    handle, provider, clock, transport, _ = rig
    sentinels = (KEY, "sentinel-bearer-F014", "sentinel-body-F014",
                 "X-Sentinel-Header-F014", "/private/sentinel-path-F014")
    def fail(*args):
        raise smoke.PodTransportError(message.format(secret=" ".join((*sentinels, provider.token))))
    transport.get = fail
    real_run = smoke.run_smoke
    monkeypatch.setattr(smoke, "run_smoke", lambda **kwargs: real_run(
        **kwargs, provider=provider, clock=clock, transport_factory=lambda *a, **kw: transport))
    monkeypatch.setattr(cli, "load_runpod_env", lambda: None)
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    assert cli.main(["transport-smoke", "--handle", str(handle), "--max-hourly", ".1",
                     "--payload-bytes", "4096"]) == 1
    printed = capsys.readouterr()
    result = json.loads(printed.out)
    assert json.loads(handle.read_text()) == result
    assert {key: value for key, value in result.items() if key.startswith("transport_")} == {
        "transport_failure": "transport_error"}
    assert result["phase"] == "download" and result["cleanup_get_status"] == 404
    assert len(provider.create_calls) == 1 and provider.delete_calls == ["fakepod1"]
    assert "diagnostic" not in result
    for sentinel in (*sentinels, provider.token):
        assert sentinel not in printed.out + printed.err + handle.read_text()


@pytest.mark.parametrize("mode,framing,status,offset,received", [
    ("rejected", "json", 412, 5, None),
    ("rejected", "range", 403, 0, None),
    ("malformed", "json", 200, 0, None),
    ("malformed", "range", 206, 0, 3),
    ("short", "range", 206, 0, 3),
    ("unavailable", "json", None, 0, None),
    ("unavailable", "range", None, 0, None),
])
def test_integrated_download_errors_reach_failed_cli_receipt(
        rig, monkeypatch, capsys, mode, framing, status, offset, received):
    import base64

    handle, provider, clock, transport, _ = rig
    sentinels = (KEY, "sentinel-body-F014", "X-Sentinel-Header-F014", "/private/sentinel-path-F014")
    calls = []
    def fail_download(remote, destination):
        def respond(request):
            calls.append(request)
            secret = " ".join((*sentinels, "Bearer " + provider.token))
            if mode == "unavailable":
                raise httpx.ReadTimeout(secret, request=request)
            if offset and len(calls) == 1:
                chunk = b"first"
                return httpx.Response(200, json={"offset": 0, "total": 6, "etag": secret,
                    "sha256": hashlib.sha256(chunk).hexdigest(), "data": base64.b64encode(chunk).decode()})
            headers = {"ETag": secret, "X-Sentinel-Header-F014": secret}
            if mode == "short":
                headers["Content-Range"] = "bytes 0-4/5"
            else:
                headers["Content-Range"] = secret
            return httpx.Response(status, content=b"bad" if received is not None else secret.encode(),
                                  headers=headers)
        client = HttpsPodTransport("fakepod1", provider.token, base_url="http://127.0.0.1", poll_interval=0)
        client.http.close()
        client.http = httpx.Client(base_url="http://127.0.0.1", transport=httpx.MockTransport(respond))
        try:
            client.get(sentinels[-1], destination, use_ranges=framing == "range")
        finally:
            client.close()
    transport.get = fail_download
    real_run = smoke.run_smoke
    monkeypatch.setattr(smoke, "run_smoke", lambda **kwargs: real_run(
        **kwargs, provider=provider, clock=clock, transport_factory=lambda *a, **kw: transport))
    monkeypatch.setattr(cli, "load_runpod_env", lambda: None)
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    assert cli.main(["transport-smoke", "--handle", str(handle), "--max-hourly", ".1",
                     "--payload-bytes", "4096"]) == 1
    printed = capsys.readouterr()
    result = json.loads(printed.out)
    expected = {"transport_operation": "download", "transport_framing": framing,
                "transport_offset": offset,
                "transport_failure": "http_rejected" if mode == "rejected" else "transport_error"}
    if status is not None:
        expected["transport_http_status"] = status
    if received is not None:
        expected["transport_received_bytes"] = received
    assert {key: value for key, value in result.items() if key.startswith("transport_")} == expected
    assert result["phase"] == "download" and result["status"] == "failed" and not result["qualified"]
    assert result["cleanup_get_status"] == 404 and transport.disconnected
    assert len(provider.create_calls) == 1 and provider.delete_calls == ["fakepod1"]
    assert len(calls) == (3 if mode == "unavailable" else 2 if offset else 1)
    assert json.loads(handle.read_text()) == result
    for sentinel in (*sentinels, provider.token):
        assert sentinel not in printed.out + printed.err + handle.read_text()


def test_failed_download_metadata_survives_cleanup_only_without_token(rig):
    handle, provider, clock, transport, run = rig
    provider.delete_error = True
    def fail(*args):
        raise smoke.PodTransportError(
            "request rejected (operation=download framing=json offset=1048576 HTTP 412)")
    transport.get = fail
    result = run(max_seconds=30)
    assert result["cleanup"] == "pending" and "cleanup_get_status" not in result
    metadata = {key: value for key, value in result.items() if key.startswith("transport_")}
    assert metadata == {"transport_operation": "download", "transport_framing": "json",
                        "transport_offset": 1048576, "transport_http_status": 412,
                        "transport_failure": "http_rejected"}
    Path(result["token_file"]).unlink()
    provider.delete_error = False
    result = smoke.run_smoke(handle=handle, api_key=KEY, cleanup_only=True,
                             provider=provider, clock=clock, max_seconds=15)
    assert {key: value for key, value in result.items() if key.startswith("transport_")} == metadata
    assert result["phase"] == "download" and result["status"] == "failed" and not result["qualified"]
    assert result["cleanup_get_status"] == 404 and len(provider.create_calls) == 1
    assert json.loads(handle.read_text()) == result
    assert KEY not in handle.read_text() and provider.token not in handle.read_text()


@pytest.mark.parametrize("lost_reply", [False, True])
def test_storage_failure_after_create_still_attempts_release(rig, monkeypatch, lost_reply):
    handle, provider, _, _, run = rig
    provider.create_error = RuntimeError(KEY) if lost_reply else None
    real_save = smoke._save
    def no_space(path, record):
        if provider.create_calls:
            raise OSError("test simulated disk full")
        real_save(path, record)
    monkeypatch.setattr(smoke, "_save", no_space)
    # Provider reads do not depend on a functioning local filesystem. Disable
    # the normal fake's assertion that identity was already successfully saved.
    def inspect(pod_id):
        return (200, provider.pod.copy()) if provider.pod else (404, None)
    provider.inspect = inspect
    with pytest.raises(OSError):
        run()
    assert provider.delete_calls == ["fakepod1"] and provider.pod is None
    durable = json.loads(handle.read_text())
    assert durable["request_name"] == provider.create_calls[0] and durable["create_attempts"] == 1


def test_product_cli_runs_receipt_and_failure_categories(rig, monkeypatch, capsys):
    handle, provider, clock, transport, run = rig
    real_run = smoke.run_smoke
    def offline(**kwargs):
        return real_run(**kwargs, provider=provider, clock=clock,
                        transport_factory=lambda *a, **kw: transport)
    monkeypatch.setattr(smoke, "run_smoke", offline)
    monkeypatch.setattr(cli, "load_runpod_env", lambda: None)
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    assert cli.main(["transport-smoke", "--handle", str(handle), "--max-hourly", ".1"]) == 0
    printed = capsys.readouterr()
    result = json.loads(printed.out)
    assert result["qualified"] and result["cleanup_get_status"] == 404
    assert KEY not in printed.out + printed.err and provider.token not in printed.out + printed.err
    assert json.loads(handle.read_text()) == result


def test_product_cli_returns_pending_instead_of_pass(rig, monkeypatch, capsys):
    handle, provider, clock, transport, _ = rig
    provider.delete_ack_only = True
    real_run = smoke.run_smoke
    def offline(**kwargs):
        return real_run(**kwargs, provider=provider, clock=clock,
                        transport_factory=lambda *a, **kw: transport)
    monkeypatch.setattr(smoke, "run_smoke", offline)
    monkeypatch.setattr(cli, "load_runpod_env", lambda: None)
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    assert cli.main(["transport-smoke", "--handle", str(handle), "--max-hourly", ".1"]) == 3
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "passed" and result["cleanup"] == "pending"
    assert not result["qualified"] and "cleanup_get_status" not in result


def test_cleanup_recheck_does_not_retain_stale_success_if_get_fails(rig):
    handle, provider, clock, _, run = rig
    result = run(payload_bytes=50 * smoke.MIB)
    assert result["qualified"] and result["cleanup_get_status"] == 404
    provider.lookup_error = True
    result = smoke.run_smoke(handle=handle, api_key=KEY, cleanup_only=True,
                             provider=provider, clock=clock, max_seconds=15)
    assert result["cleanup"] == "pending" and not result["qualified"]
    assert "cleanup_get_status" not in result and "released_at" not in result
    assert "lifetime_upper_bound_seconds" not in result
