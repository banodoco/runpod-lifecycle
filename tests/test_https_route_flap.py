"""Provider route flapping: RunPod's proxy answers an EMPTY 404 while it has no
route to the pod port, and live pods showed 200, 404, 200, 404... for several
seconds after the first successful health check (opus-https probe2). These
tests put a real loopback flapping proxy in front of the real agent."""
from __future__ import annotations

import hashlib
import http.client
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from runpod_lifecycle import cli
from runpod_lifecycle import https as https_module
from runpod_lifecycle import transport_smoke as smoke
from runpod_lifecycle.https import HttpsPodTransport, PodTransportError
from runpod_lifecycle.pod_agent import CHUNK, AgentServer

TOKEN = "route-flap-test-token-" + "y" * 40
HOP = {"connection", "keep-alive", "transfer-encoding", "proxy-connection", "te", "upgrade"}


class FlapProxy:
    """Forward to the agent except for scripted requests, which get the
    provider's no-route reply: 404 with an empty body (never reaches agent)."""

    def __init__(self, agent_port, flap):
        self.flap = flap  # callable(sequence_number, method, path) -> bool
        self.seen = []    # (seq, method, path, flapped)
        self.lock = threading.Lock()
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def handle_one_request(self):
                try:
                    super().handle_one_request()
                except ConnectionError:
                    self.close_connection = True

            def relay(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length else b""
                with proxy.lock:
                    seq = len(proxy.seen)
                    flapped = bool(proxy.flap(seq, self.command, self.path))
                    proxy.seen.append((seq, self.command, self.path.split("?")[0], flapped))
                if flapped:
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                upstream = http.client.HTTPConnection("127.0.0.1", agent_port, timeout=10)
                try:
                    headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP | {"host"}}
                    upstream.request(self.command, self.path, body=body, headers=headers)
                    reply = upstream.getresponse()
                    data = reply.read()
                finally:
                    upstream.close()
                self.send_response(reply.status)
                for key, value in reply.getheaders():
                    if key.lower() not in HOP | {"content-length", "server", "date"}:
                        self.send_header(key, value)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = do_PUT = do_DELETE = relay

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def agent(tmp_path):
    server = AgentServer(("127.0.0.1", 0), TOKEN, tmp_path / "state")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def client_for(proxy, **kwargs):
    kwargs.setdefault("poll_interval", 0.01)
    return HttpsPodTransport("pod1", TOKEN, base_url=proxy.url, request_timeout=10, **kwargs)


def test_route_flaps_during_transfer_and_exec_are_retried_end_to_end(agent, tmp_path, caplog):
    # Mirrors probe2: health OK, upload start OK, then intermittent no-route
    # replies hit chunk, job, poll, verify and download requests.
    flapped = {2, 3, 5, 8, 9, 13, 14, 15, 21, 30, 31}
    source = tmp_path / "source.bin"
    payload = bytes(range(256)) * (3 * CHUNK // 256) + b"tail"
    source.write_bytes(payload)
    with FlapProxy(agent.server_address[1], lambda seq, *_: seq in flapped) as proxy, \
            caplog.at_level(logging.WARNING, logger="runpod_lifecycle.https"):
        client = client_for(proxy)
        try:
            client.connect()
            client.put(source, str(tmp_path / "remote.bin"))
            code, out, _ = client.execute_command("echo routed")
            client.get(str(tmp_path / "remote.bin"), tmp_path / "download.bin")
        finally:
            client.close()
    assert code == 0 and out == "routed\n"
    assert (tmp_path / "download.bin").read_bytes() == payload
    assert hashlib.sha256((tmp_path / "remote.bin").read_bytes()).hexdigest() == hashlib.sha256(payload).hexdigest()
    assert {seq for seq, *_rest, was in proxy.seen if was} == {s for s in flapped if s < len(proxy.seen)}
    assert any(method == "PUT" and was for _seq, method, _path, was in proxy.seen)
    messages = [r.getMessage() for r in caplog.records]
    assert "pod HTTPS route unavailable (HTTP 404 without agent body); retrying" in messages
    assert not any(TOKEN in m for m in messages)


def test_agent_json_404_stays_final_without_route_retry(agent):
    with FlapProxy(agent.server_address[1], lambda *_: False) as proxy:
        client = client_for(proxy, route_grace=60)
        try:
            started = time.monotonic()
            with pytest.raises(PodTransportError, match=r"^pod HTTPS request rejected \(HTTP 404\)$"):
                client.poll("f" * 32)
            assert time.monotonic() - started < 5
        finally:
            client.close()
    assert [(m, p) for _s, m, p, _f in proxy.seen] == [("GET", "/jobs/" + "f" * 32)]


def test_route_grace_is_bounded_and_keeps_v1_error_and_stage(agent, tmp_path, caplog):
    source = tmp_path / "source.bin"
    source.write_bytes(b"x")
    with FlapProxy(agent.server_address[1], lambda seq, *_: seq >= 1) as proxy, \
            caplog.at_level(logging.WARNING, logger="runpod_lifecycle.https"):
        client = client_for(proxy, route_grace=0.3)
        try:
            client.connect()
            started = time.monotonic()
            with pytest.raises(PodTransportError, match=r"^pod HTTPS request rejected \(HTTP 404\)$") as caught:
                client.put(source, "/tmp/never-written")
            assert 0.3 <= time.monotonic() - started < 5
        finally:
            client.close()
    assert smoke._transport_failure(caught.value) == {"transport_failure": "http_rejected",
                                                      "transport_http_status": 404}
    assert "pod HTTPS upload failed (stage=initiate HTTP 404)" in [r.getMessage() for r in caplog.records]
    assert len(agent.jobs) == 0


def test_wait_ready_needs_consecutive_health_not_one_lucky_reply(agent):
    pattern = {0: True, 1: False, 2: True, 3: False, 4: False, 5: False, 6: False}
    with FlapProxy(agent.server_address[1], lambda seq, *_: pattern.get(seq, False)) as proxy:
        client = client_for(proxy)
        try:
            client.wait_ready(timeout=10)
        finally:
            client.close()
    # 404, 200, 404, then three consecutive 200s.
    assert [f for _s, _m, _p, f in proxy.seen] == [True, False, True, False, False, False]


def test_wait_ready_does_not_spend_route_grace_per_probe(agent, caplog):
    with FlapProxy(agent.server_address[1], lambda *_: True) as proxy, \
            caplog.at_level(logging.WARNING, logger="runpod_lifecycle.https"):
        client = client_for(proxy, route_grace=60, poll_interval=0.05)
        try:
            started = time.monotonic()
            with pytest.raises(PodTransportError, match="readiness timed out"):
                client.wait_ready(timeout=0.5)
            assert time.monotonic() - started < 5
        finally:
            client.close()
    assert "pod HTTPS readiness timed out (last=route_unavailable)" in [r.getMessage() for r in caplog.records]


def test_smoke_cli_no_longer_mutes_https_diagnostics(tmp_path, monkeypatch, capsys):
    key = "provider-secret-that-must-not-escape"
    monkeypatch.setenv("RUNPOD_API_KEY", key)
    monkeypatch.setattr(cli, "load_runpod_env", lambda: None)

    def run(**_kwargs):
        https_module.logger.warning("pod HTTPS upload failed (stage=%s HTTP %s)", "chunk", 404)
        with smoke._quiet_provider():  # provider paths stay muted
            logging.getLogger("runpod_lifecycle.api").warning(key)
            print(key)
        print("stray stdout must not corrupt the receipt")
        return {"status": "failed", "cleanup": "verified_absent", "cleanup_get_status": 404}

    monkeypatch.setattr(smoke, "run_smoke", run)
    assert cli.main(["transport-smoke", "--handle", str(tmp_path / "h"), "--max-hourly", ".1"]) == 1
    out = capsys.readouterr()
    assert json.loads(out.out)["cleanup_get_status"] == 404
    assert "pod HTTPS upload failed (stage=chunk HTTP 404)" in out.err
    assert key not in out.out + out.err


def test_pod_facade_wait_ready_needs_settled_route(agent, monkeypatch):
    import asyncio
    from runpod_lifecycle.config import RunPodConfig
    from runpod_lifecycle.pod import Pod

    pattern = {0: True, 1: False, 2: True}
    with FlapProxy(agent.server_address[1], lambda seq, *_: pattern.get(seq, False)) as proxy:
        pod = Pod("fakepod", "test", RunPodConfig(api_key="test", transport="https",
                                                  https_token=TOKEN, template_id=""))
        monkeypatch.setattr(pod, "https_transport", lambda: client_for(proxy))

        async def status():
            return {"desired_status": "RUNNING", "ports": []}

        monkeypatch.setattr(pod, "status", status)
        assert asyncio.run(pod.wait_ready(timeout=10))["desired_status"] == "RUNNING"
    assert [f for _s, _m, _p, f in proxy.seen] == [True, False, True, False, False, False]
