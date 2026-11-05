"""Origin restrictions and safe download diagnostics; no provider traffic."""
from __future__ import annotations

import threading
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from runpod_lifecycle.https import HttpsPodTransport, PodTransportError
from runpod_lifecycle.pod_agent import CHUNK

TOKEN = "test-only-secret-" + "x" * 40


def test_upload_client_matches_independent_protocol_fixture(tmp_path):
    source = tmp_path / "source"
    payload = b"x" * CHUNK + b"tail"
    source.write_bytes(payload)
    upload_id = "a" * 64
    calls = []
    committed = bytearray()

    def respond(request):
        calls.append((request.method, request.url.path))
        assert request.url.host == "pod1-8000.proxy.runpod.net"
        assert request.headers["Authorization"] == "Bearer " + TOKEN
        assert "transfer-encoding" not in request.headers
        if len(calls) == 1:
            assert calls[-1] == ("POST", "/uploads")
            assert json.loads(request.content) == {"path": "/tmp/fixture", "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(), "reserved": False}
        elif len(calls) in (2, 3):
            assert calls[-1] == ("PUT", "/uploads/" + upload_id)
            assert dict(request.url.params) == {"offset": str(len(committed))}
            assert int(request.headers["Content-Length"]) == len(request.content) <= CHUNK
            committed.extend(request.content)
        elif len(calls) == 4:
            assert calls[-1] == ("POST", "/uploads/" + upload_id)
            assert request.content == b""
            assert committed == payload
        else:
            assert len(calls) == 5 and calls[-1] == ("GET", "/uploads/" + upload_id)
        return httpx.Response(200, json={"id": upload_id, "offset": len(committed),
                                         "complete": len(calls) == 5, "error": None})

    client = HttpsPodTransport("pod1", TOKEN, poll_interval=0)
    client.http.close()
    client.http = httpx.Client(base_url="https://pod1-8000.proxy.runpod.net",
        headers={"Authorization": "Bearer " + TOKEN}, transport=httpx.MockTransport(respond),
        follow_redirects=False)
    try:
        client.put(source, "/tmp/fixture")
        assert len(calls) == 5
    finally:
        client.close()


def test_upload_reconciliation_failure_preserves_original_rejection(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"bounded chunk")
    client = HttpsPodTransport("pod1", TOKEN)
    chunk_error = PodTransportError("pod HTTPS request rejected (HTTP 403)")
    calls = []

    def request(method, path, **kwargs):
        calls.append(method)
        if method == "POST":
            return httpx.Response(200, json={"id": "a" * 64, "offset": 0, "complete": False})
        if method == "PUT":
            raise chunk_error
        raise PodTransportError("pod HTTPS request rejected (HTTP 404)")

    monkeypatch.setattr(client, "_request", request)
    try:
        with pytest.raises(PodTransportError) as caught:
            client.put(source, "/private/" + TOKEN)
        assert caught.value is chunk_error
        assert calls == ["POST", "PUT", "GET"]
        assert TOKEN not in str(caught.value)
    finally:
        client.close()


@pytest.mark.parametrize("status", [302, 403, 404, None])
def test_upload_failure_logging_omits_secrets_and_never_changes_route(tmp_path, caplog, status):
    source = tmp_path / "source"
    source.write_bytes(b"chunk")
    upload_id = "b" * 64
    requests = []

    def respond(request):
        requests.append((request.method, request.url.path))
        assert request.url.host == "pod1-8000.proxy.runpod.net"
        if request.method == "PUT":
            if status is None:
                raise httpx.ReadError(TOKEN, request=request)
            return httpx.Response(status, text=TOKEN,
                headers={"X-Secret": TOKEN, "Location": "https://attacker.example/" + TOKEN})
        return httpx.Response(200, json={"id": upload_id, "offset": 0, "complete": False})

    client = HttpsPodTransport("pod1", TOKEN, poll_interval=0)
    client.http.close()
    client.http = httpx.Client(base_url="https://pod1-8000.proxy.runpod.net",
        headers={"Authorization": "Bearer " + TOKEN}, transport=httpx.MockTransport(respond),
        follow_redirects=False)
    try:
        with pytest.raises(PodTransportError):
            client.put(source, "/private/" + TOKEN)
        assert requests == [("POST", "/uploads")] + [
            ("PUT", "/uploads/" + upload_id)] * (3 if status is None else 1) + [
            ("GET", "/uploads/" + upload_id)]
        messages = [r.getMessage() for r in caplog.records if r.name == "runpod_lifecycle.https"]
        assert messages == [f"pod HTTPS upload failed (stage=chunk HTTP {status or 'unavailable'})"]
        assert not any(secret in messages[0] for secret in (TOKEN, upload_id, "X-Secret", "attacker", "/private"))
    finally:
        client.close()


@pytest.mark.parametrize("url", [
    "https://attacker.example", "https://other-8000.proxy.runpod.net",
    "https://pod1-8001.proxy.runpod.net", "https://pod1-8000.proxy.runpod.net:8000",
    "https://pod1-8000.proxy.runpod.net.evil.example",
    "https://pod1-8000.proxy.runpod.net@attacker.example",
    "https://user:password@pod1-8000.proxy.runpod.net",
    "https://@pod1-8000.proxy.runpod.net", "https://pod1-8000.proxy.runpod.net:",
    "https://pod1-8000.proxy.runpod.net:bad", "https://pod1-8000.proxy.runpod.net:65536",
    "https://pod1-8000.proxy.runpod.net/prefix", "https://pod1-8000.proxy.runpod.net//",
    "https://pod1-8000.proxy.runpod.net/%2e%2e", "https://pod1-8000.proxy.runpod.net?",
    "https://pod1-8000.proxy.runpod.net#", "https://pod1-8000.proxy.runpod.net?token=" + TOKEN,
    "https://pod1-8000.proxy.runpod.net/#" + TOKEN,
    "https://pod1-8000.proxy.runpod.net\\@attacker.example",
    " https://pod1-8000.proxy.runpod.net", "https://pod1-8000.proxy.runpod.net\n",
    "http://pod1-8000.proxy.runpod.net", "https://", "https://[bad", "",
    "http://localhost.evil.example", "http://127.0.0.1/prefix",
])
def test_untrusted_override_rejected_before_client_or_bearer_creation(url, monkeypatch):
    def forbidden_client(**_kwargs):
        pytest.fail("invalid origin reached HTTP client construction")
    monkeypatch.setattr(httpx, "Client", forbidden_client)
    with pytest.raises(ValueError) as caught:
        HttpsPodTransport("pod1", TOKEN, base_url=url)
    assert TOKEN not in str(caught.value)
    assert "attacker" not in str(caught.value)
    assert "password" not in str(caught.value)


@pytest.mark.parametrize("url", [
    None, "https://pod1-8000.proxy.runpod.net", "https://pod1-8000.proxy.runpod.net/",
    "https://pod1-8000.proxy.runpod.net:443/", "http://localhost:8123/",
    "http://127.0.0.1:8123", "http://[::1]:8123", "https://localhost:8123",
])
def test_exact_origin_and_explicit_loopback_overrides(url):
    with_client = HttpsPodTransport("pod1", TOKEN, base_url=url)
    with_client.close()


def test_override_still_validates_pod_id_and_selected_agent_port():
    with pytest.raises(ValueError):
        HttpsPodTransport("bad/pod", TOKEN, base_url="http://localhost:8123")
    with pytest.raises(ValueError):
        HttpsPodTransport("pod1", TOKEN, port=0, base_url="http://localhost:8123")
    client = HttpsPodTransport("pod1", TOKEN, port=9000,
                               base_url="https://pod1-9000.proxy.runpod.net")
    client.close()
    with pytest.raises(ValueError):
        HttpsPodTransport("pod1", TOKEN, port=9000,
                          base_url="https://pod1-8000.proxy.runpod.net")


def test_redirect_target_never_receives_request_or_bearer(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    requests = []
    class Redirect(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            requests.append(self.path)
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/sink?{TOKEN}")
            self.send_header("X-Secret", TOKEN)
            self.send_header("Content-Length", "0")
            self.end_headers()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HttpsPodTransport("pod1", TOKEN, base_url=f"http://127.0.0.1:{server.server_port}")
    try:
        with pytest.raises(PodTransportError, match="HTTP 302") as caught:
            client.connect()
        assert requests == ["/health"]
        assert TOKEN not in str(caught.value)
        assert "/sink" not in str(caught.value)
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(2)


@pytest.mark.parametrize("ranges,status", [(False, 403), (True, 503), (False, 302)])
def test_download_rejection_has_safe_context_and_no_response_secrets(tmp_path, ranges, status):
    client = HttpsPodTransport("pod1", TOKEN, poll_interval=0)
    client.http.close()
    def response(request):
        assert request.headers["Authorization"] == "Bearer " + TOKEN
        return httpx.Response(status, text=TOKEN, headers={"X-Secret": TOKEN, "Location": "https://evil/" + TOKEN})
    client.http = httpx.Client(base_url="https://pod1-8000.proxy.runpod.net",
                               headers={"Authorization": "Bearer " + TOKEN},
                               transport=httpx.MockTransport(response), follow_redirects=False)
    try:
        with pytest.raises(PodTransportError) as caught:
            client.get("/private/" + TOKEN, tmp_path / "output", use_ranges=ranges)
        message = str(caught.value)
        assert f"HTTP {status}" in message and "operation=download" in message
        assert f"framing={'range' if ranges else 'json'}" in message and "offset=0" in message
        assert TOKEN not in message and "X-Secret" not in message and "evil" not in message
        assert not (tmp_path / "output").exists()
    finally:
        client.close()


@pytest.mark.parametrize("ranges", [False, True])
def test_malformed_download_framing_is_sanitized(tmp_path, monkeypatch, ranges):
    client = HttpsPodTransport("pod1", TOKEN)
    response = httpx.Response(206 if ranges else 200, text=TOKEN,
                              headers={"Content-Range": TOKEN, "ETag": TOKEN})
    monkeypatch.setattr(client, "_request", lambda *_a, **_kw: response)
    try:
        with pytest.raises(PodTransportError) as caught:
            client.get("/private/" + TOKEN, tmp_path / "output", use_ranges=ranges)
        message = str(caught.value)
        assert "operation=download" in message and "offset=0" in message
        assert f"HTTP {response.status_code}" in message
        assert TOKEN not in message and "Content-Range" not in message and "ETag" not in message
        assert not (tmp_path / "output").exists()
    finally:
        client.close()


def test_transport_exception_does_not_disclose_request_or_token(tmp_path):
    client = HttpsPodTransport("pod1", TOKEN, poll_interval=0)
    client.http.close()
    def fail(request):
        raise httpx.ReadError(TOKEN, request=request)
    client.http = httpx.Client(base_url="https://pod1-8000.proxy.runpod.net",
                               transport=httpx.MockTransport(fail))
    try:
        with pytest.raises(PodTransportError) as caught:
            client.get("/private/" + TOKEN, tmp_path / "output")
        assert "HTTP unavailable" in str(caught.value) and "framing=json offset=0" in str(caught.value)
        assert TOKEN not in str(caught.value) and caught.value.__suppress_context__
    finally:
        client.close()


def test_later_chunk_failure_reports_committed_offset_without_remote_metadata(tmp_path):
    import base64
    import hashlib

    client = HttpsPodTransport("pod1", TOKEN)
    client.http.close()
    offsets = []
    def response(request):
        import json
        offset = json.loads(request.content)["offset"]
        offsets.append(offset)
        if offset:
            return httpx.Response(412, text=TOKEN, headers={"ETag": TOKEN})
        chunk = b"x" * CHUNK
        return httpx.Response(200, json={"offset": 0, "total": CHUNK + 1, "etag": TOKEN,
            "sha256": hashlib.sha256(chunk).hexdigest(), "data": base64.b64encode(chunk).decode()})
    client.http = httpx.Client(base_url="https://pod1-8000.proxy.runpod.net",
                               transport=httpx.MockTransport(response))
    dest = tmp_path / "output"
    dest.write_bytes(b"previous complete output")
    try:
        with pytest.raises(PodTransportError) as caught:
            client.get("/private/" + TOKEN, dest)
        assert offsets == [0, CHUNK]
        assert f"offset={CHUNK} HTTP 412" in str(caught.value)
        assert TOKEN not in str(caught.value)
        assert dest.read_bytes() == b"previous complete output"
    finally:
        client.close()
