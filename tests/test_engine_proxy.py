"""Local sockets only; these tests do not qualify provider availability."""
import asyncio
import json
import threading
from contextlib import contextmanager

import httpx
import pytest
from aiohttp import web

from runpod_lifecycle import pod_agent
from runpod_lifecycle.https import agent_bootstrap
from runpod_lifecycle.pod_agent import AgentServer, EngineProxy

TOKEN = "local-test-token-" + "x" * 40


@pytest.fixture(autouse=True)
def no_websocket_constructors(monkeypatch):
    import websockets
    import websockets.sync.client
    import websockets.asyncio.client
    def forbidden(*args, **kwargs):
        raise AssertionError("HTTP-only proxy attempted a WebSocket")
    monkeypatch.setattr(websockets, "connect", forbidden)
    monkeypatch.setattr(websockets.sync.client, "connect", forbidden)
    monkeypatch.setattr(websockets.sync.client, "ClientConnection", forbidden)
    monkeypatch.setattr(websockets.asyncio.client, "connect", forbidden)
    monkeypatch.setattr(web, "WebSocketResponse", forbidden)


@pytest.fixture
def comfy():
    loop = asyncio.new_event_loop()
    facts = {"requests": [], "redirect": None, "delay": 0, "body": bytes(range(256)) * 12000}

    async def handle(request):
        facts["requests"].append((request.method, request.path_qs, dict(request.headers)))
        if facts["delay"]:
            await asyncio.sleep(facts["delay"])
        if facts.get("not_found") is not None and request.path.startswith("/history"):
            framing, body = facts["not_found"]
            if framing == "declared":
                return web.Response(status=404, body=body)
            response = web.StreamResponse(status=404)
            response.enable_chunked_encoding()
            await response.prepare(request)
            if body:
                await response.write(body)
            await response.write_eof()
            return response
        if facts["redirect"]:
            return web.Response(status=307, headers={"Location": facts["redirect"]})
        if request.path == "/view":
            response = web.StreamResponse(headers={"Content-Type": "image/png"})
            await response.prepare(request)
            for offset in range(0, len(facts["body"]), 8192):
                await response.write(facts["body"][offset:offset + 8192])
            await response.write_eof()
            return response
        if request.path.startswith("/upload/"):
            facts["upload"] = await request.read()
            return web.json_response({"name": "input.png", "bytes": len(facts["upload"])})
        if request.path == "/prompt":
            facts["prompt"] = await request.json()
            return web.json_response({"prompt_id": "prompt-1", "node_errors": {}})
        if request.path == "/object_info":
            return web.json_response({"ExampleNode": {"input": {"required": {}}}})
        if request.path == "/system_stats":
            return web.json_response({"system": {"comfyui_version": "0.26.0"}})
        if request.path.startswith("/history"):
            return web.json_response({"prompt-1": {"outputs": {}}})
        return web.json_response({"ok": True})

    async def start():
        app = web.Application(client_max_size=16 * 1024 * 1024)
        app.router.add_route("*", "/{tail:.*}", handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        return runner, site._server.sockets[0].getsockname()[1]

    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    runner, facts["port"] = asyncio.run_coroutine_threadsafe(start(), loop).result(10)
    try:
        yield facts
    finally:
        asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()


@contextmanager
def agent_server(tmp_path, comfy, *, enable=True, **kwargs):
    server = AgentServer(("127.0.0.1", 0), TOKEN, tmp_path / "state",
                         engine_proxy={"endpoint_id": "single", "port": comfy["port"]} if enable else None, **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}/engines/single"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_required_http_operations_and_rejected_raw_media(tmp_path, comfy):
    with agent_server(tmp_path, comfy) as (_, url), httpx.Client(
            headers={"Authorization": "Bearer " + TOKEN}, trust_env=False, timeout=10) as client:
        assert "ExampleNode" in client.get(url + "/object_info").json()
        prompt = {"prompt": {"1": {"class_type": "ExampleNode", "inputs": {}}}}
        assert client.post(url + "/prompt", json=prompt).json()["prompt_id"] == "prompt-1"
        assert comfy["prompt"] == prompt
        assert "prompt-1" in client.get(url + "/history/prompt-1").json()
        for method, path in [("GET", "/system_stats"), ("GET", "/queue"),
                             ("POST", "/queue"), ("POST", "/interrupt"), ("POST", "/api/free")]:
            assert client.request(method, url + path, json={} if method == "POST" else None).status_code == 200
        for method, path in [("GET", "/view"), ("POST", "/upload/image"), ("POST", "/upload/mask")]:
            assert client.request(method, url + path).status_code == 403
    assert all("Authorization" not in headers and "Cookie" not in headers
               for _, _, headers in comfy["requests"])


def test_websocket_upgrade_and_range_explicitly_rejected(tmp_path, comfy):
    with agent_server(tmp_path, comfy) as (_, url), httpx.Client(
            headers={"Authorization": "Bearer " + TOKEN}, trust_env=False) as client:
        assert client.get(url + "/ws?clientId=test").status_code == 403
        assert client.get(url + "/api/ws").status_code == 403
        for headers in ({"Upgrade": "websocket", "Connection": "Upgrade"}, {"Range": "bytes=0-4"},
                        {"Upgrade": "h2c"}, {"Connection": "upgrade"}):
            assert client.get(url + "/object_info", headers=headers).status_code == 403
    assert comfy["requests"] == []


@pytest.mark.parametrize("auth", [None, "Bearer wrong", "Basic " + TOKEN])
def test_http_and_ws_require_bearer_before_upstream(tmp_path, comfy, auth):
    headers = {} if auth is None else {"Authorization": auth}
    with agent_server(tmp_path, comfy) as (_, url):
        with httpx.Client(trust_env=False) as client:
            for method, path in [("GET", "/object_info"), ("POST", "/upload/image"),
                                 ("POST", "/prompt"), ("GET", "/history/x"), ("GET", "/view")]:
                assert client.request(method, url + path, headers=headers).status_code == 401
        with httpx.Client(trust_env=False) as client:
            assert client.get(url + "/ws", headers={**headers, "Upgrade": "websocket"}).status_code == 401
    assert comfy["requests"] == []


@pytest.mark.parametrize("path", ["/engines/other/object_info", "/engines/single/../object_info",
                                    "/engines/single/http://foreign/view", "/engines/single/%2fview",
                                    "/engines/single/jobs", "/object_info"])
def test_single_slot_no_arbitrary_destination(tmp_path, comfy, path):
    with agent_server(tmp_path, comfy) as (server, _), httpx.Client(trust_env=False) as client:
        response = client.get(f"http://127.0.0.1:{server.server_port}" + path,
                              headers={"Authorization": "Bearer " + TOKEN})
        assert response.status_code in {403, 404}
    assert comfy["requests"] == []


def test_redirect_is_not_forwarded_even_with_auth_following_client(tmp_path, comfy):
    with agent_server(tmp_path, comfy) as (_, url):
        with agent_server(tmp_path / "recipient", comfy) as (recipient, _):
            comfy["redirect"] = f"http://127.0.0.1:{recipient.server_port}/health"
            with httpx.Client(trust_env=False, follow_redirects=True) as client:
                response = client.get(url + "/object_info", headers={"Authorization": "Bearer " + TOKEN})
            assert response.status_code == 502 and "Location" not in response.headers
            assert len(comfy["requests"]) == 1


def test_observation_requires_explicit_trusted_readiness_handoff(tmp_path, comfy):
    policy = {"schema": pod_agent.OBSERVATION_SCHEMA, "roots": [], "files": [str(tmp_path / "ready.json")]}
    with agent_server(tmp_path, comfy, enable=False, observation_policy=policy) as (server, url):
        assert httpx.get(url + "/object_info", headers={"Authorization": "Bearer " + TOKEN},
                         trust_env=False).status_code == 403
        assert comfy["requests"] == []
        server.enable_engine("single", comfy["port"])
        server.enable_engine("single", comfy["port"])  # same slot, no second engine
        with pytest.raises(ValueError, match="already bound"):
            server.enable_engine("replacement", comfy["port"])
        assert httpx.get(url + "/object_info", headers={"Authorization": "Bearer " + TOKEN},
                         trust_env=False).status_code == 200
        assert httpx.post(f"http://127.0.0.1:{server.server_port}/jobs", json={},
                          headers={"Authorization": "Bearer " + TOKEN}, trust_env=False).status_code == 403
    boot = agent_bootstrap(TOKEN, observation_policy=policy,
                           engine_proxy={"endpoint_id": "single", "port": comfy["port"]})
    assert pod_agent.OBSERVATION_ENV in boot["env"] and pod_agent.ENGINE_ENV in boot["env"]



@pytest.mark.parametrize("config", [{"endpoint_id": "../x", "port": 8188},
                                      {"endpoint_id": "single", "port": True},
                                      {"endpoint_id": "single", "port": 65536},
                                      {"endpoint_id": "single", "port": 8188, "host": "example.com"}])
def test_only_fixed_loopback_upstream_is_configurable(config):
    with pytest.raises((TypeError, ValueError)):
        EngineProxy(**config)


def test_bootstrap_exposes_only_agent_port_and_snapshots_slot():
    config = {"endpoint_id": "single", "port": 8188}
    boot = agent_bootstrap(TOKEN, port=8000, engine_proxy=config)
    config["port"] = 9999
    assert boot["ports"] == "8000/http"
    assert json.loads(boot["env"][pod_agent.ENGINE_ENV]) == {"endpoint_id": "single", "port": 8188}


def test_duplicate_bearer_and_oversized_upload_refused_before_upstream(tmp_path, comfy):
    import http.client
    with agent_server(tmp_path, comfy) as (server, url):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        connection.putrequest("GET", "/engines/single/object_info")
        connection.putheader("Authorization", "Bearer " + TOKEN)
        connection.putheader("Authorization", "Bearer wrong")
        connection.endheaders()
        assert connection.getresponse().status == 401
        connection.close()
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        connection.putrequest("POST", "/engines/single/prompt")
        connection.putheader("Authorization", "Bearer " + TOKEN)
        connection.putheader("Content-Length", str(pod_agent.ENGINE_BODY_LIMIT + 1))
        connection.endheaders()
        assert connection.getresponse().status == 400
        connection.close()
    assert comfy["requests"] == []


def test_engine_http_wait_is_short_and_bounded(tmp_path, comfy, monkeypatch):
    import time
    assert pod_agent.ENGINE_TRANSFER_SECONDS == 15
    monkeypatch.setattr(pod_agent, "ENGINE_TRANSFER_SECONDS", 0.15)
    with agent_server(tmp_path, comfy) as (_, url):
        comfy["delay"] = 0.5
        started = time.monotonic()
        response = httpx.get(url + "/object_info", headers={"Authorization": "Bearer " + TOKEN},
                             trust_env=False, timeout=3)
        assert response.status_code == 502 and time.monotonic() - started < 1


@pytest.mark.parametrize("startup", ["sdk", "rest"])
def test_real_isolated_inline_engine_startup(tmp_path, comfy, startup):
    import os
    import shlex
    import socket
    import subprocess
    import sys
    from runpod_lifecycle.https import HttpsPodTransport

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    policy = {"schema": pod_agent.OBSERVATION_SCHEMA, "roots": [], "files": [str(tmp_path / "ready.json")]}
    boot = agent_bootstrap(TOKEN, port, observation_policy=policy,
                           engine_proxy={"endpoint_id": "single", "port": comfy["port"]})
    command = (shlex.split(boot["docker_args"]) if startup == "sdk"
               else boot["docker_entrypoint"] + boot["docker_start_cmd"])
    command[-1] = command[-1].replace("python3 -u", shlex.quote(sys.executable) + " -S -u")
    environment = dict(os.environ, **boot["env"], ASTRID_POD_STATE=str(tmp_path / "state"))
    process = subprocess.Popen(command, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    client = HttpsPodTransport("local", TOKEN, base_url=f"http://127.0.0.1:{port}", request_timeout=.2, poll_interval=.02)
    try:
        client.wait_ready(timeout=5)
        root = f"http://127.0.0.1:{port}"
        with httpx.Client(trust_env=False, headers={"Authorization": "Bearer " + TOKEN}) as http:
            assert "ExampleNode" in http.get(root + "/engines/single/object_info").json()
            assert http.post(root + "/jobs", json={}).status_code == 403
        with httpx.Client(trust_env=False, headers={"Authorization": "Bearer " + TOKEN}) as http:
            assert http.get(root + "/engines/single/ws", headers={"Upgrade": "websocket"}).status_code == 403
    finally:
        client.disconnect()
        process.terminate()
        process.wait(timeout=5)
        process.stderr.close()


def test_trusted_transfer_handoff_exact_writes_and_output_json_only(tmp_path, comfy):
    import hashlib
    from runpod_lifecycle.https import HttpsPodTransport, PodTransportError
    input_root, output_root = tmp_path / "input", tmp_path / "output"
    input_root.mkdir()
    output_root.mkdir()
    source = tmp_path / "approved.png"
    source.write_bytes(b"approved-input" * 100000)
    target = input_root / "approved.png"
    output = output_root / "result.png"
    output.write_bytes(b"accepted-output" * 200000)
    policy = {"schema": pod_agent.OBSERVATION_SCHEMA, "roots": [], "files": [str(tmp_path / "ready.json")]}
    with agent_server(tmp_path, comfy, enable=False, observation_policy=policy) as (server, _):
        origin = f"http://127.0.0.1:{server.server_port}"
        with HttpsPodTransport("local", TOKEN, base_url=origin, poll_interval=.01) as client:
            with pytest.raises(PodTransportError, match="403"):
                client.put(source, str(target))
            assert not (tmp_path / "state").exists()
            with pytest.raises(PodTransportError, match="403"):
                client.get(str(output), tmp_path / "download")
            server.enable_engine("single", comfy["port"], input_files=[str(target)], output_roots=[str(output_root)])
            server.enable_engine("single", comfy["port"], input_files=[str(target)], output_roots=[str(output_root)])
            with pytest.raises(ValueError, match="already bound"):
                server.enable_engine("single", comfy["port"], input_files=[str(input_root / "other.png")])
            client.put(source, str(target))
            assert target.read_bytes() == source.read_bytes()
            local = tmp_path / "download"
            client.get(str(output), local, use_ranges=False)
            assert hashlib.sha256(local.read_bytes()).digest() == hashlib.sha256(output.read_bytes()).digest()
            for path in (input_root / "other.png", output_root / "arbitrary.png", tmp_path / "arbitrary"):
                with pytest.raises(PodTransportError, match="403"):
                    client.put(source, str(path))
                assert not path.exists()
            with pytest.raises(PodTransportError, match="403"):
                client.get(str(source), tmp_path / "denied")
        with httpx.Client(trust_env=False, headers={"Authorization": "Bearer " + TOKEN}) as http:
            assert http.post(origin + "/jobs", json={"id": "a" * 32, "command": "true"}).status_code == 403
            assert http.get(origin + "/files", params={"path": str(output)},
                            headers={"Range": "bytes=0-9"}).status_code == 403
            assert http.post(origin + "/engines/single/config", json={}).status_code == 403


@pytest.mark.parametrize("unsafe", ["symlink", "parent-symlink", "hardlink", "reserved", "noncanonical"])
def test_configured_upload_paths_cannot_escape_or_write_secrets(tmp_path, comfy, unsafe):
    from runpod_lifecycle.https import HttpsPodTransport, PodTransportError
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    target = inputs / "input.png"
    victim = tmp_path / "private-token"
    victim.write_bytes(b"unchanged")
    if unsafe == "symlink":
        target.symlink_to(victim)
    elif unsafe == "hardlink":
        import os
        os.link(victim, target)
    elif unsafe == "parent-symlink":
        real = tmp_path / "real"
        inputs.rmdir()
        real.mkdir()
        inputs.symlink_to(real, target_is_directory=True)
    source = tmp_path / "source"
    source.write_bytes(b"approved")
    policy = {"schema": pod_agent.OBSERVATION_SCHEMA, "roots": [], "files": [str(victim)]}
    with agent_server(tmp_path, comfy, enable=False, observation_policy=policy) as (server, _):
        server.enable_engine("single", comfy["port"], input_files=[str(target)])
        with HttpsPodTransport("local", TOKEN, base_url=f"http://127.0.0.1:{server.server_port}", poll_interval=.01) as client:
            remote = str(target).replace("/inputs/", "/inputs//") if unsafe == "noncanonical" else str(target)
            with pytest.raises(PodTransportError):
                client.put(source, remote, reserved=unsafe == "reserved")
    assert victim.read_bytes() == b"unchanged"


@pytest.mark.parametrize("framing", ["declared", "chunked"])
@pytest.mark.parametrize("body", [b"", b"404: Not Found"])
def test_agent_never_relays_the_provider_empty_404_signature(tmp_path, comfy, framing, body):
    """Clients retry only an EMPTY 404 (provider no-route). Agent 404s are final."""
    from runpod_lifecycle.https import HttpsPodTransport
    comfy["not_found"] = (framing, body)
    with agent_server(tmp_path, comfy) as (_, url), httpx.Client(
            headers={"Authorization": "Bearer " + TOKEN}, trust_env=False, timeout=10) as client:
        response = client.get(url + "/history/missing")
    assert response.status_code == 404 and response.content
    assert not HttpsPodTransport.route_unavailable(response)
    if body:
        assert response.content == body
    else:
        assert response.json() == {"error": "engine resource not found"}
    assert len([r for r in comfy["requests"] if r[1] == "/history/missing"]) == 1


PROXY_NAMES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def test_loopback_gateway_client_stays_local_and_pod_origin_keeps_env_egress(tmp_path, comfy, monkeypatch):
    from runpod_lifecycle.https import HttpsPodTransport
    for name in PROXY_NAMES:
        # A discard-port proxy: any request through it fails.
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    with agent_server(tmp_path, comfy) as (server, _):
        with HttpsPodTransport("local", TOKEN, base_url=f"http://127.0.0.1:{server.server_port}",
                               poll_interval=.01) as local:
            assert local.http.trust_env is False
            local.connect()
    remote = HttpsPodTransport("pod123", TOKEN)
    try:
        # The exact pod origin keeps the supported controller egress settings.
        assert remote.http.trust_env is True
    finally:
        remote.disconnect()


def test_caller_deadline_bounds_route_grace_retries():
    import http.server
    import time
    from runpod_lifecycle.https import HttpsPodTransport, PodTransportError
    seen = []

    class NoRoute(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.path)
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), NoRoute)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = HttpsPodTransport("local", TOKEN, base_url=f"http://127.0.0.1:{server.server_port}",
                                   poll_interval=.05, route_grace=90, deadline=time.monotonic() + 1)
        started = time.monotonic()
        with pytest.raises(PodTransportError, match="caller deadline exceeded"):
            client.connect()
        assert time.monotonic() - started < 3
        assert len(seen) >= 2
        client.disconnect()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
