"""Host-side bridge checks using loopback-only HTTP and WebSocket servers."""
import asyncio
import threading
from contextlib import contextmanager

import httpx
import pytest
from aiohttp import web
from websockets.sync.client import connect
from websockets.exceptions import ConnectionClosed

from runpod_lifecycle.engine_bridge import EngineBridge
from runpod_lifecycle.https import HttpsPodTransport
from runpod_lifecycle.pod_agent import AgentServer

TOKEN = "local-bridge-credential-" + "x" * 40


@pytest.fixture
def engine():
    loop = asyncio.new_event_loop()
    facts = {"requests": []}

    async def handle(request):
        facts["requests"].append((request.path_qs, dict(request.headers)))
        if request.path == "/ws":
            ws = web.WebSocketResponse(autoping=False, compress=False)
            await ws.prepare(request)
            await ws.send_str("ready")
            async for message in ws:
                if message.type == web.WSMsgType.TEXT:
                    await ws.send_str(message.data)
                elif message.type == web.WSMsgType.BINARY:
                    await ws.send_bytes(message.data)
                elif message.type == web.WSMsgType.PING:
                    await ws.pong(message.data)
            return ws
        if request.path == "/prompt":
            return web.Response(body=await request.read(), content_type="application/json")
        return web.json_response({"ok": True})

    async def start():
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        return runner, site._server.sockets[0].getsockname()[1]

    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    runner, facts["port"] = asyncio.run_coroutine_threadsafe(start(), loop).result(5)
    try:
        yield facts
    finally:
        asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(5)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(3)
        loop.close()


@contextmanager
def bridge_stack(tmp_path, engine):
    agent = AgentServer(("127.0.0.1", 0), TOKEN, tmp_path / "state")
    agent_thread = threading.Thread(target=agent.serve_forever, daemon=True)
    agent_thread.start()
    transport = HttpsPodTransport(
        "local", TOKEN, base_url=f"http://127.0.0.1:{agent.server_port}",
        poll_interval=0, route_grace=0)
    config = {"endpoint_id": "comfy", "port": engine["port"]}
    transport.enable_engine_proxy(**config)
    bridge = EngineBridge(transport, config)
    try:
        yield agent, transport, bridge
    finally:
        bridge.close()
        agent.shutdown()
        agent.server_close()
        agent_thread.join(3)


def test_bridge_http_websocket_authorization_filtering_and_origin(tmp_path, engine):
    with bridge_stack(tmp_path, engine) as (_, _, bridge):
        assert bridge.server.server_address == ("127.0.0.1", bridge.server.server_port)
        assert bridge.probe_get("/system_stats") == (200, b'{"ok": true}')
        # No local bearer is required: this listener trusts the single user's
        # loopback callers and the transport retains its own authenticated bearer.
        with httpx.Client(trust_env=False) as client:
            response = client.get(bridge.url + "/object_info", headers={
                "Cookie": "private", "X-Secret": TOKEN})
            assert response.status_code == 200 and response.json() == {"ok": True}
            assert client.post(bridge.url + "/prompt", content=b'{"prompt":{}}').content == b'{"prompt":{}}'
            assert client.get(bridge.url + "/object_info", headers={"Host": "attacker"}).status_code == 403
            assert client.get(bridge.url + "/object_info", headers={
                "Origin": "http://attacker"}).status_code == 403
        with connect(bridge.url.replace("http:", "ws:") + "/ws?clientId=test",
                     additional_headers={"Cookie": "private", "X-Secret": TOKEN},
                     proxy=None, compression=None) as ws:
            assert ws.recv(timeout=3) == "ready"
            ws.send("héllo")
            assert ws.recv(timeout=3) == "héllo"
            ws.send(b"\x00\xff")
            assert ws.recv(timeout=3) == b"\x00\xff"
            assert ws.ping(b"ping").wait(3)
        assert all("Authorization" not in headers and "Cookie" not in headers
                   and "X-Secret" not in headers for _, headers in engine["requests"])


def test_bridge_close_interrupts_active_websocket(tmp_path, engine):
    with bridge_stack(tmp_path, engine) as (_, _, bridge):
        with connect(bridge.url.replace("http:", "ws:") + "/ws", proxy=None,
                     compression=None) as ws:
            assert ws.recv(timeout=3) == "ready"
            bridge.close()
            with pytest.raises(ConnectionClosed):
                ws.recv(timeout=3)
