"""Opt-in session surface; all traffic stays on local test servers."""
import asyncio
import threading
from contextlib import contextmanager

import httpx
import pytest
from aiohttp import web
from websockets.sync.client import connect

from runpod_lifecycle.https import HttpsPodTransport, PodTransportError
from runpod_lifecycle.pod_agent import AgentServer, OBSERVATION_SCHEMA

TOKEN = 'synthetic-session-claim-' + 'x' * 40


@contextmanager
def running_agent(tmp_path, **options):
    server = AgentServer(('127.0.0.1', 0), TOKEN, tmp_path / 'state', **options)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    transport = HttpsPodTransport('local', TOKEN, base_url=f'http://127.0.0.1:{server.server_port}',
                                  poll_interval=0, route_grace=0)
    try:
        yield server, transport
    finally:
        transport.disconnect()
        server.shutdown()
        server.server_close()
        thread.join(3)


@pytest.fixture
def comfy():
    loop = asyncio.new_event_loop()
    facts = {'requests': [], 'body': bytes(range(256)) * 8193, 'prompts': []}
    async def handle(request):
        facts['requests'].append((request.path_qs, dict(request.headers)))
        if request.path.endswith('/ws'):
            ws = web.WebSocketResponse(autoping=False, compress=False)
            await ws.prepare(request)
            await ws.send_str('early')
            async for message in ws:
                if message.type == web.WSMsgType.TEXT:
                    await ws.send_str(message.data)
                elif message.type == web.WSMsgType.BINARY:
                    await ws.send_bytes(message.data)
                elif message.type == web.WSMsgType.PING:
                    await ws.pong(message.data)
            facts['closed'] = True
            return ws
        if request.path == '/prompt':
            facts['prompts'].append(await request.read())
            return web.json_response({'prompt_id': 'p1'})
        if request.path == '/upload/image':
            facts.setdefault('uploads', []).append((await request.read(), request.headers.get('Content-Type')))
            if facts.get('upload_delay'):
                await asyncio.sleep(facts['upload_delay'])
            return web.json_response({'name': 'input.png', 'subfolder': '', 'type': 'input'})
        if request.path == '/view':
            response = web.StreamResponse(headers={'Content-Type': 'image/png'})
            await response.prepare(request)
            for offset in range(0, len(facts['body']), 8192):
                await response.write(facts['body'][offset:offset + 8192])
            await response.write_eof()
            return response
        return web.json_response({'ok': True})
    async def start():
        app = web.Application()
        app.router.add_route('*', '/{tail:.*}', handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        return runner, site._server.sockets[0].getsockname()[1]
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    runner, facts['port'] = asyncio.run_coroutine_threadsafe(start(), loop).result(5)
    try:
        yield facts
    finally:
        asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(5)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(3)
        loop.close()


def test_session_enable_binary_http_and_no_bearer_projection(tmp_path, comfy, caplog, monkeypatch):
    monkeypatch.setenv('ASTRID_POD_TOKEN', TOKEN)
    with running_agent(tmp_path) as (server, transport):
        transport.enable_engine_proxy('comfy', comfy['port'])
        slot = server.engine_proxy
        transport.enable_engine_proxy('comfy', comfy['port'])
        assert server.engine_proxy.configuration() == slot.configuration()
        assert not (tmp_path / 'state' / 'engine-bearer.token').exists()
        url = str(transport.http.base_url) + '/engines/comfy'
        with httpx.Client(trust_env=False, headers={'Authorization': 'Bearer ' + TOKEN}) as client:
            assert client.get(url + '/object_info').json() == {'ok': True}
            payload = b'{"prompt":{"binary":"\\u0000"}}'
            assert client.post(url + '/prompt', content=payload).json() == {'prompt_id': 'p1'}
            response = client.get(url + '/view?filename=image.png&type=output', headers={
                'Cookie': 'private-cookie', 'X-Forwarded-Host': 'attacker', 'X-Secret': TOKEN})
            assert response.status_code == 200 and response.content == comfy['body']
            assert response.headers['Content-Type'] == 'image/png'
        assert comfy['prompts'] == [payload]
        assert all(not any(k.lower() in {'authorization', 'cookie', 'x-secret', 'x-forwarded-host'}
                           for k in headers) for _, headers in comfy['requests'])
        code, output, _ = transport.execute_command("printf '%s' \"${ASTRID_POD_TOKEN-unset}\"")
        assert code == 0 and output == 'unset'
        assert TOKEN not in caplog.text
        assert all(TOKEN.encode() not in p.read_bytes() for p in tmp_path.rglob('*') if p.is_file())


@pytest.mark.parametrize('auth', [None, 'Bearer wrong', ['Bearer ' + TOKEN, 'Bearer ' + TOKEN]])
def test_session_enable_and_ws_auth_before_upstream(tmp_path, comfy, auth):
    with running_agent(tmp_path) as (_, transport):
        headers = [] if auth is None else [('Authorization', value) for value in (auth if isinstance(auth, list) else [auth])]
        origin = str(transport.http.base_url)
        with httpx.Client(trust_env=False) as client:
            assert client.post(origin + '/engine-proxy', headers=headers,
                json={'endpoint_id': 'comfy', 'port': comfy['port']}).status_code == 401
        transport.enable_engine_proxy('comfy', comfy['port'])
        with httpx.Client(trust_env=False) as client:
            assert client.get(origin + '/engines/comfy/ws', headers=headers + [('Upgrade', 'websocket')]).status_code == 401
        assert comfy['requests'] == []


def test_session_enable_rejects_invalid_conflicting_and_observation_claims(tmp_path, comfy):
    with running_agent(tmp_path) as (_, transport):
        for config in ({'endpoint_id': 'comfy', 'port': True}, {'endpoint_id': '../bad', 'port': 8188},
                       {'endpoint_id': 'comfy', 'port': 0}, {'endpoint_id': 'comfy', 'port': 8188, 'host': 'foreign'}):
            assert transport.http.post('/engine-proxy', json=config).status_code == 400
        transport.enable_engine_proxy('comfy', comfy['port'])
        with pytest.raises(PodTransportError, match='already bound'):
            transport.enable_engine_proxy('other', comfy['port'])
        for path in ('/jobs', '/upload/mask', '/api/upload/image', '/%2fview', '/http://foreign/view'):
            assert transport.http.get('/engines/comfy' + path).status_code == 403
        assert transport.http.get('/engines/comfy/view', headers={'Range': 'bytes=0-4'}).status_code == 403
    policy = {'schema': OBSERVATION_SCHEMA, 'files': [str(tmp_path / 'public.json')], 'roots': []}
    with running_agent(tmp_path / 'observation', observation_policy=policy) as (server, transport):
        with pytest.raises(PodTransportError, match='command authority'):
            transport.enable_engine_proxy('comfy', comfy['port'])
        assert server.engine_proxy is None


def test_session_websocket_frames_ping_close_and_header_filtering(tmp_path, comfy):
    with running_agent(tmp_path) as (_, transport):
        transport.enable_engine_proxy('comfy', comfy['port'])
        url = str(transport.http.base_url).replace('http:', 'ws:') + '/engines/comfy/ws?clientId=session'
        with connect(url, additional_headers={'Authorization': 'Bearer ' + TOKEN,
                     'Cookie': 'private', 'X-Secret': TOKEN}, proxy=None, compression=None,
                     max_size=4 * 1024 * 1024) as ws:
            assert ws.recv(timeout=3) == 'early'
            ws.send('héllo')
            assert ws.recv(timeout=3) == 'héllo'
            payload = bytes(range(256)) * 5000
            ws.send(payload)
            assert ws.recv(timeout=3) == payload
            assert ws.ping(b'ping').wait(3)
        assert all('Authorization' not in h and 'Cookie' not in h and 'X-Secret' not in h
                   for _, h in comfy['requests'])


def test_session_upload_image_is_bounded_and_relayed_without_client_secrets(tmp_path, comfy, monkeypatch):
    with running_agent(tmp_path) as (_, transport):
        transport.enable_engine_proxy('comfy', comfy['port'])
        url = str(transport.http.base_url) + '/engines/comfy'
        payload = b'--upload\r\nContent-Disposition: form-data; name="image"; filename="input.png"\r\nContent-Type: image/png\r\n\r\nPNG\x00bytes\r\n--upload--\r\n'
        with httpx.Client(trust_env=False) as client:
            response = client.post(url + '/upload/image?overwrite=true', content=payload, headers={
                'Authorization': 'Bearer ' + TOKEN,
                'Content-Type': 'multipart/form-data; boundary=upload',
                'Cookie': 'private', 'X-Secret': TOKEN})
            assert response.status_code == 200
            assert response.json() == {'name': 'input.png', 'subfolder': '', 'type': 'input'}
            assert client.post(url + '/upload/mask', content=payload,
                               headers={'Authorization': 'Bearer ' + TOKEN}).status_code == 403
            assert client.post(url + '/api/upload/image', content=payload,
                               headers={'Authorization': 'Bearer ' + TOKEN}).status_code == 403
            assert client.get(url + '/upload/image', headers={
                'Authorization': 'Bearer ' + TOKEN}).status_code == 403
            unauthenticated = client.post(url + '/upload/image', content=payload,
                                          headers={'Content-Type': 'multipart/form-data; boundary=upload'})
            assert unauthenticated.status_code == 401
        import http.client
        for framing in ('oversized', 'chunked'):
            connection = http.client.HTTPConnection(transport.http.base_url.host,
                                                     transport.http.base_url.port, timeout=3)
            connection.putrequest('POST', '/engines/comfy/upload/image')
            connection.putheader('Authorization', 'Bearer ' + TOKEN)
            connection.putheader('Content-Type', 'multipart/form-data; boundary=upload')
            if framing == 'oversized':
                connection.putheader('Content-Length', str(1024 * 1024 * 1024 + 1))
            else:
                connection.putheader('Transfer-Encoding', 'chunked')
            connection.endheaders()
            response = connection.getresponse()
            assert response.status == 400
            response.read()
            connection.close()
        assert comfy['uploads'][0] == (payload, 'multipart/form-data; boundary=upload')
        import time
        from runpod_lifecycle import pod_agent
        assert pod_agent.ENGINE_TRANSFER_SECONDS == 15
        monkeypatch.setattr(pod_agent, 'ENGINE_TRANSFER_SECONDS', 0.15)
        comfy['upload_delay'] = 0.5
        started = time.monotonic()
        timed_out = httpx.post(url + '/upload/image', content=payload, headers={
            'Authorization': 'Bearer ' + TOKEN,
            'Content-Type': 'multipart/form-data; boundary=upload'}, trust_env=False, timeout=3)
        assert timed_out.status_code == 502 and time.monotonic() - started < 1
        assert len(comfy['requests']) == 2
        assert all('Authorization' not in headers and 'Cookie' not in headers and 'X-Secret' not in headers
                   for _, headers in comfy['requests'])


@pytest.mark.parametrize('status,reason', [
    (400, 'invalid engine proxy request'),
    (403, 'requires command authority'),
    (404, 'does not support session engine proxy'),
    (409, 'already bound'),
])
def test_session_enable_transport_rejections_are_specific_and_sanitized(status, reason, caplog):
    transport = HttpsPodTransport('pod1', TOKEN, route_grace=0)
    transport.http.close()
    seen = []
    def respond(request):
        seen.append(request)
        assert request.headers['Authorization'] == 'Bearer ' + TOKEN
        return httpx.Response(status, text=TOKEN, headers={'Location': 'https://foreign/' + TOKEN})
    transport.http = httpx.Client(base_url='https://pod1-8000.proxy.runpod.net',
        headers={'Authorization': 'Bearer ' + TOKEN}, transport=httpx.MockTransport(respond))
    try:
        with pytest.raises(PodTransportError, match=reason) as caught:
            transport.enable_engine_proxy('comfy', 8188)
        assert TOKEN not in str(caught.value) and TOKEN not in caplog.text
        assert len(seen) == 1 and seen[0].url.path == '/engine-proxy'
    finally:
        transport.disconnect()


@pytest.mark.parametrize('failure', ['redirect', 'lost-reply'])
def test_session_prompt_relay_never_retries_or_exposes_transport_secrets(tmp_path, failure, caplog):
    from http.server import ThreadingHTTPServer
    from runpod_lifecycle.pod_agent import Handler
    seen = []
    transport = HttpsPodTransport('pod1', TOKEN, route_grace=0)
    transport.http.close()
    def respond(request):
        seen.append(request)
        assert request.headers['Authorization'] == 'Bearer ' + TOKEN
        assert 'Cookie' not in request.headers and 'X-Secret' not in request.headers
        if failure == 'lost-reply':
            raise httpx.ReadError(TOKEN, request=request)
        return httpx.Response(307, headers={'Location': 'https://foreign/' + TOKEN}, stream=httpx.ByteStream(b''))
    transport.http = httpx.Client(base_url='https://pod1-8000.proxy.runpod.net',
        headers={'Authorization': 'Bearer ' + TOKEN}, transport=httpx.MockTransport(respond))
    class Relay(Handler):
        def dispatch(self):
            return transport.relay_engine(self, 'comfy')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Relay)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        response = httpx.post(f'http://127.0.0.1:{server.server_port}/prompt', content=b'{"prompt":{}}',
            headers={'Authorization': 'Bearer untrusted', 'Cookie': 'private', 'X-Secret': TOKEN}, trust_env=False)
        assert response.status_code == 502 and 'Location' not in response.headers
        assert TOKEN not in response.text and TOKEN not in caplog.text
        assert len(seen) == 1 and seen[0].url.path == '/engines/comfy/prompt'
        assert seen[0].content == b'{"prompt":{}}'
    finally:
        transport.disconnect()
        server.shutdown()
        server.server_close()
        thread.join(3)
