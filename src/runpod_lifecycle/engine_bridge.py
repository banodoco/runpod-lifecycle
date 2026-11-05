"""Trusted single-user loopback HTTP and WebSocket bridge.

The listener is host-side and ephemeral. It binds only to 127.0.0.1, checks
Host and Origin to block browser cross-origin use and DNS rebinding, then sends
requests through the caller's existing authenticated transport. It does not
authenticate local callers: callers sharing this user's loopback environment
are trusted. Keep credentials in transport custody; never put them in URLs.
"""
import http.client
import re
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .pod_agent import Handler

PROBE_TIMEOUT = 12
PROBE_MAX_BYTES = 1024 * 1024


def engine_proxy_config(value):
    """Validate the fixed endpoint/port slot before opening the listener."""
    if not isinstance(value, dict) or set(value) != {"endpoint_id", "port"}:
        raise ValueError("engine proxy requires only endpoint_id and port")
    endpoint, port = value["endpoint_id"], value["port"]
    if not isinstance(endpoint, str) or not re.fullmatch(r"[a-z][a-z0-9-]*", endpoint):
        raise ValueError("invalid engine proxy endpoint_id")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("engine proxy port must be an integer from 1 to 65535")
    return dict(value)


class _BridgeServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, transport, endpoint):
        self.transport, self.endpoint = transport, endpoint
        self.connections = set()
        self.connection_lock = threading.Lock()
        self.stopping = False
        super().__init__(("127.0.0.1", 0), _BridgeHandler)

    def handle_error(self, request, client_address):
        # Even exception text from an HTTP dependency may contain a bearer.
        pass


class _BridgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    reply = Handler.reply

    def log_message(self, *_args):
        pass

    def setup(self):
        super().setup()
        self.connection.settimeout(15)
        with self.server.connection_lock:
            if self.server.stopping:
                raise ConnectionAbortedError()
            self.server.connections.add(self.connection)

    def finish(self):
        try:
            super().finish()
        finally:
            with self.server.connection_lock:
                self.server.connections.discard(self.connection)

    def dispatch(self):
        expected = f"127.0.0.1:{self.server.server_port}"
        # Block browser cross-origin requests and DNS rebinding to this listener.
        if (self.headers.get_all("Host", []) != [expected]
                or self.headers.get("Origin", "http://" + expected) != "http://" + expected):
            return self.reply(403, {"error": "local engine origin denied"})
        try:
            return self.server.transport.relay_engine(self, self.server.endpoint)
        except Exception:
            return self.reply(502, {"error": "engine bridge transport failed"})

    do_GET = do_POST = dispatch


class EngineBridge:
    """Own a loopback listener backed by an authenticated transport.

    The transport is dedicated to this bridge and is disconnected by close().
    Local callers are trusted under the single-user loopback assumption above.
    """

    def __init__(self, transport, config):
        config = engine_proxy_config(config)
        self.transport = transport
        self.server = _BridgeServer(transport, config["endpoint_id"])
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()

    def probe_get(self, path):
        """Bounded GET over this session's authenticated engine relay."""
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port,
                                                 timeout=PROBE_TIMEOUT)
        try:
            connection.request("GET", path, headers={"Accept": "application/json",
                                                     "Connection": "close"})
            response = connection.getresponse()
            if 300 <= response.status < 400:
                raise ValueError("engine redirect refused")
            declared = response.getheader("Content-Length")
            if declared is not None:
                try:
                    size = int(declared)
                except ValueError:
                    raise ValueError("invalid engine response length") from None
                if not 0 <= size <= PROBE_MAX_BYTES:
                    raise ValueError("engine response exceeds probe limit")
            body = response.read(PROBE_MAX_BYTES + 1)
            if len(body) > PROBE_MAX_BYTES:
                raise ValueError("engine response exceeds probe limit")
            if declared is not None and len(body) != size:
                raise ValueError("short engine response")
            return response.status, body
        except (OSError, http.client.HTTPException, TimeoutError):
            raise ValueError("engine probe timed out or transport failed") from None
        finally:
            connection.close()

    def close(self):
        with self.server.connection_lock:
            self.server.stopping = True
            connections = list(self.server.connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        self.transport.disconnect()
