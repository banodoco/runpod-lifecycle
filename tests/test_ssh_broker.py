"""Tests for the optional Astrid broker transport."""

from __future__ import annotations

import socket
import threading

from runpod_lifecycle.ssh import open_broker_socket


def test_open_broker_socket_connects_raw_tcp_stream() -> None:
    proxy = socket.socket()
    proxy.bind(("127.0.0.1", 0))
    proxy.listen()
    proxy_port = proxy.getsockname()[1]
    upstream_port = 53603
    observed: list[bytes] = []

    def serve() -> None:
        client, _ = proxy.accept()
        try:
            request = b""
            while b"\r\n\r\n" not in request:
                request += client.recv(4096)
            observed.append(request)
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            client.sendall(b"ready")
            payload = client.recv(64)
            client.sendall(b"echo:" + payload)
        finally:
            client.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        connection = open_broker_socket(
            f"http://127.0.0.1:{proxy_port}", "198.51.100.7", upstream_port
        )
        assert connection.recv(16) == b"ready"
        connection.sendall(b"hello")
        assert connection.recv(32) == b"echo:hello"
        connection.close()
        thread.join(timeout=2)
        assert observed and f"CONNECT 198.51.100.7:{upstream_port} HTTP/1.1".encode() in observed[0]
    finally:
        proxy.close()
