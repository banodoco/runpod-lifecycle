"""Local loopback tests for the optional Astrid broker transport."""

from __future__ import annotations

from contextlib import contextmanager
import select
import socket
import threading

import pytest

from runpod_lifecycle.ssh import open_broker_socket


HEADER = b"HTTP/1.1 200 Connection Established\r\n\r\n"


@contextmanager
def broker(monkeypatch, parts: tuple[bytes, ...], *, exchange: bool = False, segmented: bool = True):
    """Force receive boundaries using real TCP bytes, never scheduling sleeps.

    Each server write waits for the preceding part to be consumed. The client
    socket collects exactly min(recv size, remaining part size) actual bytes,
    so the old recv(4096) deterministically includes any coalesced ready bytes.
    A parser using smaller reads still exercises the same socket and writes.
    """
    consumed = [threading.Event() for _ in parts]
    observed: list[bytes] = []
    reads: list[bytes] = []
    errors: list[Exception] = []
    clients: list[socket.socket] = []

    class SegmentedSocket(socket.socket):
        part = 0
        offset = 0

        def recv(self, size, flags=0):
            if self.part == len(parts):
                return super().recv(size, flags)
            assert flags == 0
            size = min(size, len(parts[self.part]) - self.offset)
            data = b""
            while len(data) < size:
                chunk = super().recv(size - len(data))
                if not chunk:
                    break
                data += chunk
            reads.append(data)
            self.offset += len(data)
            if self.offset == len(parts[self.part]):
                consumed[self.part].set()
                self.part += 1
                self.offset = 0
            return data

    with socket.socket() as proxy:
        proxy.bind(("127.0.0.1", 0))
        proxy.listen()
        proxy.settimeout(3)
        address = proxy.getsockname()

        def connect(actual_address, timeout):
            assert actual_address == address  # no target connection or DNS
            assert segmented or len(parts) == 1
            client = SegmentedSocket() if segmented else socket.socket()
            clients.append(client)
            client.settimeout(timeout)
            client.connect(address)
            return client

        monkeypatch.setattr("runpod_lifecycle.ssh.socket.create_connection", connect)

        def serve():
            try:
                with proxy.accept()[0] as client:
                    client.settimeout(3)
                    request = b""
                    while not request.endswith(b"\r\n\r\n"):
                        chunk = client.recv(4096)
                        if not chunk:
                            return
                        request += chunk
                    observed.append(request)
                    for index, part in enumerate(parts):
                        client.sendall(part)
                        if index < len(parts) - 1:
                            assert consumed[index].wait(3), "client did not consume response part"
                    if exchange:
                        payload = client.recv(64)
                        if payload:
                            client.sendall(b"echo:" + payload)
            except Exception as exc:
                errors.append(exc)

        thread = threading.Thread(target=serve, name="test-connect-broker")
        thread.start()
        try:
            yield f"http://127.0.0.1:{address[1]}", clients, observed, reads
        finally:
            for client in clients:
                client.close()
            for event in consumed:
                event.set()
            thread.join(timeout=4)
            assert not thread.is_alive(), "fake broker thread leaked"
            assert not errors, errors


def assert_stream(monkeypatch, parts, target="198.51.100.7", authority="198.51.100.7:53603"):
    with broker(monkeypatch, parts, exchange=True) as (url, clients, observed, reads):
        connection = open_broker_socket(url, target, 53603)
        assert connection is clients[0]
        assert isinstance(connection, socket.socket)
        assert connection.fileno() >= 0
        assert connection.gettimeout() is None
        connection.settimeout(1)  # bound a regression locally, before the outer watchdog
        # Small reads must preserve order, with no loss or duplicate prefix.
        try:
            prefix = connection.recv(2)
        except TimeoutError:
            received = b"".join(reads)
            pytest.fail(f"application stream lost; real recv consumed {len(received)} bytes, tail={received[-64:]!r}")
        assert prefix == b"re", (parts, reads)
        assert connection.recv(3) == b"ady", (parts, reads)
        connection.sendall(b"hello")
        assert connection.recv(32) == b"echo:hello"
        assert connection.recv(1) == b""
        assert observed == [
            f"CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n"
            "Proxy-Connection: Keep-Alive\r\n\r\n".encode("ascii")
        ]


def test_open_broker_socket_connects_raw_tcp_stream(monkeypatch) -> None:
    # The exact historical nodeid now forces the formerly timing-dependent case.
    assert_stream(monkeypatch, (HEADER + b"ready",))


@pytest.mark.parametrize("split", [1, 9, len(HEADER) - 3, len(HEADER) - 2, len(HEADER) - 1, len(HEADER)])
def test_open_broker_socket_split_headers(monkeypatch, split) -> None:
    assert_stream(monkeypatch, (HEADER[:split], HEADER[split:] + b"ready"))


def test_open_broker_socket_bytewise_header(monkeypatch) -> None:
    assert_stream(monkeypatch, tuple(bytes([byte]) for byte in HEADER) + (b"ready",))


def test_open_broker_socket_ipv6_authority(monkeypatch) -> None:
    assert_stream(monkeypatch, (HEADER + b"ready",), "[2001:db8::7]", "[2001:db8::7]:53603")


def test_open_broker_socket_native_read_apis(monkeypatch) -> None:
    # Also exercise an ordinary, uninstrumented socket: readiness, recv_into,
    # and makefile must all see bytes left after the CONNECT delimiter.
    with broker(monkeypatch, (HEADER + b"ready",), exchange=True, segmented=False) as state:
        url, clients, _, _ = state
        connection = open_broker_socket(url, "198.51.100.7", 53603)
        assert connection is clients[0]
        assert type(connection) is socket.socket
        assert connection.gettimeout() is None
        assert select.select([connection], [], [], 1)[0] == [connection]
        connection.settimeout(1)
        first = bytearray(1)
        assert connection.recv_into(first) == 1
        assert first == b"r"
        with connection.makefile("rb") as stream:
            assert stream.read(4) == b"eady"
            connection.sendall(b"hello")
            assert stream.read() == b"echo:hello"


@pytest.mark.parametrize("status", [b"HTTP/1.0 200 OK", b"HTTP/1.1 200", b"HTTP/1.1 200 OK"])
def test_open_broker_socket_success_status(monkeypatch, status) -> None:
    assert_stream(monkeypatch, (status + b"\r\n\r\nready",))


@pytest.mark.parametrize("response", [
    b"",
    b"HTTP/1.1 200 Connection Established\r\n",
    b"HTTP/1.1 200 Connection Established\r\nX: " + b"a" * 8192,
    b"HTTP/1.1 403 Forbidden\r\n\r\n",
    b"HTTP/1.1 403 reason 200 misleading\r\n\r\n",
    b"HTTP/garbage 200 OK\r\n\r\n",
], ids=["empty", "truncated", "unterminated-oversized", "rejected", "misleading-reason", "bad-version"])
def test_open_broker_socket_rejects_invalid_headers(monkeypatch, response) -> None:
    with broker(monkeypatch, (response,)) as (url, clients, _, _):
        with pytest.raises(RuntimeError, match="broker CONNECT failed"):
            open_broker_socket(url, "198.51.100.7", 53603)
        assert clients[0].fileno() == -1


@pytest.mark.parametrize("header_size", [8192, 8193])
def test_open_broker_socket_header_size_limit(monkeypatch, header_size) -> None:
    prefix = b"HTTP/1.1 200 OK\r\nX: "
    header = prefix + b"a" * (header_size - len(prefix) - 4) + b"\r\n\r\n"
    assert len(header) == header_size
    if header_size == 8192:
        assert_stream(monkeypatch, (header + b"ready",))
    else:
        with broker(monkeypatch, (header,)) as (url, clients, _, _):
            with pytest.raises(RuntimeError, match="broker CONNECT failed"):
                open_broker_socket(url, "198.51.100.7", 53603)
            assert clients[0].fileno() == -1
