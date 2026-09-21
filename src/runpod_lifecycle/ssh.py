"""SSH transport helpers for worker pod commands."""

from __future__ import annotations

import logging
import os
import socket
import time
from urllib.parse import urlsplit

try:
    import paramiko
except ImportError:  # pragma: no cover - exercised indirectly before deps install.
    paramiko = None  # type: ignore[assignment]

logger = logging.getLogger("runpod_lifecycle.ssh")


def open_broker_socket(
    proxy_url: str,
    target_host: str,
    target_port: int,
    *,
    timeout: int = 10,
) -> socket.socket:
    """Open a raw TCP stream through Astrid's host-owned CONNECT broker."""

    parsed = urlsplit(proxy_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.port is None:
        raise RuntimeError("broker proxy must be an explicit http(s)://host:port URL")
    target = str(target_host).strip("[]")
    authority = f"[{target}]:{int(target_port)}" if ":" in target else f"{target}:{int(target_port)}"
    connection = socket.create_connection((parsed.hostname, parsed.port), timeout=timeout)
    try:
        connection.sendall(
            f"CONNECT {authority} HTTP/1.1\r\n"
            f"Host: {authority}\r\n"
            "Proxy-Connection: Keep-Alive\r\n\r\n".encode("ascii")
        )
        response = b""
        while b"\r\n\r\n" not in response and len(response) < 8192:
            chunk = connection.recv(4096)
            if not chunk:
                break
            response += chunk
        first_line = response.split(b"\r\n", 1)[0].decode("ascii", "replace")
        if not first_line.startswith("HTTP/") or " 200 " not in first_line:
            raise RuntimeError(f"broker CONNECT failed: {first_line or 'empty response'}")
        connection.settimeout(None)
        return connection
    except Exception:
        connection.close()
        raise


class SSHClient:
    """Minimal paramiko wrapper for executing commands over SSH."""

    def __init__(
        self,
        hostname: str,
        port: int,
        username: str,
        password: str | None = None,
        private_key_path: str | None = None,
        private_key_content: str | None = None,
        timeout: int = 10,
        proxy_url: str | None = None,
    ):
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.private_key_path = private_key_path
        self.private_key_content = private_key_content
        self.timeout = timeout
        self.proxy_url = proxy_url if proxy_url is not None else os.environ.get("ASTRID_BROKER_PROXY")
        self.client: paramiko.SSHClient | None = None

    def connect(self) -> None:
        if paramiko is None:
            raise RuntimeError("paramiko package is required for SSH operations")

        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        connect_kwargs = {
            "hostname": self.hostname,
            "port": self.port,
            "username": self.username,
            "timeout": self.timeout,
            "allow_agent": False,
            "look_for_keys": False,
        }
        pkey = None

        if self.private_key_content:
            try:
                from io import StringIO

                try:
                    pkey = paramiko.Ed25519Key.from_private_key(StringIO(self.private_key_content))
                except Exception:
                    try:
                        pkey = paramiko.RSAKey.from_private_key(StringIO(self.private_key_content))
                    except Exception:
                        pkey = paramiko.ECDSAKey.from_private_key(StringIO(self.private_key_content))
            except Exception as exc:
                logger.error("Failed to load private key from environment variable: %s", exc)
                raise RuntimeError(
                    f"Failed to load private key from environment variable: {exc}"
                ) from exc
        elif self.private_key_path and os.path.exists(os.path.expanduser(self.private_key_path)):
            expanded_key = os.path.expanduser(self.private_key_path)
            try:
                try:
                    pkey = paramiko.Ed25519Key.from_private_key_file(expanded_key)
                except Exception:
                    try:
                        pkey = paramiko.RSAKey.from_private_key_file(expanded_key)
                    except Exception:
                        pkey = paramiko.ECDSAKey.from_private_key_file(expanded_key)
            except Exception as exc:
                logger.error("Failed to load private key %s: %s", expanded_key, exc)
                raise RuntimeError(f"Failed to load private key {expanded_key}: {exc}") from exc
        else:
            connect_kwargs["password"] = self.password

        if pkey is not None:
            connect_kwargs["pkey"] = pkey

        broker_socket = None
        if self.proxy_url:
            broker_socket = open_broker_socket(
                self.proxy_url,
                self.hostname,
                self.port,
                timeout=self.timeout,
            )
            connect_kwargs["sock"] = broker_socket
        try:
            self.client.connect(**connect_kwargs)
            transport = self.client.get_transport()
            if transport is not None:
                # The host broker has a bounded idle select window.  Paramiko
                # keepalives keep long GPU jobs from losing an otherwise quiet
                # SSH tunnel without broadening the route grant.
                transport.set_keepalive(
                    max(1, int(os.environ.get("RUNPOD_SSH_KEEPALIVE_SECONDS", "5")))
                )
        except Exception:
            if broker_socket is not None:
                broker_socket.close()
            self.client.close()
            self.client = None
            raise

    def execute_command(self, command: str, timeout: int = 600) -> tuple[int, str, str]:
        if not self.client:
            raise RuntimeError("SSH client not connected. Call connect() first.")

        _, stdout, stderr = self.client.exec_command(command, timeout=timeout)
        channel = stdout.channel

        out_chunks: list[bytes] = []
        err_chunks: list[bytes] = []
        start_time = time.time()
        while not channel.exit_status_ready():
            while channel.recv_ready():
                out_chunks.append(channel.recv(65536))
            while channel.recv_stderr_ready():
                err_chunks.append(channel.recv_stderr(65536))
            elapsed = time.time() - start_time
            if elapsed > timeout:
                channel.close()
                out = b"".join(out_chunks).decode(errors="replace")
                err = b"".join(err_chunks).decode(errors="replace")
                if err:
                    err = f"{err}\nCommand timed out after {timeout} seconds"
                else:
                    err = f"Command timed out after {timeout} seconds"
                return -1, out, err
            time.sleep(0.1)

        exit_status = channel.recv_exit_status()
        while channel.recv_ready():
            out_chunks.append(channel.recv(65536))
        while channel.recv_stderr_ready():
            err_chunks.append(channel.recv_stderr(65536))
        out = b"".join(out_chunks).decode(errors="replace")
        err = b"".join(err_chunks).decode(errors="replace")
        return exit_status, out, err

    def disconnect(self) -> None:
        if self.client:
            self.client.close()
            self.client = None


__all__ = ["SSHClient", "open_broker_socket"]
