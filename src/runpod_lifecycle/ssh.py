"""SSH transport helpers for worker pod commands."""

from __future__ import annotations

import logging
import os
import time

try:
    import paramiko
except ImportError:  # pragma: no cover - exercised indirectly before deps install.
    paramiko = None  # type: ignore[assignment]

logger = logging.getLogger("runpod_lifecycle.ssh")


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
    ):
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.private_key_path = private_key_path
        self.private_key_content = private_key_content
        self.timeout = timeout
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

        self.client.connect(**connect_kwargs)

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


__all__ = ["SSHClient"]
