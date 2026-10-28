"""Async facade for interacting with a single RunPod pod lifecycle."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import time
from pathlib import Path
from typing import Any

from . import api
from .config import RunPodConfig
from .errors import CleanupPendingError, LaunchFailure, NotReadyTimeout, SSHError, TerminateError
from .events import EventHooks, PodState, _emit_error, _emit_state
from .ssh import SSHClient
from .storage import STORAGE_CHECK_COMMAND, evaluate_storage_health, parse_df_output

logger = logging.getLogger("runpod_lifecycle.pod")


class Pod:
    """Async operations for a specific RunPod pod."""

    def __init__(
        self,
        pod_id: str,
        name: str,
        config: RunPodConfig,
        hooks: EventHooks | None = None,
        ram_tier: int = 0,
        storage_volume: str | None = None,
    ) -> None:
        self.id = pod_id
        self.name = name
        self.config = config
        self.hooks = hooks or EventHooks()
        self._ram_tier = ram_tier
        self._storage_volume = storage_volume
        self._ssh_details: dict[str, Any] | None = None
        # Per-claim private agent state (engine startup only); see terminate().
        self.claim_id: str | None = None
        self._engine_state_dir: str | None = None
        self._last_exec_at: float | None = None
        self._created_at = time.monotonic()

    async def wait_ready(self, timeout: int = 600) -> dict[str, Any]:
        start_time = time.monotonic()
        poll_interval = 5
        current_state: PodState | None = None

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > timeout:
                raise NotReadyTimeout(f"Pod {self.id} did not become ready within {timeout} seconds")

            status = await self.status()
            desired_status = status.get("desired_status") if status else None
            ports = (status.get("ports") or []) if status else []

            if current_state is None:
                current_state = PodState.PROVISIONING
                await _emit_state(self.hooks, self.id, PodState.PROVISIONING, {"status": status or {}})

            if desired_status in {"FAILED", "TERMINATED"}:
                failed_state = PodState.TERMINATED if desired_status == "TERMINATED" else PodState.FAILED
                await _emit_state(self.hooks, self.id, failed_state, {"status": status or {}})
                raise LaunchFailure(f"Pod {self.id} entered terminal state {desired_status}")

            if desired_status == "RUNNING" and self.config.transport == "https":
                client = self.https_transport()
                try:
                    # Settled readiness (consecutive health replies), bounded per
                    # iteration so provider FAILED/TERMINATED states are still seen.
                    await asyncio.to_thread(client.wait_ready, min(15, max(1, timeout - elapsed)))
                except Exception as exc:
                    from .https import PodTransportError
                    if not isinstance(exc, PodTransportError) or "HTTP 401" in str(exc) or "HTTP 403" in str(exc):
                        raise
                else:
                    await _emit_state(self.hooks, self.id, PodState.READY, {"status": status})
                    return status
                finally:
                    client.disconnect()

            has_ssh_port = any(port.get("privatePort") == 22 for port in ports)
            if self.config.transport == "ssh" and desired_status == "RUNNING" and has_ssh_port:
                if current_state is not PodState.READY:
                    self._ssh_details = await self._ensure_ssh_details()
                    current_state = PodState.READY
                    await _emit_state(self.hooks, self.id, PodState.READY, {"status": status})
                return status

            if current_state is not PodState.STARTING:
                current_state = PodState.STARTING
                await _emit_state(self.hooks, self.id, PodState.STARTING, {"status": status or {}})

            await asyncio.sleep(poll_interval)

    async def status(self) -> dict[str, Any] | None:
        return await asyncio.to_thread(api.get_pod_status, self.id, self.config.api_key)

    def https_transport(self):
        from .https import HttpsPodTransport
        if self.config.transport != "https":
            raise ValueError("pod does not use HTTPS")
        return HttpsPodTransport(self.id, self.config.https_token or "")

    async def exec_ssh(self, cmd: str, timeout: int = 600) -> tuple[int, str, str]:
        """Execute through the configured transport (legacy method name)."""
        if self.config.transport == "https":
            client = self.https_transport()
            try:
                result = await asyncio.to_thread(client.execute_command, cmd, timeout)
                self._last_exec_at = time.monotonic()
                return result
            finally:
                client.disconnect()
        ssh_details = await self._ensure_ssh_details()
        ssh_client = self._build_ssh_client(ssh_details)

        def _run_command() -> tuple[int, str, str]:
            ssh_client.connect()
            try:
                return ssh_client.execute_command(cmd, timeout)
            finally:
                ssh_client.disconnect()

        try:
            result = await asyncio.to_thread(_run_command)
        except Exception as exc:
            raise SSHError(f"SSH command failed for pod {self.id}: {exc}") from exc

        self._last_exec_at = time.monotonic()
        return result

    async def is_idle(self, threshold_seconds: int) -> bool:
        last_activity = self._last_exec_at or self._created_at
        if time.monotonic() - last_activity < threshold_seconds:
            return False

        try:
            exit_code, stdout, _stderr = await self.exec_ssh(
                "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"
            )
            if exit_code != 0:
                return False

            first_line = stdout.strip().splitlines()[0]
            return int(first_line) < 5
        except (IndexError, ValueError, SSHError) as exc:
            logger.warning("Could not determine GPU utilization for pod %s: %s", self.id, exc)
            return False

    async def _remove_engine_bearer(self) -> None:
        """Best effort: never leave the engine bearer on the persistent volume.

        The pod is still terminated if this fails; the file then sits in a
        claim-unique directory that no later claim will reuse.
        """
        if not self._engine_state_dir or self.config.transport != "https":
            return
        command = "rm -f -- " + shlex.quote(f"{self._engine_state_dir}/engine-bearer.token")
        client = self.https_transport()
        try:
            await asyncio.wait_for(asyncio.to_thread(client.execute_command, command, 20), 30)
        except Exception as exc:
            logger.warning("Could not remove the engine bearer of pod %s: %s", self.id, type(exc).__name__)
        finally:
            client.disconnect()

    async def terminate(self) -> None:
        await self._remove_engine_bearer()
        try:
            await asyncio.to_thread(api.terminate_pod, self.id, self.config.api_key)
        except CleanupPendingError as exc:
            await _emit_error(
                self.hooks,
                exc,
                {"pod_id": self.id, "status": exc.status, "cleanup_pending": True},
            )
            raise
        except Exception as exc:
            await _emit_error(self.hooks, exc, {"pod_id": self.id})
            raise TerminateError(f"Failed to terminate pod {self.id}: {exc}") from exc

        await _emit_state(self.hooks, self.id, PodState.TERMINATED, {"pod_id": self.id})

    async def check_storage_health(
        self,
        min_free_gb: int = 50,
        max_percent_used: int = 85,
    ) -> dict[str, Any]:
        _exit_code, raw_output, _stderr = await self.exec_ssh(STORAGE_CHECK_COMMAND, timeout=30)
        parsed = parse_df_output(raw_output)

        api_total_gb = None
        if self._storage_volume:
            volumes = await asyncio.to_thread(api.get_network_volumes, self.config.api_key)
            volume_info = next((volume for volume in volumes if volume.get("id") == self._storage_volume), None)
            api_total_gb = volume_info.get("size") if volume_info else None

        return evaluate_storage_health(parsed, api_total_gb, min_free_gb, max_percent_used)

    def open_ssh_client(self) -> Any:
        """Open and return the underlying connected paramiko SSH client."""
        if self.config.transport == "https":
            raise ValueError("raw SSH unavailable for HTTPS pod; use put_file/get_file")
        ssh_details = self._ensure_ssh_details_sync()
        ssh_client = self._build_ssh_client(ssh_details)
        ssh_client.connect()
        raw_client = getattr(ssh_client, "client", None)
        if raw_client is None:
            raise SSHError(f"SSH client for pod {self.id} did not expose a raw client")
        return raw_client

    # ------------------------------------------------------------------
    # Composable surface methods (Sprint 4)
    # ------------------------------------------------------------------

    async def upload_path(
        self,
        local: Path,
        remote: str,
        exclude: set[str] | None = None,
        mode: str = "sftp_walk",
    ) -> None:
        """Upload *local* directory tree to *remote* on the pod.

        Delegates to shipping primitives based on *mode*:
        ``"sftp_walk"`` (default) or ``"tarball"``.
        """
        from .shipping import _build_upload_tarball, _upload_remote_script, _upload_tarball, upload_dir

        exclude_set = exclude or set()
        if self.config.transport == "https":
            import shlex
            client = self.https_transport()
            try:
                local = Path(local)
                if local.is_file():
                    await asyncio.to_thread(client.put, local, remote)
                else:
                    code, _, _ = await asyncio.to_thread(client.execute_command, "mkdir -p " + shlex.quote(remote))
                    if code:
                        raise RuntimeError("remote directory creation failed")
                    for directory, dirs, files in os.walk(local):
                        relative = Path(directory).relative_to(local)
                        dirs[:] = [d for d in dirs if d not in exclude_set and not (Path(directory) / d).is_symlink()]
                        for name in files:
                            source = Path(directory) / name
                            if name not in exclude_set and not source.is_symlink():
                                await asyncio.to_thread(client.put, source, str(Path(remote) / relative / name))
                return
            finally:
                client.disconnect()
        if mode == "tarball":
            await _upload_tarball(
                self,
                exclude_set,
                local_root=local,
                remote_root=remote,
            )
        else:
            client = self.open_ssh_client()
            try:
                sftp = client.open_sftp()
                try:
                    upload_dir(sftp, local, remote, exclude_set, local_root=local)
                finally:
                    sftp.close()
            finally:
                client.close()

    async def put_file(self, local: Path, remote: str, *, reserved: bool = False) -> None:
        """Write a single file; reserved preserves a pre-created private inode."""
        if self.config.transport == "https":
            client = self.https_transport()
            try:
                await asyncio.to_thread(client.put, local, remote, reserved=reserved)
            finally:
                client.disconnect()
            return
        def upload():
            client = self.open_ssh_client()
            try:
                sftp = client.open_sftp()
                try:
                    if reserved:
                        with sftp.open(remote, "r+b") as dst, local.open("rb") as src:
                            for chunk in iter(lambda: src.read(65536), b""):
                                dst.write(chunk)
                            dst.flush()
                    else:
                        sftp.put(str(local), remote)
                finally:
                    sftp.close()
            finally:
                client.close()
        await asyncio.to_thread(upload)

    async def get_file(self, remote: str, local: Path) -> None:
        if self.config.transport == "https":
            client = self.https_transport()
            try:
                await asyncio.to_thread(client.get, remote, local)
            finally:
                client.disconnect()
            return
        def download():
            client = self.open_ssh_client()
            try:
                sftp = client.open_sftp()
                try:
                    sftp.get(remote, str(local))
                finally:
                    sftp.close()
            finally:
                client.close()
        await asyncio.to_thread(download)

    async def download_archive(
        self,
        remote_root: str,
        local: Path,
        *,
        artifact_paths: list[str] | None = None,
    ) -> Path | None:
        """Download artifact directories from the pod into *local*.

        Thin facade around :func:`shipping.download_artifact_archive`.
        """
        from .shipping import download_artifact_archive

        return await download_artifact_archive(
            self,
            remote_root=remote_root,
            artifact_paths=artifact_paths or ["out", "output"],
            local_artifact_root=local,
        )

    @staticmethod
    async def create_storage(
        name: str,
        size_gb: int,
        datacenter_id: str,
    ) -> dict[str, Any]:
        """Create a RunPod network volume.

        Thin facade calling :func:`api.create_network_volume`.
        Requires the ``RUNPOD_API_KEY`` env var to be set (read from
        config when called via a bound Pod, or passed explicitly via
        the static method).
        """
        import os as _os

        runpod_api_key = _os.environ["RUNPOD_API_KEY"]
        return await asyncio.to_thread(
            api.create_network_volume,
            runpod_api_key,
            name,
            size_gb,
            datacenter_id,
        )

    @staticmethod
    async def list_storages() -> list[dict[str, Any]]:
        """Return all RunPod network volumes for the account.

        Thin facade calling :func:`api.get_network_volumes`.
        """
        import os as _os

        runpod_api_key = _os.environ["RUNPOD_API_KEY"]
        return await asyncio.to_thread(api.get_network_volumes, runpod_api_key)

    @staticmethod
    async def get_storage(name_or_id: str) -> dict[str, Any] | None:
        """Look up a RunPod network volume by name or ID.

        Returns the volume dict on match, ``None`` if not found.
        """
        volumes = await Pod.list_storages()
        for vol in volumes:
            if vol.get("id") == name_or_id or vol.get("name") == name_or_id:
                return vol
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_ssh_details(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._ensure_ssh_details_sync)

    def _ensure_ssh_details_sync(self) -> dict[str, Any]:
        if self._ssh_details is None:
            self._ssh_details = api.get_pod_ssh_details(self.id, self.config.api_key)

        if not self._ssh_details:
            raise SSHError(f"Could not get SSH details for pod {self.id}")
        return self._ssh_details

    def _build_ssh_client(self, ssh_details: dict[str, Any]) -> SSHClient:
        if self.config.ssh_private_key:
            return SSHClient(
                hostname=ssh_details["ip"],
                port=ssh_details["port"],
                username="root",
                private_key_content=self.config.ssh_private_key,
            )

        if self.config.ssh_private_key_path:
            expanded_path = os.path.expanduser(self.config.ssh_private_key_path)
            if os.path.exists(expanded_path):
                return SSHClient(
                    hostname=ssh_details["ip"],
                    port=ssh_details["port"],
                    username="root",
                    private_key_path=self.config.ssh_private_key_path,
                )

        return SSHClient(
            hostname=ssh_details["ip"],
            port=ssh_details["port"],
            username="root",
            password=ssh_details.get("password", "runpod"),
        )
