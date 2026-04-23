"""Async facade for interacting with a single RunPod pod lifecycle."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from . import api
from .config import RunPodConfig
from .errors import LaunchFailure, NotReadyTimeout, SSHError, TerminateError
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
            ports = status.get("ports", []) if status else []

            if current_state is None:
                current_state = PodState.PROVISIONING
                await _emit_state(self.hooks, self.id, PodState.PROVISIONING, {"status": status or {}})

            if desired_status in {"FAILED", "TERMINATED"}:
                failed_state = PodState.TERMINATED if desired_status == "TERMINATED" else PodState.FAILED
                await _emit_state(self.hooks, self.id, failed_state, {"status": status or {}})
                raise LaunchFailure(f"Pod {self.id} entered terminal state {desired_status}")

            has_ssh_port = any(port.get("privatePort") == 22 for port in ports)
            if desired_status == "RUNNING" and has_ssh_port:
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

    async def exec_ssh(self, cmd: str, timeout: int = 600) -> tuple[int, str, str]:
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

    async def terminate(self) -> None:
        try:
            await asyncio.to_thread(api.terminate_pod, self.id, self.config.api_key)
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

    async def _ensure_ssh_details(self) -> dict[str, Any]:
        if self._ssh_details is None:
            self._ssh_details = await asyncio.to_thread(api.get_pod_ssh_details, self.id, self.config.api_key)

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
