"""High-level ship-and-run orchestration on top of lifecycle primitives."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from .config import RunPodConfig
from .guard import PodGuard
from .lifecycle import launch as _launch_pod
from .pod import Pod
from .shipping import (
    UploadHeartbeat,
    _upload_remote_script,
    _upload_tarball,
    download_artifact_archive,
    upload_dir,
)

logger = logging.getLogger("runpod_lifecycle.runner")

# -- Defaults for ship_and_run_detached ------------------------------------

DEFAULT_POLL_EXIT_MARKER = "/tmp/runpod-lifecycle-exit-code"
DEFAULT_POLL_COMMAND_TEMPLATE = "cat {poll_exit_marker} 2>/dev/null || echo ''"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_current_task_cancellation() -> None:
    """Uncancel the current asyncio Task so cleanup can proceed."""
    task = asyncio.current_task()
    if task is None or not hasattr(task, "uncancel"):
        return
    while task.cancelling():
        task.uncancel()


def _parse_detached_exit(stdout: str) -> int | None:
    """Extract exit code from polled command output.

    Looks for a bare integer (or negative integer) on the first
    non-empty line.  Returns ``None`` when no exit code is found.
    """
    for line in stdout.strip().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.isdigit():
            return int(stripped)
        if stripped.startswith("-") and stripped[1:].isdigit():
            return int(stripped)
    return None


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ShipAndRunResult:
    """Result of a :func:`ship_and_run` or :func:`ship_and_run_detached`."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    pod: Pod | None = None
    artifact_root: Path | None = None
    cost_per_hr: float | None = None
    breach_log: list[dict] = field(default_factory=list)
    terminated: bool = False
    upload_info: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def ship_and_run(
    config: RunPodConfig,
    remote_script: str,
    *,
    local_root: Path,
    remote_root: str,
    exclude: set[str],
    upload_mode: Literal["sftp_walk", "tarball"] = "sftp_walk",
    timeout: int = 600,
    name_prefix: str = "pod",
    terminate_after_exec: bool = True,
    guard_factory: Callable[..., PodGuard] | None = None,
) -> ShipAndRunResult:
    """Launch a pod, ship a payload, run *remote_script*, and tear down.

    Parameters
    ----------
    terminate_after_exec:
        When ``True`` (the default) the pod is terminated in ``finally``.
        When ``False`` the pod is left alive and returned in the result.
    guard_factory:
        Optional callable that returns a :class:`PodGuard`.  Defaults to
        :class:`PodGuard` *at the call site* so that monkeypatching in
        tests works correctly.
    """
    # Resolve *guard_factory* at call site — NOT as a default-argument
    # value (which Python evaluates at import time, before any test
    # monkeypatch).
    _factory: Callable[..., PodGuard] = (
        guard_factory if guard_factory is not None else PodGuard
    )

    guard = _factory(
        name_prefix=name_prefix,
        default_max_runtime_seconds=max(timeout * 2, 7200),
        auto_terminate=terminate_after_exec,
    )

    result = ShipAndRunResult(returncode=-1)
    pod: Pod | None = None

    try:
        # ---- launch -------------------------------------------------------
        logger.info("Launching pod name_prefix=%s", name_prefix)
        # Backward-compat: if the guard exposes an old-style ``launch()``
        # method (e.g. FakeGuard in tests), use it instead of the direct
        # lifecycle call so monkeypatch injection still works.
        if hasattr(guard, "launch"):
            pod = await guard.launch()
            # guard.launch() may have already attached internally
            if guard.pod is None and pod is not None:
                guard.attach(pod)
        else:
            pod = await _launch_pod(
                config, name=f"{name_prefix}-{int(time.time())}"
            )
            guard.attach(pod)
        result.pod = pod

        # ---- wait for SSH -------------------------------------------------
        await pod.wait_ready(timeout=300)
        ssh_details = await pod._ensure_ssh_details()
        logger.info(
            "Pod SSH ready: root@%s:%s",
            ssh_details["ip"],
            ssh_details["port"],
        )

        # ---- GPU check ----------------------------------------------------
        code, stdout, stderr = await pod.exec_ssh("nvidia-smi -L", timeout=60)
        if code != 0:
            result.returncode = code
            result.stdout = stdout
            result.stderr = stderr
            return result

        # ---- upload -------------------------------------------------------
        if upload_mode == "tarball":
            result.upload_info = await _upload_tarball(
                pod, exclude, local_root=local_root, remote_root=remote_root
            )
        else:
            client = pod.open_ssh_client()
            try:
                sftp = client.open_sftp()
                try:
                    progress = UploadHeartbeat(label="sftp_upload")
                    upload_dir(
                        sftp,
                        local_root,
                        remote_root,
                        exclude,
                        progress=progress,
                        local_root=local_root,
                    )
                    progress.tick(force=True)
                finally:
                    sftp.close()
            finally:
                client.close()
            result.upload_info = {
                "mode": "sftp_walk",
                "local_root": str(local_root),
                "remote_root": remote_root,
            }

        # ---- execute ------------------------------------------------------
        code, stdout, stderr = await pod.exec_ssh(
            remote_script, timeout=timeout
        )
        result.returncode = code
        result.stdout = stdout
        result.stderr = stderr

        return result

    except asyncio.CancelledError:
        _clear_current_task_cancellation()
        result.returncode = 130
        logger.info("ship_and_run cancelled, returning 130")
        return result

    finally:
        if terminate_after_exec:
            await guard.terminate()
            result.terminated = True
        else:
            result.pod = pod
        result.breach_log = list(guard.breach_log)


async def ship_and_run_detached(
    config: RunPodConfig | None = None,
    remote_script: str = "",
    *,
    pod: Pod | None = None,
    local_root: Path | None = None,
    remote_root: str = "/workspace",
    exclude: set[str] | None = None,
    upload_mode: Literal["sftp_walk", "tarball"] = "sftp_walk",
    timeout: int = 3600,
    name_prefix: str = "pod",
    terminate_after_exec: bool = False,
    poll_interval: int = 30,
) -> ShipAndRunResult:
    """Run *remote_script* detached on a new pod or an existing pod.

    Either pass *config* to provision a new pod or pass *pod* to reattach
    to an already-provisioned pod. When both are supplied, *pod* wins and
    *config* is used only as an API-key source.
    """
    if config is None and pod is None:
        raise ValueError("ship_and_run_detached requires either config or pod")

    if pod is not None and config is not None:
        logger.info(
            "ship_and_run_detached received both config and pod; "
            "using pod and treating config as an api_key source only"
        )
        pod_config = getattr(pod, "config", None)
        if pod_config is not None and hasattr(pod_config, "api_key"):
            pod_config.api_key = config.api_key

    result = ShipAndRunResult(returncode=-1)
    active_pod = pod
    guard: PodGuard | None = None
    exclude_set = exclude or set()
    poll_exit_marker = DEFAULT_POLL_EXIT_MARKER
    poll_command_template = DEFAULT_POLL_COMMAND_TEMPLATE
    artifact_paths = ["out", "output"]
    remote_script_path = "/tmp/runpod-lifecycle-remote-run.sh"

    try:
        if active_pod is None:
            assert config is not None
            guard = PodGuard(
                name_prefix=name_prefix,
                default_max_runtime_seconds=max(timeout * 2, 7200),
                auto_terminate=terminate_after_exec,
            )
            active_pod = await _launch_pod(
                config, name=f"{name_prefix}-{int(time.time())}"
            )
            guard.attach(active_pod)
            await active_pod.wait_ready(timeout=300)

        result.pod = active_pod
        hourly_rate = getattr(active_pod, "hourly_rate", None)
        if isinstance(hourly_rate, int | float):
            result.cost_per_hr = float(hourly_rate)

        # ---- SSH ----------------------------------------------------------
        ssh_details = await active_pod._ensure_ssh_details()
        logger.info(
            "Pod SSH ready: root@%s:%s",
            ssh_details["ip"],
            ssh_details["port"],
        )

        # ---- GPU check ----------------------------------------------------
        code, stdout, stderr = await active_pod.exec_ssh("nvidia-smi -L", timeout=60)
        if code != 0:
            result.returncode = code
            result.stdout = stdout
            result.stderr = stderr
            return result

        # ---- upload -------------------------------------------------------
        if local_root is not None:
            if upload_mode == "tarball":
                result.upload_info = await _upload_tarball(
                    active_pod,
                    exclude_set,
                    local_root=local_root,
                    remote_root=remote_root,
                )
            else:
                client = active_pod.open_ssh_client()
                try:
                    sftp = client.open_sftp()
                    try:
                        progress = UploadHeartbeat(label="sftp_upload")
                        upload_dir(
                            sftp,
                            local_root,
                            remote_root,
                            exclude_set,
                            progress=progress,
                            local_root=local_root,
                        )
                        progress.tick(force=True)
                    finally:
                        sftp.close()
                finally:
                    client.close()
                result.upload_info = {
                    "mode": "sftp_walk",
                    "local_root": str(local_root),
                    "remote_root": remote_root,
                }
        else:
            result.upload_info = {"mode": "none", "remote_root": remote_root}

        # ---- upload remote script -----------------------------------------
        await _upload_remote_script(active_pod, remote_script)

        # ---- launch detached command --------------------------------------
        launch_command = (
            f"cd {remote_root} && "
            f"rm -f {poll_exit_marker} && "
            f"nohup bash {remote_script_path} "
            f"> /tmp/runpod-lifecycle-remote-live.log 2>&1; "
            f'rc=$?; printf "%s" "$rc" > {poll_exit_marker}; exit "$rc"'
        )

        code, stdout, stderr = await active_pod.exec_ssh(
            f"nohup bash -lc {launch_command!r} "
            f">/tmp/runpod-lifecycle-launch.log 2>&1 & echo $!",
            timeout=30,
        )
        if stdout.strip():
            logger.info("remote_pid=%s", stdout.strip())
        if code != 0:
            result.returncode = code
            result.stderr = stderr
            return result

        # ---- poll loop ----------------------------------------------------
        start = time.monotonic()
        while True:
            if time.monotonic() - start > timeout:
                logger.warning("detached_timeout=%s", timeout)
                result.returncode = 124
                return result

            poll_cmd = poll_command_template.format(
                poll_exit_marker=poll_exit_marker
            )
            try:
                code, stdout, stderr = await active_pod.exec_ssh(
                    poll_cmd, timeout=60
                )
            except Exception as exc:
                logger.warning("poll_ssh_failed=%s", exc)
                await asyncio.sleep(poll_interval)
                continue

            exit_code = _parse_detached_exit(stdout)
            if exit_code is not None:
                result.returncode = exit_code
                if local_root is not None:
                    try:
                        artifact_root = await download_artifact_archive(
                            active_pod,
                            remote_root=remote_root,
                            artifact_paths=artifact_paths,
                            local_artifact_root=local_root / "artifacts",
                            exit_code=exit_code,
                            remote_command=launch_command,
                            upload=result.upload_info,
                        )
                        result.artifact_root = artifact_root
                    except Exception as exc:
                        logger.warning("artifact_download_failed=%s", exc)
                return result

            await asyncio.sleep(poll_interval)

    except asyncio.CancelledError:
        _clear_current_task_cancellation()
        result.returncode = 130
        return result

    finally:
        if terminate_after_exec and active_pod is not None:
            if guard is not None:
                await guard.terminate()
            else:
                await active_pod.terminate()
            result.terminated = True
        else:
            result.pod = active_pod
        if guard is not None:
            result.breach_log = list(guard.breach_log)
