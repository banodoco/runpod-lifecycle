"""Launch orchestration for RunPod pods with RAM-tier and storage fallback."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from .api import create_pod, find_gpu_type, get_network_volumes
from .config import RunPodConfig
from .errors import LaunchFailure
from .events import EventHooks, PodState, _emit_error, _emit_state
from .pod import Pod
from .storage import check_and_expand_storage, get_storage_volume_id

logger = logging.getLogger("runpod_lifecycle.lifecycle")


def _resolve_public_key_string(config: RunPodConfig) -> str | None:
    if config.ssh_public_key:
        return config.ssh_public_key

    if config.ssh_public_key_path:
        expanded_path = os.path.expanduser(config.ssh_public_key_path)
        try:
            return open(expanded_path, "r", encoding="utf-8").read().strip()
        except OSError as exc:
            logger.warning("Could not read SSH public key from %s: %s", expanded_path, exc)

    logger.warning("No SSH public key configured; pod access may require password auth")
    return None


def _build_ram_tiers(config: RunPodConfig) -> list[int]:
    if not config.ram_tiers_enabled:
        return [config.min_memory_gb]

    ram_tiers = [tier for tier in config.ram_tiers if tier >= config.min_memory_gb]
    return ram_tiers or [config.min_memory_gb]


async def _resolve_storage_targets(config: RunPodConfig) -> list[tuple[str | None, str | None]]:
    raw_targets: list[str] = []
    for storage_name in [config.storage_name, *config.storage_volumes]:
        if storage_name and storage_name not in raw_targets:
            raw_targets.append(storage_name)

    if not raw_targets:
        return [(None, None)]

    resolved_targets: list[tuple[str | None, str | None]] = []
    for storage_name in raw_targets:
        volume_id = await asyncio.to_thread(get_storage_volume_id, config.api_key, storage_name)
        if volume_id:
            resolved_targets.append((storage_name, volume_id))
        else:
            logger.warning("Storage '%s' not found, skipping", storage_name)

    return resolved_targets


class _GpuCandidateFailure(Exception):
    """Internal: signals one candidate GPU exhausted its RAM x storage matrix."""

    def __init__(
        self,
        gpu_type: str,
        reason: str,
        attempted_pairs: list[dict[str, Any]],
        last_error: Exception | None,
    ) -> None:
        super().__init__(reason)
        self.gpu_type = gpu_type
        self.reason = reason
        self.attempted_pairs = attempted_pairs
        self.last_error = last_error


async def _try_launch_one_gpu(
    config: RunPodConfig,
    gpu_type: str,
    *,
    pod_name: str,
    hooks: EventHooks,
    public_key_string: str | None,
    ram_tiers: list[int],
    storage_targets: list[tuple[str | None, str | None]],
    expanded_storage_ids: set[str],
) -> Pod:
    """Resolve a single GPU type and walk the RAM-tier x storage matrix.

    Returns a launched Pod on success, otherwise raises ``_GpuCandidateFailure``
    summarising the per-pair errors for this candidate.
    """
    gpu_info = await asyncio.to_thread(find_gpu_type, gpu_type, config.api_key)
    if not gpu_info:
        raise _GpuCandidateFailure(
            gpu_type=gpu_type,
            reason=f"{gpu_type}: GPU type could not be resolved",
            attempted_pairs=[],
            last_error=None,
        )

    attempted_pairs: list[dict[str, Any]] = []
    last_error: Exception | None = None

    for ram_tier in ram_tiers:
        for storage_name, storage_volume_id in storage_targets:
            attempted_pairs.append(
                {
                    "gpu_type": gpu_type,
                    "ram_tier": ram_tier,
                    "storage_name": storage_name,
                    "storage_volume_id": storage_volume_id,
                }
            )

            if storage_volume_id and storage_volume_id not in expanded_storage_ids:
                await asyncio.to_thread(
                    check_and_expand_storage,
                    config.api_key,
                    storage_volume_id,
                    50,
                    storage_name,
                )
                expanded_storage_ids.add(storage_volume_id)

            try:
                pod_details = await asyncio.to_thread(
                    create_pod,
                    api_key=config.api_key,
                    gpu_type_id=gpu_info["id"],
                    image_name=config.worker_image,
                    name=pod_name,
                    network_volume_id=storage_volume_id,
                    volume_mount_path=config.volume_mount_path,
                    disk_in_gb=config.disk_size_gb,
                    container_disk_in_gb=config.container_disk_gb,
                    public_key_string=public_key_string,
                    env_vars=config.env_vars,
                    min_vcpu_count=config.min_vcpu_count,
                    min_memory_in_gb=ram_tier,
                    template_id=config.template_id,
                    ports=config.ports,
                )
            except Exception as exc:
                last_error = exc
                error_message = str(exc).lower()
                if "no longer any instances available" in error_message:
                    logger.warning(
                        "No instances available for gpu=%s storage=%s ram=%sGB",
                        gpu_type,
                        storage_name,
                        ram_tier,
                    )
                else:
                    logger.warning(
                        "Pod creation failed for gpu=%s storage=%s ram=%sGB: %s",
                        gpu_type,
                        storage_name,
                        ram_tier,
                        exc,
                    )
                continue

            pod = Pod(
                pod_id=pod_details["id"],
                name=pod_name,
                config=config,
                hooks=hooks,
                ram_tier=ram_tier,
                storage_volume=storage_volume_id,
            )
            await _emit_state(
                hooks,
                pod.id,
                PodState.PROVISIONING,
                {
                    "name": pod_name,
                    "gpu_type": gpu_type,
                    "ram_tier": ram_tier,
                    "storage_name": storage_name,
                    "storage_volume_id": storage_volume_id,
                    "pod_details": pod_details,
                },
            )
            return pod

    reason = (
        f"{gpu_type}: failed RAM tiers {ram_tiers} across storages "
        f"{[pair['storage_name'] for pair in attempted_pairs] or [None]}"
    )
    if last_error is not None:
        reason = f"{reason} ({last_error})"
    raise _GpuCandidateFailure(
        gpu_type=gpu_type,
        reason=reason,
        attempted_pairs=attempted_pairs,
        last_error=last_error,
    )


async def launch(
    config: RunPodConfig,
    *,
    name: str | None = None,
    hooks: EventHooks | None = None,
) -> Pod:
    hooks = hooks or EventHooks()
    pod_name = name or f"{config.name_prefix}-{int(time.time())}"

    candidates = config.gpu_type_candidates
    if not candidates:
        error = LaunchFailure("No GPU types configured for launch")
        await _emit_error(hooks, error, {"name": pod_name})
        raise error

    await _emit_state(hooks, None, PodState.PROVISIONING, {"name": pod_name})

    public_key_string = _resolve_public_key_string(config)
    ram_tiers = _build_ram_tiers(config)
    storage_targets = await _resolve_storage_targets(config)

    input_storages = [value for value in [config.storage_name, *config.storage_volumes] if value]
    if input_storages and not storage_targets:
        error = LaunchFailure(
            f"Configured storage volumes could not be resolved: {', '.join(input_storages)}"
        )
        await _emit_error(
            hooks,
            error,
            {"storages": input_storages, "name": pod_name},
        )
        raise error

    expanded_storage_ids: set[str] = set()
    candidate_failures: list[_GpuCandidateFailure] = []

    for gpu_type in candidates:
        await _emit_state(
            hooks,
            None,
            PodState.PROVISIONING,
            {"name": pod_name, "gpu_type": gpu_type},
        )
        try:
            return await _try_launch_one_gpu(
                config,
                gpu_type,
                pod_name=pod_name,
                hooks=hooks,
                public_key_string=public_key_string,
                ram_tiers=ram_tiers,
                storage_targets=storage_targets,
                expanded_storage_ids=expanded_storage_ids,
            )
        except _GpuCandidateFailure as failure:
            candidate_failures.append(failure)
            continue

    reasons = "; ".join(failure.reason for failure in candidate_failures) or "no attempts made"
    aggregated_pairs = [pair for failure in candidate_failures for pair in failure.attempted_pairs]
    last_underlying = next(
        (failure.last_error for failure in reversed(candidate_failures) if failure.last_error),
        None,
    )

    error = LaunchFailure(
        f"Failed to launch pod across GPU candidates {list(candidates)}: {reasons}"
    )
    await _emit_error(
        hooks,
        error,
        {
            "name": pod_name,
            "gpu_candidates": list(candidates),
            "ram_tiers": ram_tiers,
            "attempted_pairs": aggregated_pairs,
            "candidate_reasons": [failure.reason for failure in candidate_failures],
            "last_error": str(last_underlying) if last_underlying else None,
        },
    )
    if last_underlying is not None:
        raise error from last_underlying
    raise error


__all__ = ["find_gpu_type", "get_network_volumes", "launch"]
