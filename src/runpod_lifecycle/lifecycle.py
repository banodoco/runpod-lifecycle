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


async def launch(
    config: RunPodConfig,
    *,
    name: str | None = None,
    hooks: EventHooks | None = None,
) -> Pod:
    hooks = hooks or EventHooks()
    pod_name = name or f"{config.name_prefix}-{int(time.time())}"

    await _emit_state(hooks, None, PodState.PROVISIONING, {"name": pod_name})

    gpu_info = await asyncio.to_thread(find_gpu_type, config.gpu_type, config.api_key)
    if not gpu_info:
        error = LaunchFailure(f"RunPod GPU type '{config.gpu_type}' could not be resolved")
        await _emit_error(hooks, error, {"gpu_type": config.gpu_type, "name": pod_name})
        raise error

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
    attempted_pairs: list[dict[str, Any]] = []
    last_error: Exception | None = None

    for ram_tier in ram_tiers:
        for storage_name, storage_volume_id in storage_targets:
            attempted_pairs.append(
                {
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
                        "No instances available for storage=%s ram=%sGB",
                        storage_name,
                        ram_tier,
                    )
                else:
                    logger.warning(
                        "Pod creation failed for storage=%s ram=%sGB: %s",
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
                    "ram_tier": ram_tier,
                    "storage_name": storage_name,
                    "storage_volume_id": storage_volume_id,
                    "pod_details": pod_details,
                },
            )
            return pod

    if last_error is None:
        last_error = LaunchFailure("No pod launch attempts were made")

    error = LaunchFailure(
        f"Failed to create pod after trying RAM tiers {ram_tiers} across storages "
        f"{[pair['storage_name'] for pair in attempted_pairs] or [None]}: {last_error}"
    )
    await _emit_error(
        hooks,
        error,
        {
            "name": pod_name,
            "ram_tiers": ram_tiers,
            "attempted_pairs": attempted_pairs,
            "last_error": str(last_error),
        },
    )
    raise error from last_error


__all__ = ["find_gpu_type", "get_network_volumes", "launch"]
