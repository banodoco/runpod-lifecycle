"""Configuration primitives for the standalone RunPod lifecycle package."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import Any

from dotenv import load_dotenv

DEFAULT_GPU_TYPE = "NVIDIA GeForce RTX 4090"
DEFAULT_WORKER_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
DEFAULT_TEMPLATE_ID = "runpod-torch-v240"
DEFAULT_VOLUME_MOUNT_PATH = "/workspace"
DEFAULT_RAM_TIERS = (72, 60, 48, 32, 16)


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str | None, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    return int(value)


def _parse_csv_tuple(value: str | None) -> tuple[str, ...]:
    if value is None or value.strip() == "":
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_int_tuple(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    parts = _parse_csv_tuple(value)
    if not parts:
        return default
    return tuple(int(part) for part in parts)


def _parse_env_vars(value: str | None) -> dict[str, str]:
    if value is None or value.strip() == "":
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("RUNPOD_ENV_VARS must decode to a JSON object")
    return {str(key): str(item) for key, item in parsed.items()}


@dataclass(slots=True)
class RunPodConfig:
    api_key: str
    gpu_type: str = DEFAULT_GPU_TYPE
    worker_image: str = DEFAULT_WORKER_IMAGE
    template_id: str = DEFAULT_TEMPLATE_ID
    volume_mount_path: str = DEFAULT_VOLUME_MOUNT_PATH
    disk_size_gb: int = 200
    container_disk_gb: int = 200
    min_vcpu_count: int = 8
    min_memory_gb: int = 32
    ram_tiers_enabled: bool = True
    ram_tiers: tuple[int, ...] = DEFAULT_RAM_TIERS
    storage_volumes: tuple[str, ...] = ()
    storage_name: str | None = None
    ssh_public_key: str | None = None
    ssh_private_key: str | None = None
    ssh_public_key_path: str | None = None
    ssh_private_key_path: str | None = None
    env_vars: dict[str, str] = field(default_factory=dict)
    name_prefix: str = "pod"

    @classmethod
    def from_env(cls, **overrides: Any) -> "RunPodConfig":
        load_dotenv()

        data: dict[str, Any] = {
            "api_key": os.getenv("RUNPOD_API_KEY"),
            "gpu_type": os.getenv("RUNPOD_GPU_TYPE", DEFAULT_GPU_TYPE),
            "worker_image": os.getenv("RUNPOD_WORKER_IMAGE", DEFAULT_WORKER_IMAGE),
            "template_id": os.getenv("RUNPOD_TEMPLATE_ID", DEFAULT_TEMPLATE_ID),
            "volume_mount_path": os.getenv("RUNPOD_VOLUME_MOUNT_PATH", DEFAULT_VOLUME_MOUNT_PATH),
            "disk_size_gb": _parse_int(os.getenv("RUNPOD_DISK_SIZE_GB"), 200),
            "container_disk_gb": _parse_int(os.getenv("RUNPOD_CONTAINER_DISK_GB"), 200),
            "min_vcpu_count": _parse_int(os.getenv("RUNPOD_MIN_VCPU_COUNT"), 8),
            "min_memory_gb": _parse_int(os.getenv("RUNPOD_MIN_MEMORY_GB"), 32),
            "ram_tiers_enabled": _parse_bool(
                os.getenv("RUNPOD_RAM_TIERS_ENABLED", os.getenv("RUNPOD_RAM_TIER_FALLBACK")),
                True,
            ),
            "ram_tiers": _parse_int_tuple(os.getenv("RUNPOD_RAM_TIERS"), DEFAULT_RAM_TIERS),
            "storage_volumes": _parse_csv_tuple(os.getenv("RUNPOD_STORAGE_VOLUMES")),
            "storage_name": os.getenv("RUNPOD_STORAGE_NAME"),
            "ssh_public_key": os.getenv("RUNPOD_SSH_PUBLIC_KEY"),
            "ssh_private_key": os.getenv("RUNPOD_SSH_PRIVATE_KEY"),
            "ssh_public_key_path": os.getenv("RUNPOD_SSH_PUBLIC_KEY_PATH"),
            "ssh_private_key_path": os.getenv("RUNPOD_SSH_PRIVATE_KEY_PATH"),
            "env_vars": _parse_env_vars(os.getenv("RUNPOD_ENV_VARS")),
            "name_prefix": os.getenv("RUNPOD_NAME_PREFIX", "pod"),
        }
        data.update(overrides)

        if not data.get("api_key"):
            raise ValueError("RUNPOD_API_KEY environment variable is required")

        return cls(**data)

    def merge(self, **overrides: Any) -> "RunPodConfig":
        return replace(self, **overrides)
