"""Configuration primitives for the standalone RunPod lifecycle package."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from dotenv import dotenv_values, load_dotenv


def astrid_env_file_path(environ: dict[str, str] | None = None) -> Path:
    """Return the shared Astrid dotenv path without importing Astrid itself."""

    env = os.environ if environ is None else environ
    astrid_home = env.get("ASTRID_HOME", "").strip()
    root = (Path(astrid_home).expanduser() if astrid_home else Path.home() / ".astrid").resolve()
    override = env.get("ASTRID_ENV_FILE", "").strip()
    if override:
        override_path = Path(override).expanduser()
        return override_path if override_path.is_absolute() else root / override_path
    return root / "astrid.env"


def load_runpod_env(env_file: Path | str | None = None) -> None:
    """Load shared Astrid credentials before tool-local dotenv settings.

    The shared file wins over the process environment and project-local
    dotenv copies. Environment-only operation remains available when the
    shared file is absent, as on CI and service hosts.
    """

    process_credential = os.environ.get("RUNPOD_API_KEY", "").strip()
    shared_file = astrid_env_file_path()
    shared_credential = ""
    if shared_file.is_file():
        shared_values = dotenv_values(shared_file, interpolate=False)
        shared_credential = str(shared_values.get("RUNPOD_API_KEY") or "").strip()
        # Shared API keys are literal strings; do not interpolate ${...}.
        load_dotenv(shared_file, override=True, interpolate=False)
    if env_file is None:
        load_dotenv(override=False)
    else:
        load_dotenv(env_file, override=False)
    # Project dotenv files remain useful for non-secret lifecycle settings,
    # but never act as a second local source for the RunPod credential. Keep a
    # pre-existing process injection or the shared-file value intact.
    if shared_credential:
        os.environ["RUNPOD_API_KEY"] = shared_credential
    elif process_credential:
        os.environ["RUNPOD_API_KEY"] = process_credential
    else:
        os.environ.pop("RUNPOD_API_KEY", None)

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


def _normalize_cuda_versions(value: str | Iterable[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    parts = (value,) if isinstance(value, str) else tuple(value)
    result: list[str] = []
    for part in parts:
        if not isinstance(part, str):
            raise TypeError("allowed_cuda_versions entries must be strings")
        version = part.strip()
        if not version or not re.fullmatch(r"\d+\.\d+", version):
            raise ValueError("allowed_cuda_versions entries must be non-empty CUDA versions like '12.4'")
        if version not in result:
            result.append(version)
    return tuple(result)


def _normalize_gpu_type(
    value: str | Iterable[str] | None,
) -> tuple[str, ...]:
    """Normalize a gpu_type input (str | list | tuple) to a tuple of strings.

    Empty/whitespace entries are dropped while preserving order. A single
    string is wrapped into a 1-tuple. ``None`` becomes an empty tuple.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        return (stripped,) if stripped else ()
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError(
                f"gpu_type entries must be strings, got {type(item).__name__}"
            )
        stripped = item.strip()
        if stripped:
            items.append(stripped)
    return tuple(items)


def _parse_gpu_type_env(value: str | None) -> str | tuple[str, ...]:
    """Parse RUNPOD_GPU_TYPE env. Returns a tuple when comma-separated."""
    if value is None:
        return DEFAULT_GPU_TYPE
    parts = _parse_csv_tuple(value)
    if len(parts) <= 1:
        # Preserve single-string behavior for backwards compatibility.
        return parts[0] if parts else DEFAULT_GPU_TYPE
    return parts


def _parse_optional_string(value: str | None) -> str | None:
    """Coalesce missing/blank env values to ``None``.

    ``os.getenv`` yields ``""`` when a variable is set to the empty string
    (e.g. ``RUNPOD_STORAGE_NAME=`` in a ``.env`` file). Empty strings are
    truthy in some downstream paths, so treat blank as unset.
    """
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


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
    gpu_type: str | tuple[str, ...] | list[str] = DEFAULT_GPU_TYPE
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
    ports: str | None = None
    allowed_cuda_versions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Normalize list inputs to a tuple while preserving str inputs as-is.
        if isinstance(self.gpu_type, list):
            object.__setattr__(self, "gpu_type", _normalize_gpu_type(self.gpu_type))
        elif isinstance(self.gpu_type, tuple):
            # Re-normalize tuples to strip empties / whitespace consistently.
            object.__setattr__(self, "gpu_type", _normalize_gpu_type(self.gpu_type))
        object.__setattr__(self, "allowed_cuda_versions", _normalize_cuda_versions(self.allowed_cuda_versions))

    @property
    def gpu_type_candidates(self) -> tuple[str, ...]:
        """Return the ordered list of GPU types to try, regardless of input form."""
        if isinstance(self.gpu_type, str):
            return (self.gpu_type,) if self.gpu_type else ()
        return tuple(self.gpu_type)

    @classmethod
    def from_env(cls, **overrides: Any) -> "RunPodConfig":
        load_runpod_env()

        raw_cuda_versions = os.getenv("RUNPOD_ALLOWED_CUDA_VERSIONS")
        if "allowed_cuda_versions" not in overrides and raw_cuda_versions is not None and not raw_cuda_versions.strip():
            raise ValueError("RUNPOD_ALLOWED_CUDA_VERSIONS must contain at least one version")
        if "allowed_cuda_versions" not in overrides and raw_cuda_versions is not None and not _parse_csv_tuple(raw_cuda_versions):
            raise ValueError("RUNPOD_ALLOWED_CUDA_VERSIONS must contain at least one version")

        # Astrid's subprocess environment policy treats names containing
        # ``PRIVATE_KEY``/``PUBLIC_KEY`` as secret-like, even when the value
        # is only a path.  The identity aliases are path-only names used at
        # that boundary; the documented variables remain canonical for
        # direct lifecycle-library callers.
        ssh_public_key_path = (
            os.getenv("RUNPOD_SSH_PUBLIC_KEY_PATH")
            or os.getenv("RUNPOD_SSH_IDENTITY_PUBLIC_PATH")
        )
        ssh_private_key_path = (
            os.getenv("RUNPOD_SSH_PRIVATE_KEY_PATH")
            or os.getenv("RUNPOD_SSH_IDENTITY_PATH")
        )

        data: dict[str, Any] = {
            "api_key": os.getenv("RUNPOD_API_KEY"),
            "gpu_type": _parse_gpu_type_env(os.getenv("RUNPOD_GPU_TYPE")),
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
            "storage_name": _parse_optional_string(os.getenv("RUNPOD_STORAGE_NAME")),
            # Prefer scoped filesystem identities over any stale inline material
            # inherited from another dotenv file or parent process. Explicit
            # keyword overrides below continue to have final precedence.
            "ssh_public_key": None
            if ssh_public_key_path
            else os.getenv("RUNPOD_SSH_PUBLIC_KEY"),
            "ssh_private_key": None
            if ssh_private_key_path
            else os.getenv("RUNPOD_SSH_PRIVATE_KEY"),
            "ssh_public_key_path": ssh_public_key_path,
            "ssh_private_key_path": ssh_private_key_path,
            "env_vars": _parse_env_vars(os.getenv("RUNPOD_ENV_VARS")),
            "name_prefix": os.getenv("RUNPOD_NAME_PREFIX", "pod"),
            "ports": _parse_optional_string(os.getenv("RUNPOD_PORTS")),
            "allowed_cuda_versions": _parse_csv_tuple(raw_cuda_versions),
        }
        data.update(overrides)

        if not data.get("api_key"):
            raise ValueError(
                "RUNPOD_API_KEY is required (run `astrid-credential set runpod` locally, "
                "or inject it through the process environment)"
            )

        return cls(**data)

    def merge(self, **overrides: Any) -> "RunPodConfig":
        return replace(self, **overrides)
