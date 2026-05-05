from __future__ import annotations

import pytest

from runpod_lifecycle.config import RunPodConfig


def test_from_env_reads_documented_runpod_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("runpod_lifecycle.config.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("RUNPOD_API_KEY", "api-key")
    monkeypatch.setenv("RUNPOD_GPU_TYPE", "GPU X")
    monkeypatch.setenv("RUNPOD_WORKER_IMAGE", "image:test")
    monkeypatch.setenv("RUNPOD_TEMPLATE_ID", "template-1")
    monkeypatch.setenv("RUNPOD_VOLUME_MOUNT_PATH", "/data")
    monkeypatch.setenv("RUNPOD_DISK_SIZE_GB", "25")
    monkeypatch.setenv("RUNPOD_CONTAINER_DISK_GB", "60")
    monkeypatch.setenv("RUNPOD_MIN_VCPU_COUNT", "12")
    monkeypatch.setenv("RUNPOD_MIN_MEMORY_GB", "48")
    monkeypatch.setenv("RUNPOD_RAM_TIER_FALLBACK", "false")
    monkeypatch.setenv("RUNPOD_RAM_TIERS", "80,64,48")
    monkeypatch.setenv("RUNPOD_STORAGE_VOLUMES", "vol-a, vol-b")
    monkeypatch.setenv("RUNPOD_STORAGE_NAME", "primary")
    monkeypatch.setenv("RUNPOD_SSH_PUBLIC_KEY", "ssh-ed25519 AAAA test")
    monkeypatch.setenv("RUNPOD_SSH_PRIVATE_KEY", "private-key")
    monkeypatch.setenv("RUNPOD_SSH_PUBLIC_KEY_PATH", "~/.ssh/test.pub")
    monkeypatch.setenv("RUNPOD_SSH_PRIVATE_KEY_PATH", "~/.ssh/test")
    monkeypatch.setenv("RUNPOD_ENV_VARS", "{\"HELLO\": \"world\"}")
    monkeypatch.setenv("RUNPOD_NAME_PREFIX", "worker")

    config = RunPodConfig.from_env()

    assert config.api_key == "api-key"
    assert config.gpu_type == "GPU X"
    assert config.worker_image == "image:test"
    assert config.template_id == "template-1"
    assert config.volume_mount_path == "/data"
    assert config.disk_size_gb == 25
    assert config.container_disk_gb == 60
    assert config.min_vcpu_count == 12
    assert config.min_memory_gb == 48
    assert config.ram_tiers_enabled is False
    assert config.ram_tiers == (80, 64, 48)
    assert config.storage_volumes == ("vol-a", "vol-b")
    assert config.storage_name == "primary"
    assert config.ssh_public_key == "ssh-ed25519 AAAA test"
    assert config.ssh_private_key == "private-key"
    assert config.ssh_public_key_path == "~/.ssh/test.pub"
    assert config.ssh_private_key_path == "~/.ssh/test"
    assert config.env_vars == {"HELLO": "world"}
    assert config.name_prefix == "worker"


def test_storage_volumes_are_comma_split(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("runpod_lifecycle.config.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("RUNPOD_API_KEY", "api-key")
    monkeypatch.setenv("RUNPOD_STORAGE_VOLUMES", "one, two ,three")

    config = RunPodConfig.from_env()

    assert config.storage_volumes == ("one", "two", "three")


def test_from_env_defaults_to_dual_stack_disk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("runpod_lifecycle.config.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("RUNPOD_API_KEY", "api-key")
    monkeypatch.delenv("RUNPOD_DISK_SIZE_GB", raising=False)
    monkeypatch.delenv("RUNPOD_CONTAINER_DISK_GB", raising=False)

    config = RunPodConfig.from_env()

    assert config.disk_size_gb == 200
    assert config.container_disk_gb == 200


def test_missing_api_key_raises_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("runpod_lifecycle.config.load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)

    with pytest.raises(ValueError, match="RUNPOD_API_KEY"):
        RunPodConfig.from_env()


def test_merge_returns_new_instance_with_overrides() -> None:
    original = RunPodConfig(api_key="api-key", name_prefix="pod", min_memory_gb=32)

    merged = original.merge(name_prefix="worker", min_memory_gb=64)

    assert merged is not original
    assert merged.name_prefix == "worker"
    assert merged.min_memory_gb == 64
    assert original.name_prefix == "pod"
    assert original.min_memory_gb == 32
