from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from runpod_lifecycle.config import RunPodConfig


@pytest.fixture
def runpod_sdk_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock = MagicMock()
    mock.get_gpus.return_value = []
    mock.create_pod.return_value = {}
    mock.get_pod.return_value = {}
    mock.terminate_pod.return_value = None
    mock.get_network_volumes.return_value = []
    monkeypatch.setattr("runpod_lifecycle.api.runpod", mock)
    return mock


@pytest.fixture
def base_config() -> RunPodConfig:
    return RunPodConfig(
        api_key="test",
        storage_volumes=("vol-a", "vol-b"),
        ram_tiers=(64, 32, 16),
        ssh_public_key="ssh-ed25519 AAAA test",
    )


@pytest.fixture
def volumeless_config() -> RunPodConfig:
    return RunPodConfig(
        api_key="test",
        storage_name=None,
        storage_volumes=(),
    )
