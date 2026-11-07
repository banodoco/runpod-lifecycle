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
    # Lifecycle owns the GraphQL builders; route them to the same mock so tests
    # assert on the arguments handed to them (api_key is transport-only).
    for name in ("create_pod", "get_pod"):
        monkeypatch.setattr(
            f"runpod_lifecycle.graphql.{name}",
            (lambda n: lambda *a, api_key=None, **kw: getattr(mock, n)(*a, **kw))(name),
        )
    monkeypatch.setattr(
        "runpod_lifecycle.graphql.get_pods", lambda api_key=None: mock.get_pods()
    )
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


@pytest.fixture(autouse=True)
def isolate_shared_astrid_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Keep the developer's real per-user credentials out of package tests."""

    monkeypatch.setenv("ASTRID_ENV_FILE", str(tmp_path / "isolated-astrid.env"))


@pytest.fixture(autouse=True)
def offline_transport_preflight(request, monkeypatch):
    # Fake provider payload tests must not depend on the test runner's egress.
    if not request.module.__name__.endswith("test_https"):
        monkeypatch.setattr("runpod_lifecycle.preflight.preflight_transport", lambda *_a, **_kw: None)
