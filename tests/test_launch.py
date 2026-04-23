from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from runpod_lifecycle.errors import LaunchFailure
from runpod_lifecycle.events import EventHooks, PodState
from runpod_lifecycle.lifecycle import launch
from runpod_lifecycle.pod import Pod


def test_launch_happy_path_returns_pod_and_emits_provisioning(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str | None, str, dict[str, object]]] = []

    async def on_state(event) -> None:  # type: ignore[no-untyped-def]
        events.append((event.pod_id, event.state.value, event.detail))

    create_pod_mock = MagicMock(return_value={"id": "pod-123"})

    monkeypatch.setattr(
        "runpod_lifecycle.lifecycle.find_gpu_type",
        lambda gpu_type, api_key: {"id": "gpu-1", "displayName": gpu_type},
    )
    monkeypatch.setattr(
        "runpod_lifecycle.lifecycle.get_storage_volume_id",
        lambda api_key, storage_name: {"vol-a": "id-a", "vol-b": "id-b"}[storage_name],
    )
    monkeypatch.setattr(
        "runpod_lifecycle.lifecycle.check_and_expand_storage",
        lambda api_key, volume_id, min_free_gb=50, storage_name=None: {"ok": True},
    )
    monkeypatch.setattr("runpod_lifecycle.lifecycle.create_pod", create_pod_mock)

    pod = asyncio.run(
        launch(
            base_config,
            name="happy-pod",
            hooks=EventHooks(on_state_change=on_state),
        )
    )

    assert isinstance(pod, Pod)
    assert pod.id == "pod-123"
    assert pod._ram_tier == 64
    assert pod._storage_volume == "id-a"
    assert events[0] == (None, PodState.PROVISIONING.value, {"name": "happy-pod"})
    assert create_pod_mock.call_count == 1
    assert create_pod_mock.call_args.kwargs["min_memory_in_gb"] == 64
    assert create_pod_mock.call_args.kwargs["network_volume_id"] == "id-a"


def test_launch_uses_ram_tier_fallback(base_config, monkeypatch: pytest.MonkeyPatch) -> None:
    create_pod_mock = MagicMock(
        side_effect=[
            RuntimeError("no longer any instances available"),
            RuntimeError("no longer any instances available"),
            {"id": "pod-123"},
        ]
    )

    monkeypatch.setattr(
        "runpod_lifecycle.lifecycle.find_gpu_type",
        lambda gpu_type, api_key: {"id": "gpu-1", "displayName": gpu_type},
    )
    monkeypatch.setattr(
        "runpod_lifecycle.lifecycle.get_storage_volume_id",
        lambda api_key, storage_name: {"vol-a": "id-a", "vol-b": "id-b"}[storage_name],
    )
    monkeypatch.setattr(
        "runpod_lifecycle.lifecycle.check_and_expand_storage",
        lambda api_key, volume_id, min_free_gb=50, storage_name=None: {"ok": True},
    )
    monkeypatch.setattr("runpod_lifecycle.lifecycle.create_pod", create_pod_mock)

    pod = asyncio.run(launch(base_config, name="ram-fallback"))

    assert pod._ram_tier == 32
    assert [call.kwargs["min_memory_in_gb"] for call in create_pod_mock.call_args_list] == [64, 64, 32]


def test_launch_falls_back_to_second_storage_within_tier(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_pod_mock = MagicMock(side_effect=[RuntimeError("boom"), {"id": "pod-123"}])

    monkeypatch.setattr(
        "runpod_lifecycle.lifecycle.find_gpu_type",
        lambda gpu_type, api_key: {"id": "gpu-1", "displayName": gpu_type},
    )
    monkeypatch.setattr(
        "runpod_lifecycle.lifecycle.get_storage_volume_id",
        lambda api_key, storage_name: {"vol-a": "id-a", "vol-b": "id-b"}[storage_name],
    )
    monkeypatch.setattr(
        "runpod_lifecycle.lifecycle.check_and_expand_storage",
        lambda api_key, volume_id, min_free_gb=50, storage_name=None: {"ok": True},
    )
    monkeypatch.setattr("runpod_lifecycle.lifecycle.create_pod", create_pod_mock)

    pod = asyncio.run(launch(base_config, name="storage-fallback"))

    assert pod._ram_tier == 64
    assert pod._storage_volume == "id-b"
    assert [call.kwargs["network_volume_id"] for call in create_pod_mock.call_args_list] == ["id-a", "id-b"]


def test_launch_exhausted_fallback_emits_on_error_once(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error_calls: list[tuple[str, dict[str, object]]] = []

    def on_error(error: Exception, detail: dict[str, object]) -> None:
        error_calls.append((str(error), detail))

    create_pod_mock = MagicMock(side_effect=RuntimeError("all launch attempts failed"))

    monkeypatch.setattr(
        "runpod_lifecycle.lifecycle.find_gpu_type",
        lambda gpu_type, api_key: {"id": "gpu-1", "displayName": gpu_type},
    )
    monkeypatch.setattr(
        "runpod_lifecycle.lifecycle.get_storage_volume_id",
        lambda api_key, storage_name: {"vol-a": "id-a", "vol-b": "id-b"}[storage_name],
    )
    monkeypatch.setattr(
        "runpod_lifecycle.lifecycle.check_and_expand_storage",
        lambda api_key, volume_id, min_free_gb=50, storage_name=None: {"ok": True},
    )
    monkeypatch.setattr("runpod_lifecycle.lifecycle.create_pod", create_pod_mock)

    with pytest.raises(LaunchFailure):
        asyncio.run(launch(base_config, name="exhausted", hooks=EventHooks(on_error=on_error)))

    assert create_pod_mock.call_count == 4
    assert len(error_calls) == 1
    assert error_calls[0][1]["last_error"] == "all launch attempts failed"


def test_launch_raises_before_create_when_gpu_missing(
    base_config,
    runpod_sdk_mock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("runpod_lifecycle.lifecycle.find_gpu_type", lambda gpu_type, api_key: None)

    with pytest.raises(LaunchFailure):
        asyncio.run(launch(base_config, name="missing-gpu"))

    assert runpod_sdk_mock.create_pod.call_count == 0


def test_launch_volumeless_uses_one_create_per_ram_tier_and_skips_storage_checks(
    volumeless_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_pod_mock = MagicMock(side_effect=RuntimeError("volumeless failure"))
    expand_mock = MagicMock()

    monkeypatch.setattr(
        "runpod_lifecycle.lifecycle.find_gpu_type",
        lambda gpu_type, api_key: {"id": "gpu-1", "displayName": gpu_type},
    )
    monkeypatch.setattr("runpod_lifecycle.lifecycle.create_pod", create_pod_mock)
    monkeypatch.setattr("runpod_lifecycle.lifecycle.check_and_expand_storage", expand_mock)

    with pytest.raises(LaunchFailure):
        asyncio.run(launch(volumeless_config, name="volumeless"))

    filtered_tiers = [tier for tier in volumeless_config.ram_tiers if tier >= volumeless_config.min_memory_gb]
    assert create_pod_mock.call_count == len(filtered_tiers)
    assert [call.kwargs["network_volume_id"] for call in create_pod_mock.call_args_list] == [None] * len(
        filtered_tiers
    )
    assert expand_mock.call_count == 0
