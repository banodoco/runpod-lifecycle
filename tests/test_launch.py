from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from runpod_lifecycle.errors import LaunchFailure
from runpod_lifecycle.events import EventHooks, PodState
from runpod_lifecycle.lifecycle import launch, launch_when_available
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
        lambda *args, **kwargs: {"ok": True},
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
    assert pod._gpu_type == base_config.gpu_type_candidates[0]
    assert pod._storage_name == "vol-a"
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
        lambda *args, **kwargs: {"ok": True},
    )
    monkeypatch.setattr("runpod_lifecycle.lifecycle.create_pod", create_pod_mock)

    pod = asyncio.run(launch(base_config, name="ram-fallback"))

    assert pod._ram_tier == 32
    assert [call.kwargs["min_memory_in_gb"] for call in create_pod_mock.call_args_list] == [64, 64, 32]


def test_launch_when_available_retries_until_capacity_appears(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_pod_mock = MagicMock(
        side_effect=[
            RuntimeError("no longer any instances available"),
            RuntimeError("no longer any instances available"),
            RuntimeError("no longer any instances available"),
            RuntimeError("no longer any instances available"),
            {"id": "pod-claimed"},
        ]
    )
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

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
        lambda *args, **kwargs: {"ok": True},
    )
    monkeypatch.setattr("runpod_lifecycle.lifecycle.create_pod", create_pod_mock)
    monkeypatch.setattr("runpod_lifecycle.lifecycle.asyncio.sleep", fake_sleep)

    pod = asyncio.run(
        launch_when_available(
            base_config.merge(ram_tiers=(64,), storage_volumes=()),
            name="wait-for-4090",
            max_wait_sec=120,
            retry_interval_sec=10,
        )
    )

    assert pod.id == "pod-claimed"
    assert sleeps == [10, 10, 10, 10]
    assert create_pod_mock.call_count == 5


def test_launch_when_available_times_out_after_bounded_retries(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_pod_mock = MagicMock(side_effect=RuntimeError("no longer any instances available"))
    clock = {"value": 0.0}
    sleeps: list[float] = []

    def fake_monotonic() -> float:
        return clock["value"]

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["value"] += seconds

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
        lambda *args, **kwargs: {"ok": True},
    )
    monkeypatch.setattr("runpod_lifecycle.lifecycle.create_pod", create_pod_mock)
    monkeypatch.setattr("runpod_lifecycle.lifecycle.time.monotonic", fake_monotonic)
    monkeypatch.setattr("runpod_lifecycle.lifecycle.asyncio.sleep", fake_sleep)

    with pytest.raises(LaunchFailure, match="Capacity did not become available"):
        asyncio.run(
            launch_when_available(
                base_config.merge(ram_tiers=(64,), storage_volumes=()),
                name="wait-timeout",
                max_wait_sec=15,
                retry_interval_sec=10,
            )
        )

    assert sleeps == [10, 5]
    assert create_pod_mock.call_count == 3


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
        lambda *args, **kwargs: {"ok": True},
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
        lambda *args, **kwargs: {"ok": True},
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


def test_launch_accepts_single_string_gpu_type(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_pod_mock = MagicMock(return_value={"id": "pod-string"})
    seen: list[str] = []

    def fake_find(gpu_type: str, api_key: str) -> dict[str, str]:
        seen.append(gpu_type)
        return {"id": f"id-{gpu_type}", "displayName": gpu_type}

    monkeypatch.setattr("runpod_lifecycle.lifecycle.find_gpu_type", fake_find)
    monkeypatch.setattr(
        "runpod_lifecycle.lifecycle.get_storage_volume_id",
        lambda api_key, storage_name: {"vol-a": "id-a", "vol-b": "id-b"}[storage_name],
    )
    monkeypatch.setattr(
        "runpod_lifecycle.lifecycle.check_and_expand_storage",
        lambda *args, **kwargs: {"ok": True},
    )
    monkeypatch.setattr("runpod_lifecycle.lifecycle.create_pod", create_pod_mock)

    cfg = base_config.merge(gpu_type="NVIDIA L40S")
    assert cfg.gpu_type == "NVIDIA L40S"  # single string preserved
    assert cfg.gpu_type_candidates == ("NVIDIA L40S",)

    pod = asyncio.run(launch(cfg, name="single-str"))
    assert pod.id == "pod-string"
    assert seen == ["NVIDIA L40S"]


def test_launch_accepts_single_item_list_gpu_type(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_pod_mock = MagicMock(return_value={"id": "pod-list"})
    seen: list[str] = []

    def fake_find(gpu_type: str, api_key: str) -> dict[str, str]:
        seen.append(gpu_type)
        return {"id": f"id-{gpu_type}", "displayName": gpu_type}

    monkeypatch.setattr("runpod_lifecycle.lifecycle.find_gpu_type", fake_find)
    monkeypatch.setattr(
        "runpod_lifecycle.lifecycle.get_storage_volume_id",
        lambda api_key, storage_name: {"vol-a": "id-a", "vol-b": "id-b"}[storage_name],
    )
    monkeypatch.setattr(
        "runpod_lifecycle.lifecycle.check_and_expand_storage",
        lambda *args, **kwargs: {"ok": True},
    )
    monkeypatch.setattr("runpod_lifecycle.lifecycle.create_pod", create_pod_mock)

    cfg = base_config.merge(gpu_type=["NVIDIA L40S"])
    assert cfg.gpu_type == ("NVIDIA L40S",)  # list normalized to tuple
    assert cfg.gpu_type_candidates == ("NVIDIA L40S",)

    pod = asyncio.run(launch(cfg, name="single-list"))
    assert pod.id == "pod-list"
    assert seen == ["NVIDIA L40S"]


def test_launch_falls_back_through_gpu_candidates(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA RTX A6000",
        "NVIDIA L40S",
    ]
    # First two GPUs: every create_pod call fails. Third: succeeds on first try.
    seen_gpus: list[str] = []

    def fake_find(gpu_type: str, api_key: str) -> dict[str, str]:
        return {"id": f"id-{gpu_type}", "displayName": gpu_type}

    def fake_create(**kwargs: object) -> dict[str, str]:
        gpu_id = kwargs["gpu_type_id"]
        assert isinstance(gpu_id, str)
        seen_gpus.append(gpu_id)
        if gpu_id == f"id-{candidates[2]}":
            return {"id": "pod-third"}
        raise RuntimeError("no longer any instances available")

    monkeypatch.setattr("runpod_lifecycle.lifecycle.find_gpu_type", fake_find)
    monkeypatch.setattr(
        "runpod_lifecycle.lifecycle.get_storage_volume_id",
        lambda api_key, storage_name: {"vol-a": "id-a", "vol-b": "id-b"}[storage_name],
    )
    monkeypatch.setattr(
        "runpod_lifecycle.lifecycle.check_and_expand_storage",
        lambda *args, **kwargs: {"ok": True},
    )
    monkeypatch.setattr("runpod_lifecycle.lifecycle.create_pod", fake_create)

    events: list[tuple[str | None, str, dict[str, object]]] = []

    async def on_state(event) -> None:  # type: ignore[no-untyped-def]
        events.append((event.pod_id, event.state.value, event.detail))

    cfg = base_config.merge(gpu_type=candidates)
    pod = asyncio.run(
        launch(cfg, name="multi", hooks=EventHooks(on_state_change=on_state))
    )

    assert pod.id == "pod-third"
    # Third GPU should appear in seen_gpus exactly once (succeeded immediately).
    # First two should have exhausted the RAM x storage matrix (3 tiers * 2 storages = 6 attempts each).
    assert seen_gpus.count(f"id-{candidates[2]}") == 1

    # Provisioning events should record per-candidate iteration in metadata.
    provisioning_gpu_types = [
        detail.get("gpu_type")
        for (_pid, state, detail) in events
        if state == PodState.PROVISIONING.value and "gpu_type" in detail and _pid is None
    ]
    assert provisioning_gpu_types == candidates


def test_launch_all_gpu_candidates_fail_raises_aggregated(
    base_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA RTX A6000",
        "NVIDIA L40S",
    ]

    monkeypatch.setattr(
        "runpod_lifecycle.lifecycle.find_gpu_type",
        lambda gpu_type, api_key: {"id": f"id-{gpu_type}", "displayName": gpu_type},
    )
    monkeypatch.setattr(
        "runpod_lifecycle.lifecycle.get_storage_volume_id",
        lambda api_key, storage_name: {"vol-a": "id-a", "vol-b": "id-b"}[storage_name],
    )
    monkeypatch.setattr(
        "runpod_lifecycle.lifecycle.check_and_expand_storage",
        lambda *args, **kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        "runpod_lifecycle.lifecycle.create_pod",
        MagicMock(side_effect=RuntimeError("no longer any instances available")),
    )

    cfg = base_config.merge(gpu_type=candidates)
    with pytest.raises(LaunchFailure) as excinfo:
        asyncio.run(launch(cfg, name="all-fail"))

    message = str(excinfo.value)
    for gpu_type in candidates:
        assert gpu_type in message, f"expected {gpu_type!r} in {message!r}"


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
