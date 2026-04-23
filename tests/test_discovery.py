from __future__ import annotations

import asyncio

import pytest

from runpod_lifecycle import discovery
from runpod_lifecycle.errors import LaunchFailure, TerminateError
from runpod_lifecycle.events import EventHooks, PodState
from runpod_lifecycle.pod import Pod


def _raw_pod(
    pod_id: str,
    *,
    name: str | None = None,
    desired: str = "RUNNING",
    cost: float = 0.5,
    uptime: int | None = 100,
) -> dict:
    return {
        "id": pod_id,
        "name": name,
        "desiredStatus": desired,
        "actualStatus": desired,
        "machineType": "RTX 4090",
        "imageName": "runpod/pytorch:latest",
        "createdAt": "2026-04-01T00:00:00Z",
        "costPerHr": cost,
        "runtime": {"uptimeInSeconds": uptime, "ports": [{"privatePort": 22}]} if uptime is not None else None,
        "networkVolumeId": "vol-1",
    }


def test_list_pods_returns_summaries(runpod_sdk_mock) -> None:
    runpod_sdk_mock.get_pods.return_value = [_raw_pod("a", name="gpu_1"), _raw_pod("b", name="other")]
    summaries = asyncio.run(discovery.list_pods("test"))
    assert [s.id for s in summaries] == ["a", "b"]
    assert summaries[0].cost_per_hr == 0.5
    assert summaries[0].gpu_type == "RTX 4090"


def test_list_pods_name_prefix_filter(runpod_sdk_mock) -> None:
    runpod_sdk_mock.get_pods.return_value = [
        _raw_pod("a", name="gpu_1"),
        _raw_pod("b", name="other"),
        _raw_pod("c", name="gpu_2"),
    ]
    summaries = asyncio.run(discovery.list_pods("test", name_prefix="gpu_"))
    assert [s.id for s in summaries] == ["a", "c"]


def test_find_pods_filters_by_predicate(runpod_sdk_mock) -> None:
    runpod_sdk_mock.get_pods.return_value = [
        _raw_pod("a", cost=0.5),
        _raw_pod("b", cost=2.0),
        _raw_pod("c", cost=1.5),
    ]
    summaries = asyncio.run(discovery.find_pods("test", lambda p: p.cost_per_hr > 1.0))
    assert [s.id for s in summaries] == ["b", "c"]


def test_find_orphans_excludes_known(runpod_sdk_mock) -> None:
    runpod_sdk_mock.get_pods.return_value = [_raw_pod(x) for x in ["a", "b", "c", "d"]]
    orphans = asyncio.run(discovery.find_orphans("test", known_pod_ids={"a", "b"}))
    assert [s.id for s in orphans] == ["c", "d"]


def test_find_orphans_filters_by_age(runpod_sdk_mock) -> None:
    runpod_sdk_mock.get_pods.return_value = [
        _raw_pod("young", uptime=60),
        _raw_pod("old", uptime=7200),
    ]
    orphans = asyncio.run(discovery.find_orphans("test", known_pod_ids=[], older_than_seconds=3600))
    assert [s.id for s in orphans] == ["old"]


def test_find_orphans_skips_inactive_status(runpod_sdk_mock) -> None:
    runpod_sdk_mock.get_pods.return_value = [
        _raw_pod("running"),
        _raw_pod("stopped", desired="STOPPED"),
        _raw_pod("terminated", desired="TERMINATED"),
        _raw_pod("provisioning", desired="PROVISIONING"),
    ]
    orphans = asyncio.run(discovery.find_orphans("test", known_pod_ids=[]))
    assert sorted(s.id for s in orphans) == ["provisioning", "running"]


def test_get_pod_attaches_to_existing(runpod_sdk_mock, base_config) -> None:
    runpod_sdk_mock.get_pod.return_value = {
        "id": "abc",
        "desiredStatus": "RUNNING",
        "actualStatus": "RUNNING",
        "runtime": {"ports": [], "uptimeInSeconds": 0},
    }
    pod = asyncio.run(discovery.get_pod("abc", base_config))
    assert isinstance(pod, Pod)
    assert pod.id == "abc"


def test_get_pod_missing_raises(runpod_sdk_mock, base_config) -> None:
    runpod_sdk_mock.get_pod.return_value = None
    with pytest.raises(LaunchFailure):
        asyncio.run(discovery.get_pod("missing", base_config))


def test_module_terminate_happy_path(runpod_sdk_mock) -> None:
    events: list[tuple[str, str | None]] = []

    async def on_state(event) -> None:
        events.append((event.state.value, event.pod_id))

    asyncio.run(discovery.terminate("p1", "test", hooks=EventHooks(on_state_change=on_state)))
    runpod_sdk_mock.terminate_pod.assert_called_once_with("p1")
    assert events == [(PodState.TERMINATED.value, "p1")]


def test_module_terminate_failure_emits_error_and_raises(runpod_sdk_mock) -> None:
    runpod_sdk_mock.terminate_pod.side_effect = RuntimeError("boom")
    errors: list[str] = []

    def on_error(err: Exception, detail: dict) -> None:
        errors.append(str(err))

    with pytest.raises(TerminateError):
        asyncio.run(discovery.terminate("p1", "test", hooks=EventHooks(on_error=on_error)))
    assert errors == ["boom"]


def test_cost_summary_math() -> None:
    pods = [discovery.PodSummary(
        id=str(i), name=None, desired_status="RUNNING", actual_status="RUNNING",
        gpu_type=None, image=None, created_at=None, cost_per_hr=0.5,
        uptime_seconds=0, ports=[], network_volume_id=None,
    ) for i in range(3)]
    cost = discovery.cost_summary(pods)
    assert cost == {"total_per_hr": 1.5, "daily": 36.0, "monthly": 1080.0}
