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


@pytest.mark.parametrize("evidence, expected", [
    ({"networkVolume": {"id": "vol-1"}}, "vol-1"),
    ({"networkVolumeId": "vol-1"}, "vol-1"),
    ({"networkVolumeId": "vol-1", "networkVolume": {"id": "vol-1"}}, "vol-1"),
    ({}, None),
    ({"networkVolume": None}, None),
    ({"networkVolume": {}}, None),
    ({"networkVolume": {"id": None}}, None),
    ({"networkVolume": {"id": ""}}, None),
    ({"networkVolume": {"id": " vol-1"}}, None),
    ({"networkVolume": {"id": 1}}, None),
    ({"networkVolume": {"id": ["vol-1"]}}, None),
    ({"networkVolume": [{"id": "vol-1"}]}, None),
    ({"networkVolume": [{"id": "vol-1"}, {"id": "vol-2"}]}, None),
    ({"networkVolumeId": ["vol-1"]}, None),
    ({"networkVolumeId": None, "networkVolume": {"id": "vol-1"}}, None),
    ({"networkVolumeId": "vol-1", "networkVolume": {"id": "vol-2"}}, None),
])
def test_attachment_normalization_is_exact(evidence, expected) -> None:
    assert discovery._to_summary({"id": "p1", **evidence}).network_volume_id == expected


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


def test_find_orphans_includes_starting_from_rest_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from runpod_lifecycle import api

    monkeypatch.setattr(api, "_get_runpod", lambda: None)
    calls = []
    body = {
        "pods": [
            {"id": "starting", "name": "gpu_starting", "status": "STARTING"},
            {"id": "exited", "name": "gpu_exited", "status": "EXITED"},
        ],
        "pagination": {"hasNextPage": False, "nextCursor": None},
    }

    def get(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=200, json=lambda: body)

    monkeypatch.setattr(api.httpx, "get", get)
    orphans = asyncio.run(discovery.find_orphans("fake", known_pod_ids=[]))

    assert [pod.id for pod in orphans] == ["starting"]
    assert orphans[0].desired_status == "STARTING"
    assert len(calls) == 1
    assert calls[0][0] == api.PODS_URL
    assert calls[0][1]["params"] == {}


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


def test_module_terminate_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, str | None]] = []

    async def on_state(event) -> None:
        events.append((event.state.value, event.pod_id))

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "runpod_lifecycle.discovery.api.terminate_pod",
        lambda pod_id, api_key: calls.append((pod_id, api_key)),
    )
    asyncio.run(discovery.terminate("p1", "test", hooks=EventHooks(on_state_change=on_state)))
    assert calls == [("p1", "test")]
    assert events == [(PodState.TERMINATED.value, "p1")]


def test_module_terminate_failure_emits_error_and_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_pod_id: str, _api_key: str) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("runpod_lifecycle.discovery.api.terminate_pod", fail)
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
