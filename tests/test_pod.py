from __future__ import annotations

import asyncio
import time

import pytest

from runpod_lifecycle.errors import LaunchFailure, NotReadyTimeout
from runpod_lifecycle.events import EventHooks, PodState
from runpod_lifecycle.pod import Pod


def test_exec_ssh_delegates_and_updates_last_exec_at(base_config, monkeypatch: pytest.MonkeyPatch) -> None:
    pod = Pod("pod-1", "worker", base_config.merge(ssh_private_key="PRIVATE"))
    pod._ssh_details = {"ip": "1.2.3.4", "port": 2201, "password": "secret"}

    calls: list[object] = []

    class FakeSSHClient:
        def connect(self) -> None:
            calls.append("connect")

        def execute_command(self, cmd: str, timeout: int) -> tuple[int, str, str]:
            calls.append((cmd, timeout))
            return (0, "ok", "")

        def disconnect(self) -> None:
            calls.append("disconnect")

    monkeypatch.setattr(pod, "_build_ssh_client", lambda ssh_details: FakeSSHClient())

    before = pod._last_exec_at
    result = asyncio.run(pod.exec_ssh("echo test", timeout=45))

    assert result == (0, "ok", "")
    assert calls == ["connect", ("echo test", 45), "disconnect"]
    assert before is None
    assert pod._last_exec_at is not None


def test_is_idle_false_when_recent_activity(base_config, monkeypatch: pytest.MonkeyPatch) -> None:
    pod = Pod("pod-1", "worker", base_config)
    pod._last_exec_at = time.monotonic()

    def fail_exec(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("exec_ssh should not run for recent activity")

    monkeypatch.setattr(pod, "exec_ssh", fail_exec)

    assert asyncio.run(pod.is_idle(60)) is False


def test_is_idle_true_when_gpu_utilization_low(base_config, monkeypatch: pytest.MonkeyPatch) -> None:
    pod = Pod("pod-1", "worker", base_config)
    pod._last_exec_at = time.monotonic() - 120

    async def fake_exec(cmd: str, timeout: int = 600) -> tuple[int, str, str]:
        assert "nvidia-smi" in cmd
        return (0, "2\n", "")

    monkeypatch.setattr(pod, "exec_ssh", fake_exec)

    assert asyncio.run(pod.is_idle(60)) is True


def test_is_idle_false_when_gpu_utilization_high(base_config, monkeypatch: pytest.MonkeyPatch) -> None:
    pod = Pod("pod-1", "worker", base_config)
    pod._last_exec_at = time.monotonic() - 120

    async def fake_exec(cmd: str, timeout: int = 600) -> tuple[int, str, str]:
        assert "nvidia-smi" in cmd
        return (0, "50\n", "")

    monkeypatch.setattr(pod, "exec_ssh", fake_exec)

    assert asyncio.run(pod.is_idle(60)) is False


def test_wait_ready_uses_normalized_status_keys(base_config, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict[str, object]]] = []

    async def on_state(event) -> None:  # type: ignore[no-untyped-def]
        events.append((event.state.value, event.detail))

    pod = Pod("pod-1", "worker", base_config, hooks=EventHooks(on_state_change=on_state))
    responses = iter(
        [
            {"desired_status": "PROVISIONING", "ports": []},
            {
                "desired_status": "RUNNING",
                "ports": [{"privatePort": 22, "publicPort": 2201, "ip": "1.2.3.4"}],
            },
        ]
    )

    async def fake_status() -> dict[str, object]:
        return next(responses)

    async def fake_ssh_details() -> dict[str, object]:
        return {"ip": "1.2.3.4", "port": 2201, "password": "secret"}

    async def fake_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(pod, "status", fake_status)
    monkeypatch.setattr(pod, "_ensure_ssh_details", fake_ssh_details)
    monkeypatch.setattr("runpod_lifecycle.pod.asyncio.sleep", fake_sleep)

    result = asyncio.run(pod.wait_ready(timeout=1))

    assert result["desired_status"] == "RUNNING"
    assert "desiredStatus" not in result
    assert events[:3] == [
        (PodState.PROVISIONING.value, {"status": {"desired_status": "PROVISIONING", "ports": []}}),
        (PodState.STARTING.value, {"status": {"desired_status": "PROVISIONING", "ports": []}}),
        (
            PodState.READY.value,
            {
                "status": {
                    "desired_status": "RUNNING",
                    "ports": [{"privatePort": 22, "publicPort": 2201, "ip": "1.2.3.4"}],
                }
            },
        ),
    ]


def test_wait_ready_raises_launch_failure_on_failed_status(
    base_config, monkeypatch: pytest.MonkeyPatch
) -> None:
    pod = Pod("pod-1", "worker", base_config)

    async def fake_status() -> dict[str, object]:
        return {"desired_status": "FAILED", "ports": []}

    monkeypatch.setattr(pod, "status", fake_status)

    with pytest.raises(LaunchFailure):
        asyncio.run(pod.wait_ready(timeout=1))


# ---------------------------------------------------------------------------
# Pod composable surface: create_storage, list_storages, get_storage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_storage_find_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pod.get_storage returns the volume dict when a name match is found."""
    # Mock Pod.list_storages to return a known set
    mock_volumes = [
        {"id": "vol-abc", "name": "my-storage", "size": 50},
        {"id": "vol-def", "name": "other-storage", "size": 100},
    ]

    async def fake_list_storages() -> list[dict]:
        return mock_volumes

    monkeypatch.setattr("runpod_lifecycle.pod.Pod.list_storages", fake_list_storages)

    result = await Pod.get_storage("my-storage")
    assert result is not None
    assert result["id"] == "vol-abc"
    assert result["name"] == "my-storage"


@pytest.mark.asyncio
async def test_get_storage_find_by_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pod.get_storage returns the volume dict when an id match is found."""
    mock_volumes = [
        {"id": "vol-abc", "name": "my-storage", "size": 50},
    ]

    async def fake_list_storages() -> list[dict]:
        return mock_volumes

    monkeypatch.setattr("runpod_lifecycle.pod.Pod.list_storages", fake_list_storages)

    result = await Pod.get_storage("vol-abc")
    assert result is not None
    assert result["id"] == "vol-abc"


@pytest.mark.asyncio
async def test_get_storage_returns_none_when_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pod.get_storage returns None when no volume matches."""
    async def fake_list_storages() -> list[dict]:
        return []

    monkeypatch.setattr("runpod_lifecycle.pod.Pod.list_storages", fake_list_storages)

    result = await Pod.get_storage("nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_create_storage_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pod.create_storage delegates to api.create_network_volume."""
    called_with: list[tuple] = []

    async def fake_to_thread(fn, *args):
        called_with.append(args)
        return {"id": "vol-new", "name": args[1], "size": args[2]}

    monkeypatch.setattr("runpod_lifecycle.pod.asyncio.to_thread", fake_to_thread)
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key-override")

    result = await Pod.create_storage("new-volume", 200, "dc-west")

    assert result["id"] == "vol-new"
    assert len(called_with) == 1
    # args to api.create_network_volume: (api_key, name, size_gb, datacenter_id)
    assert called_with[0][1] == "new-volume"
    assert called_with[0][2] == 200
    assert called_with[0][3] == "dc-west"


def test_wait_ready_raises_timeout(base_config, monkeypatch: pytest.MonkeyPatch) -> None:
    pod = Pod("pod-1", "worker", base_config)

    async def fake_status() -> dict[str, object]:
        return {"desired_status": "PROVISIONING", "ports": []}

    async def fake_sleep(_seconds: float) -> None:
        return None

    state = {"calls": 0}

    def fake_monotonic() -> float:
        state["calls"] += 1
        return state["calls"] * 0.2

    monkeypatch.setattr(pod, "status", fake_status)
    monkeypatch.setattr("runpod_lifecycle.pod.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("runpod_lifecycle.pod.time.monotonic", fake_monotonic)

    with pytest.raises(NotReadyTimeout):
        asyncio.run(pod.wait_ready(timeout=0.1))
