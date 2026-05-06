from __future__ import annotations

from runpod_lifecycle import api


def test_create_pod_suppresses_sdk_stdout_with_env_values(runpod_sdk_mock, capsys) -> None:
    runpod_sdk_mock.create_pod.side_effect = lambda **_kwargs: print(
        "raw_response: {'env': ['SUPABASE_SERVICE_ROLE_KEY=secret']}"
    ) or {"id": "pod-1"}

    pod = api.create_pod(
        api_key="api-key",
        gpu_type_id="gpu-1",
        image_name="image",
        env_vars={"SUPABASE_SERVICE_ROLE_KEY": "secret"},
    )

    assert pod["id"] == "pod-1"
    assert "secret" not in capsys.readouterr().out


def test_get_pod_status_handles_explicit_none_runtime(runpod_sdk_mock) -> None:
    """Regression: SDK returns runtime: None for pods that haven't booted; .get('runtime', {}) returned None."""
    runpod_sdk_mock.get_pod.return_value = {
        "id": "p1",
        "desiredStatus": "RUNNING",
        "actualStatus": None,
        "runtime": None,
        "costPerHr": 0.5,
    }
    status = api.get_pod_status("p1", "test")
    assert status is not None
    assert status["desired_status"] == "RUNNING"
    assert status["ip"] is None
    assert status["ports"] == []
    assert status["uptime_seconds"] == 0


def test_get_pod_status_handles_missing_runtime_key(runpod_sdk_mock) -> None:
    runpod_sdk_mock.get_pod.return_value = {
        "id": "p1",
        "desiredStatus": "PROVISIONING",
        "actualStatus": "PROVISIONING",
    }
    status = api.get_pod_status("p1", "test")
    assert status is not None
    assert status["ports"] == []
    assert status["ip"] is None


def test_get_pod_status_returns_none_when_sdk_returns_falsy(runpod_sdk_mock) -> None:
    runpod_sdk_mock.get_pod.return_value = None
    assert api.get_pod_status("p1", "test") is None


def test_get_pod_status_normalizes_keys(runpod_sdk_mock) -> None:
    runpod_sdk_mock.get_pod.return_value = {
        "id": "p1",
        "desiredStatus": "RUNNING",
        "actualStatus": "RUNNING",
        "runtime": {
            "ip": "1.2.3.4",
            "ports": [{"privatePort": 22, "publicPort": 12345}],
            "sshPassword": "secret",
            "uptimeInSeconds": 60,
        },
        "createdAt": "2026-04-01T00:00:00Z",
        "lastStatusChange": "2026-04-01T00:01:00Z",
        "costPerHr": 0.69,
    }
    status = api.get_pod_status("p1", "test")
    assert status == {
        "runpod_id": "p1",
        "desired_status": "RUNNING",
        "actual_status": "RUNNING",
        "ip": "1.2.3.4",
        "ports": [{"privatePort": 22, "publicPort": 12345}],
        "ssh_password": "secret",
        "created_at": "2026-04-01T00:00:00Z",
        "last_status_change": "2026-04-01T00:01:00Z",
        "uptime_seconds": 60,
        "cost_per_hr": 0.69,
    }
