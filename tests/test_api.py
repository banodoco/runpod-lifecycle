from __future__ import annotations

from types import SimpleNamespace

import pytest

from runpod_lifecycle import api


class FakeResponse:
    def __init__(self, status_code: int, payload: dict, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or str(payload)

    def json(self) -> dict:
        return self._payload


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


def test_get_pod_status_returns_none_when_sdk_and_graphql_return_no_pod(
    monkeypatch: pytest.MonkeyPatch,
    runpod_sdk_mock,
) -> None:
    runpod_sdk_mock.get_pod.return_value = None
    monkeypatch.setattr(
        "runpod_lifecycle.api.httpx",
        SimpleNamespace(post=lambda *args, **kwargs: FakeResponse(200, {"data": {"pod": None}})),
    )
    assert api.get_pod_status("p1", "test") is None


def test_get_pod_status_falls_back_to_graphql_when_sdk_fails(
    monkeypatch: pytest.MonkeyPatch,
    runpod_sdk_mock,
) -> None:
    runpod_sdk_mock.get_pod.side_effect = RuntimeError("sdk unavailable")
    monkeypatch.setattr(
        "runpod_lifecycle.api.httpx",
        SimpleNamespace(
            post=lambda *args, **kwargs: FakeResponse(
                200,
                {
                    "data": {
                        "pod": {
                            "id": "p1",
                            "desiredStatus": "RUNNING",
                            "actualStatus": "RUNNING",
                            "runtime": {
                                "ip": "1.2.3.4",
                                "ports": [
                                    {"privatePort": 22, "publicPort": 2201, "ip": "1.2.3.4"},
                                    {"privatePort": 8888, "publicPort": 88881, "ip": "1.2.3.4"},
                                ],
                                "uptimeInSeconds": 42,
                            },
                        }
                    }
                },
            )
        ),
    )

    status = api.get_pod_status("p1", "test")

    assert status is not None
    assert status["desired_status"] == "RUNNING"
    assert status["actual_status"] == "RUNNING"
    assert status["ip"] == "1.2.3.4"
    assert status["ports"] == [
        {"privatePort": 22, "publicPort": 2201, "ip": "1.2.3.4"},
        {"privatePort": 8888, "publicPort": 88881, "ip": "1.2.3.4"},
    ]
    assert status["uptime_seconds"] == 42


def test_get_pod_status_falls_back_to_graphql_when_sdk_returns_none(
    monkeypatch: pytest.MonkeyPatch,
    runpod_sdk_mock,
) -> None:
    runpod_sdk_mock.get_pod.return_value = None
    monkeypatch.setattr(
        "runpod_lifecycle.api.httpx",
        SimpleNamespace(
            post=lambda *args, **kwargs: FakeResponse(
                200,
                {
                    "data": {
                        "pod": {
                            "id": "p1",
                            "desiredStatus": "PROVISIONING",
                            "actualStatus": None,
                            "runtime": None,
                        }
                    }
                },
            )
        ),
    )

    status = api.get_pod_status("p1", "test")

    assert status is not None
    assert status["desired_status"] == "PROVISIONING"
    assert status["actual_status"] is None
    assert status["ports"] == []


def test_get_pod_status_retries_with_minimal_graphql_query_on_schema_error(
    monkeypatch: pytest.MonkeyPatch,
    runpod_sdk_mock,
) -> None:
    responses = iter(
        [
            FakeResponse(200, {"errors": [{"message": "Cannot query field sshPassword"}]}),
            FakeResponse(
                200,
                {
                    "data": {
                        "pod": {
                            "id": "p1",
                            "desiredStatus": "RUNNING",
                            "actualStatus": "RUNNING",
                            "runtime": {
                                "ip": "1.2.3.4",
                                "ports": [{"privatePort": 22, "publicPort": 2201, "ip": "1.2.3.4"}],
                            },
                        }
                    }
                },
            ),
        ]
    )
    runpod_sdk_mock.get_pod.side_effect = RuntimeError("sdk unavailable")
    monkeypatch.setattr(
        "runpod_lifecycle.api.httpx",
        SimpleNamespace(post=lambda *args, **kwargs: next(responses)),
    )

    status = api.get_pod_status("p1", "test")

    assert status is not None
    assert status["desired_status"] == "RUNNING"
    assert status["ports"] == [{"privatePort": 22, "publicPort": 2201, "ip": "1.2.3.4"}]


def test_get_pod_status_retries_with_minimal_graphql_query_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
    runpod_sdk_mock,
) -> None:
    responses = iter(
        [
            FakeResponse(400, {"errors": [{"message": "Cannot query field sshPassword"}]}),
            FakeResponse(
                200,
                {
                    "data": {
                        "pod": {
                            "id": "p1",
                            "desiredStatus": "RUNNING",
                            "actualStatus": "RUNNING",
                            "runtime": {
                                "ip": "1.2.3.4",
                                "ports": [{"privatePort": 22, "publicPort": 2201, "ip": "1.2.3.4"}],
                            },
                        }
                    }
                },
            ),
        ]
    )
    runpod_sdk_mock.get_pod.side_effect = RuntimeError("sdk unavailable")
    monkeypatch.setattr(
        "runpod_lifecycle.api.httpx",
        SimpleNamespace(post=lambda *args, **kwargs: next(responses)),
    )

    status = api.get_pod_status("p1", "test")

    assert status is not None
    assert status["desired_status"] == "RUNNING"
    assert status["ports"] == [{"privatePort": 22, "publicPort": 2201, "ip": "1.2.3.4"}]


def test_get_pod_status_graphql_uses_schema_safe_fields_and_derives_ip(
    monkeypatch: pytest.MonkeyPatch,
    runpod_sdk_mock,
) -> None:
    calls: list[dict] = []

    def _post(*_args, **kwargs):
        calls.append(kwargs["json"])
        return FakeResponse(
            200,
            {
                "data": {
                    "pod": {
                        "id": "p1",
                        "desiredStatus": "RUNNING",
                        "runtime": {
                            "ports": [{"privatePort": 22, "publicPort": 2201, "ip": "1.2.3.4"}],
                        },
                    }
                }
            },
        )

    runpod_sdk_mock.get_pod.side_effect = RuntimeError("sdk unavailable")
    monkeypatch.setattr("runpod_lifecycle.api.httpx", SimpleNamespace(post=_post))

    status = api.get_pod_status("p1", "test")

    assert status is not None
    assert status["ip"] == "1.2.3.4"
    assert "actualStatus" not in calls[0]["query"]
    assert "runtime {\n              ip" not in calls[0]["query"]


# ---------------------------------------------------------------------------
# create_network_volume
# ---------------------------------------------------------------------------


def test_create_network_volume_posts_correct_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_network_volume POSTs the correct payload to NETWORK_VOLUMES_URL."""
    post_calls: list[dict] = []

    def fake_post(url, *, json, headers, timeout):
        post_calls.append({"url": url, "json": json})
        # Return a valid volume dict
        return FakeResponse(201, {
            "id": "vol-created-1",
            "name": "my-volume",
            "size": 100,
            "dataCenterId": "dc-test",
        })

    monkeypatch.setattr("runpod_lifecycle.api.httpx", SimpleNamespace(post=fake_post))

    result = api.create_network_volume("fake-key", "my-volume", 100, "dc-test")

    assert result["id"] == "vol-created-1"
    assert result["name"] == "my-volume"
    assert len(post_calls) == 1
    assert post_calls[0]["json"] == {
        "name": "my-volume",
        "size": 100,
        "dataCenterId": "dc-test",
    }
    assert "rest.runpod.io/v1/networkvolumes" in post_calls[0]["url"]


def test_create_network_volume_raises_on_non_200_201(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_network_volume raises RuntimeError on non-200/201 response."""
    monkeypatch.setattr(
        "runpod_lifecycle.api.httpx",
        SimpleNamespace(post=lambda *args, **kwargs: FakeResponse(400, {"error": "bad request"})),
    )

    with pytest.raises(RuntimeError, match="Failed to create network volume"):
        api.create_network_volume("fake-key", "bad-vol", 100, "dc-test")


def test_create_pod_uses_prior_default_ports_when_none(runpod_sdk_mock) -> None:
    runpod_sdk_mock.create_pod.return_value = {"id": "pod-default-ports"}
    api.create_pod(
        api_key="api-key",
        gpu_type_id="gpu-1",
        image_name="image",
    )
    kwargs = runpod_sdk_mock.create_pod.call_args.kwargs
    assert kwargs["ports"] == "22/tcp,8888/http"


def test_create_pod_allows_cloud_type_override(runpod_sdk_mock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_CLOUD_TYPE", "community")
    runpod_sdk_mock.create_pod.return_value = {"id": "pod-community"}

    api.create_pod(
        api_key="api-key",
        gpu_type_id="gpu-1",
        image_name="image",
    )

    kwargs = runpod_sdk_mock.create_pod.call_args.kwargs
    assert kwargs["cloud_type"] == "COMMUNITY"


def test_create_pod_passes_custom_ports_through(runpod_sdk_mock) -> None:
    runpod_sdk_mock.create_pod.return_value = {"id": "pod-custom-ports"}
    api.create_pod(
        api_key="api-key",
        gpu_type_id="gpu-1",
        image_name="image",
        ports="8675/http,22/tcp",
    )
    kwargs = runpod_sdk_mock.create_pod.call_args.kwargs
    assert kwargs["ports"] == "8675/http,22/tcp"
    assert "8675/http" in kwargs["ports"]


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
