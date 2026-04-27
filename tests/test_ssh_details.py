from __future__ import annotations

import logging
from unittest.mock import MagicMock
from types import SimpleNamespace

import pytest

from runpod_lifecycle.api import get_pod_ssh_details
from runpod_lifecycle.config import RunPodConfig
from runpod_lifecycle.pod import Pod


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def test_get_pod_ssh_details_uses_sdk_path_without_http(
    monkeypatch: pytest.MonkeyPatch,
    runpod_sdk_mock,
) -> None:
    runpod_sdk_mock.get_pod.return_value = {
        "runtime": {
            "sshPassword": "secret",
            "ports": [
                {"privatePort": 22, "publicPort": 2201, "ip": "1.2.3.4"},
            ],
        }
    }
    post_mock = SimpleNamespace(post=pytest.fail)
    monkeypatch.setattr("runpod_lifecycle.api.httpx", post_mock)

    details = get_pod_ssh_details("pod-123", "api-key")

    assert details == {"ip": "1.2.3.4", "port": 2201, "password": "secret"}


def test_get_pod_ssh_details_falls_back_to_graphql(
    monkeypatch: pytest.MonkeyPatch,
    runpod_sdk_mock,
) -> None:
    runpod_sdk_mock.get_pod.return_value = {"runtime": {"ports": []}}
    monkeypatch.setattr(
        "runpod_lifecycle.api.httpx",
        SimpleNamespace(
            post=lambda *args, **kwargs: FakeResponse(
                200,
                {
                    "data": {
                        "pod": {
                            "runtime": {
                                "ports": [
                                    {"privatePort": 22, "publicPort": 2202, "ip": "5.6.7.8"},
                                ]
                            }
                        }
                    }
                },
            )
        ),
    )

    details = get_pod_ssh_details("pod-123", "api-key")

    assert details == {"ip": "5.6.7.8", "port": 2202, "password": "runpod"}


def test_get_pod_ssh_details_returns_none_and_logs_warning(
    monkeypatch: pytest.MonkeyPatch,
    runpod_sdk_mock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="runpod_lifecycle.api")
    runpod_sdk_mock.get_pod.return_value = {"runtime": {"ports": []}}
    monkeypatch.setattr(
        "runpod_lifecycle.api.httpx",
        SimpleNamespace(post=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))),
    )

    details = get_pod_ssh_details("pod-123", "api-key")

    assert details is None
    assert "Could not get SSH details for pod pod-123" in caplog.text


def test_open_ssh_client_returns_connected_raw_client(monkeypatch: pytest.MonkeyPatch) -> None:
    config = RunPodConfig(api_key="api-key")
    pod = Pod("pod-123", "worker", config)
    raw_client = MagicMock()
    wrapper = SimpleNamespace(client=raw_client, connect=MagicMock())

    monkeypatch.setattr(
        "runpod_lifecycle.pod.api.get_pod_ssh_details",
        lambda pod_id, api_key: {"ip": "1.2.3.4", "port": 2201, "password": "secret"},
    )
    monkeypatch.setattr(pod, "_build_ssh_client", lambda ssh_details: wrapper)

    result = pod.open_ssh_client()

    assert result is raw_client
    wrapper.connect.assert_called_once_with()
