from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from runpod_lifecycle.api import get_pod_ssh_details


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
