"""Offline regressions for allocation ambiguity and age-filtered v2 inventory."""
import asyncio

import httpx
import pytest

from runpod_lifecycle import api, discovery
from runpod_lifecycle.errors import ProviderRequestRejected
from runpod_lifecycle.lifecycle import resume_when_available
from runpod_lifecycle.pod import Pod

CAPACITY = "There are no longer any instances available with the requested specifications."


@pytest.mark.parametrize("state", ["STARTING", "PROVISIONING"])
def test_resume_pending_pod_does_not_issue_another_start(monkeypatch, volumeless_config, state):
    pod = Pod("pending", "pending", volumeless_config)

    async def status():
        return {"desired_status": state}

    monkeypatch.setattr(pod, "status", status)
    monkeypatch.setattr("runpod_lifecycle.lifecycle.start_pod", lambda *args: pytest.fail("already starting"))
    assert asyncio.run(resume_when_available(pod)) is pod


@pytest.mark.parametrize("outcome", ["timeout", "missing-id", "invalid-json", "server-error", "invalid-request", "rate-limit", "id-with-error"])
def test_cpu_fallback_never_retries_an_unconfirmed_allocation(monkeypatch, outcome):
    calls = []

    def post(url, **kwargs):
        calls.append(kwargs["json"])
        request = httpx.Request("POST", url)
        if outcome == "timeout":
            raise httpx.ReadTimeout("allocation unknown", request=request)
        if outcome == "invalid-json":
            return httpx.Response(201, content=b"not-json", request=request)
        status, body = {
            "missing-id": (201, {}),
            "server-error": (500, {"error": "Internal server error"}),
            "invalid-request": (400, {"error": CAPACITY}),
            "rate-limit": (429, {"error": CAPACITY}),
            "id-with-error": (503, {"id": "possibly-created", "error": CAPACITY}),
        }[outcome]
        return httpx.Response(status, json=body, request=request)

    monkeypatch.setattr(api.httpx, "post", post)
    with pytest.raises((httpx.ReadTimeout, ValueError, RuntimeError, ProviderRequestRejected)):
        api.create_pod("fake", None, "image", compute_type="CPU", cpu_flavor_ids=("cpu3c", "cpu5c"))
    assert len(calls) == 1
    assert calls[0]["cpu"]["id"] == "cpu3c"


def _rest_inventory(monkeypatch, exact):
    monkeypatch.setattr(api, "_get_runpod", lambda: None)
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        request = httpx.Request("GET", url)
        if url == api.PODS_URL:
            pods = [{"id": pid, "name": "owned-" + pid, "status": "RUNNING", "runtime": None}
                    for pid in ("known", "young", "old")]
            return httpx.Response(200, json={"pods": pods, "pagination": {"hasNextPage": False}}, request=request)
        pod_id = url.rsplit("/", 1)[1]
        status, body = exact(pod_id)
        return httpx.Response(status, json=body, request=request)

    monkeypatch.setattr(api.httpx, "get", get)
    return calls


def test_age_filtered_orphans_fetch_exact_native_v2_runtime(monkeypatch):
    calls = _rest_inventory(monkeypatch, lambda pid: (200, {
        "id": pid, "status": "RUNNING", "runtime": {"uptime": 7200 if pid == "old" else 60}}))
    pods = asyncio.run(discovery.find_orphans("fake", {"known"}, name_prefix="owned-", older_than_seconds=3600))
    assert [pod.id for pod in pods] == ["old"]
    assert pods[0].uptime_seconds == 7200
    assert calls == [api.PODS_URL, api.PODS_URL + "/young", api.PODS_URL + "/old"]


@pytest.mark.parametrize("status,body", [
    (200, {"id": "young", "status": "STARTING", "runtime": None}),
    (200, {"id": "wrong", "runtime": {"uptime": 7200}}),
    (200, {"id": "young", "runtime": {"uptime": True}}),
    (200, {"id": "young", "runtime": {"uptime": -1}}),
    (403, {}),
    (404, {}),
    (503, {}),
])
def test_age_filtered_orphans_report_unavailable_evidence(monkeypatch, status, body):
    _rest_inventory(monkeypatch, lambda pid: (status, body))
    with pytest.raises(RuntimeError, match="age evidence unavailable"):
        asyncio.run(discovery.find_orphans("fake", {"known"}, older_than_seconds=3600))
