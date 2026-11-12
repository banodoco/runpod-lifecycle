"""Offline checks of v2 projections and complete inventory traversal."""
from copy import deepcopy

import pytest

from runpod_lifecycle.rest import collect_pages, normalize_pod, normalize_volume


def test_gpu_projection_preserves_record_and_adds_compatibility_fields():
    raw = {
        "id": "pod", "status": "STARTING", "image": "image:tag", "cost": .4,
        "disk": 50, "gpu": {"id": "NVIDIA GeForce RTX 4090", "count": 2},
        "mounts": {"network": [{"volumeId": "volume", "path": "/workspace"}]},
        "runtime": {"uptime": 42, "ports": [
            {"private": 22, "public": 2200, "ip": "203.0.113.1", "type": "tcp"}]},
    }
    original = deepcopy(raw)
    pod = normalize_pod(raw)
    for field in ("id", "status", "image", "cost", "disk", "gpu", "mounts"):
        assert pod[field] == raw[field]
    assert {key: pod[key] for key in (
        "desiredStatus", "imageName", "costPerHr", "containerDiskInGb",
        "machineType", "gpuCount", "computeType", "networkVolume", "volumeMountPath"
    )} == {
        "desiredStatus": "STARTING", "imageName": "image:tag", "costPerHr": .4,
        "containerDiskInGb": 50, "machineType": "NVIDIA GeForce RTX 4090",
        "gpuCount": 2, "computeType": "GPU", "networkVolume": {"id": "volume"},
        "volumeMountPath": "/workspace",
    }
    assert pod["runtime"] == {"uptime": 42, "uptimeInSeconds": 42, "ports": [
        {"privatePort": 22, "publicPort": 2200, "ip": "203.0.113.1", "type": "tcp"}]}
    assert raw == original


def test_cpu_and_null_live_fields_survive_projection():
    # List responses and stopped/provisioning single GETs have no live metrics.
    raw = {"cpu": {"id": "cpu5c", "vcpuCount": 4}, "runtime": None,
           "ssh": {"proxy": None, "direct": None}, "mounts": {}}
    pod = normalize_pod(raw)
    assert pod["cpu"] == raw["cpu"]
    assert pod["computeType"] == "CPU" and pod["gpuCount"] == 0
    assert pod["runtime"] is None and pod["ssh"] == raw["ssh"]
    assert pod["networkVolume"] is None and pod["volumeMountPath"] is None


def test_volume_projection_preserves_raw_and_rejects_conflicting_location():
    raw = {"id": "volume", "dataCenter": "US-TX-3", "size": 100}
    assert normalize_volume(raw) == {**raw, "dataCenterId": "US-TX-3"}
    assert "dataCenterId" not in raw
    with pytest.raises(ValueError, match="Conflicting"):
        normalize_volume({**raw, "dataCenterId": "EU-RO-1"})


@pytest.mark.parametrize("raw", [
    {"status": "RUNNING", "desiredStatus": "EXITED"},
    {"image": "new", "imageName": "old"},
    {"cost": .4, "costPerHr": .8},
    {"disk": 50, "containerDiskInGb": 20},
    {"gpu": {"id": "gpu", "count": 2}, "gpuCount": 1},
    {"gpu": {"id": "gpu"}, "machineType": "other"},
    {"cpu": {"id": "cpu"}, "gpuCount": 1},
    {"mounts": {"network": []}, "networkVolume": {"id": "old"}},
    {"mounts": {"network": [{"volumeId": "new", "path": "/data"}]},
     "networkVolumeId": "old"},
    {"runtime": {"uptime": 1, "uptimeInSeconds": 2}},
    {"runtime": {"ports": [{"private": 22, "privatePort": 80}]}},
])
def test_conflicting_compatibility_aliases_fail(raw):
    with pytest.raises(ValueError, match="Conflicting"):
        normalize_pod(raw)


@pytest.mark.parametrize("mounts", [None, [], {"network": None}, {"network": {}},
    {"network": [None]}, {"network": [{"volumeId": "v"}]},
    {"network": [{"volumeId": "", "path": "/data"}]},
    {"network": [{"volumeId": "v", "path": "relative"}]}])
def test_malformed_mounts_fail(mounts):
    with pytest.raises(ValueError, match="Malformed provider"):
        normalize_pod({"mounts": mounts})


def test_multiple_mounts_do_not_invent_single_volume_identity():
    mounts = {"network": [{"volumeId": "a", "path": "/a"},
                          {"volumeId": "b", "path": "/b"}]}
    pod = normalize_pod({"mounts": mounts})
    assert pod["mounts"] == mounts
    assert "networkVolume" not in pod and "networkVolumeId" not in pod
    assert "volumeMountPath" not in pod


@pytest.mark.parametrize("collection", ["pods", "networkVolumes"])
def test_collection_traversal_passes_opaque_cursor_verbatim(collection):
    cursor = "opaque+/= cursor"
    calls = []
    pages = iter([
        {collection: [{"id": "first"}], "pagination": {
            "hasNextPage": True, "nextCursor": cursor}},
        {collection: [{"id": "last"}], "pagination": {
            "hasNextPage": False, "nextCursor": None}},
    ])
    def fetch(params):
        calls.append(params)
        return next(pages)
    assert collect_pages(fetch, collection) == [{"id": "first"}, {"id": "last"}]
    assert calls == [{}, {"cursor": cursor}]


@pytest.mark.parametrize("body", [
    [], {"pods": {}}, {"pods": [None]}, {"pods": []},
    {"pods": [], "pagination": []},
    {"pods": [], "pagination": {"hasNextPage": 1}},
    {"pods": [], "pagination": {"hasNextPage": True}},
    {"pods": [], "pagination": {"hasNextPage": True, "nextCursor": ""}},
    {"pods": [], "pagination": {"hasNextPage": True, "nextCursor": None}},
])
def test_malformed_or_incomplete_inventory_fails(body):
    with pytest.raises(ValueError):
        collect_pages(lambda _: body, "pods")


def test_repeated_cursor_fails_instead_of_returning_partial_inventory():
    calls = []
    def fetch(params):
        calls.append(params)
        return {"pods": [{"id": "pod"}], "pagination": {
            "hasNextPage": True, "nextCursor": "same"}}
    with pytest.raises(ValueError, match="repeated"):
        collect_pages(fetch, "pods")
    assert calls == [{}, {"cursor": "same"}]


def test_api_walks_pod_and_volume_pages_without_v1_expansion(monkeypatch):
    from types import SimpleNamespace
    from runpod_lifecycle import api
    monkeypatch.setattr(api, "_get_runpod", lambda: None)
    calls = []
    def get(url, **kwargs):
        calls.append((url, kwargs))
        is_pods = url == api.PODS_URL
        collection = "pods" if is_pods else "networkVolumes"
        following = bool(kwargs["params"])
        item = ({"id": "p2" if following else "p1", "status": "RUNNING", "image": "image",
                 "mounts": {"network": [{"volumeId": "v", "path": "/workspace"}]}}
                if is_pods else {"id": "v", "dataCenter": "EU-RO-1"})
        body = {collection: [item], "pagination": {
            "hasNextPage": not following, "nextCursor": None if following else "opaque/+"}}
        return SimpleNamespace(status_code=200, json=lambda: body)
    monkeypatch.setattr(api.httpx, "get", get)
    pods = api.list_pods("fake")
    assert [p["id"] for p in pods] == ["p1", "p2"]
    assert all(p["networkVolume"] == {"id": "v"} for p in pods)
    volumes = api.get_network_volumes("fake")
    assert len(volumes) == 2 and all(v["dataCenterId"] == "EU-RO-1" for v in volumes)
    assert [kw["params"] for _, kw in calls] == [{}, {"cursor": "opaque/+"}] * 2
    assert all(url.startswith("https://api.runpod.io/v2/") for url, _ in calls)


def test_status_uses_v2_runtime_ports_and_cost():
    from runpod_lifecycle import api
    status = api._normalize_pod_status("p", {
        "status": "RUNNING", "cost": .4,
        "runtime": {"uptime": 42, "ports": [
            {"private": 22, "public": 2200, "ip": "203.0.113.1", "type": "tcp"}]}})
    assert status["desired_status"] == "RUNNING" and status["cost_per_hr"] == .4
    assert status["uptime_seconds"] == 42 and status["ip"] == "203.0.113.1"
    assert status["ports"] == [{"privatePort": 22, "publicPort": 2200,
                                "ip": "203.0.113.1", "type": "tcp"}]


def test_volume_inventory_requires_pagination_metadata():
    with pytest.raises(ValueError, match="pagination"):
        collect_pages(lambda _: {"networkVolumes": []}, "networkVolumes")


def test_multiple_mounts_reject_ambiguous_legacy_aliases():
    with pytest.raises(ValueError, match="Conflicting"):
        normalize_pod({"networkVolumeId": "a", "mounts": {"network": [
            {"volumeId": "a", "path": "/a"}, {"volumeId": "b", "path": "/b"}]}})


def test_start_action_accepts_v2_no_content_response(monkeypatch):
    from types import SimpleNamespace
    from runpod_lifecycle import api
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=204)
    monkeypatch.setattr(api.httpx, "post", post)
    assert api.start_pod("pod", "fake") == {}
    assert calls[0][0] == "https://api.runpod.io/v2/pods/pod/action"
    assert calls[0][1]["json"] == {"action": "start"}


def test_smoke_exact_read_preserves_native_v2_compute_rate():
    import httpx
    from runpod_lifecycle import transport_smoke
    provider = transport_smoke.Provider("fake")
    provider.http.close()
    def handle(request):
        assert request.method == "GET" and str(request.url) == "https://api.runpod.io/v2/pods/pod"
        return httpx.Response(200, json={"id": "pod", "status": "RUNNING", "cost": .4,
                                         "image": "image", "mounts": {}, "runtime": None})
    provider.http = httpx.Client(transport=httpx.MockTransport(handle))
    try:
        status, observed = provider.inspect("pod")
    finally:
        provider.close()
    assert status == 200 and observed["costPerHr"] == .4
    assert observed["desiredStatus"] == "RUNNING"
