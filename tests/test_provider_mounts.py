"""CPU-only provider query → real SDK → lifecycle normalization → CLI evidence."""

import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from runpod_lifecycle import api, cli, discovery


NONCE = "1" * 32
ENTRY = {"volumeId": "vol-1", "volumeType": "NETWORK_VOLUME", "mountPath": "/workspace", "readOnly": False}


def raw(**extra):
    return {"id": "pod-1", "volumeMounts": [deepcopy(ENTRY)], **extra}


def test_real_sdk_through_cli_exact_fresh_projection(monkeypatch, capsys):
    import runpod

    calls = []

    def post(url, **kwargs):
        query = json.loads(kwargs["data"])["query"]
        calls.append(query)
        return SimpleNamespace(status_code=200, json=lambda: {"data": {"pod": raw()}})

    monkeypatch.setattr(api, "runpod", runpod)
    monkeypatch.setattr("runpod.api.graphql.requests.post", post)
    monkeypatch.setattr(cli, "_resolve_api_key", lambda _: "offline-test")
    for nonce in (NONCE, "2" * 32):
        assert cli.main(["mounts", "pod-1", "--nonce", nonce, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload == {
            "schema": "runpod.provider-mounts.v1", "pod_id": "pod-1", "nonce": nonce,
            "volume_mounts": [{"volume_id": "vol-1", "volume_type": "NETWORK_VOLUME",
                               "mount_path": "/workspace", "read_only": False}],
            "legacy_evidence": {"present": False, "valid": True, "conflict": False},
        }
    assert len(calls) == 2
    assert all('pod(input: {podId: "pod-1"})' in q for q in calls)
    assert all("volumeMounts" in q and "myself" not in q and "env" not in q for q in calls)


@pytest.mark.parametrize("payload", [None, [], {}, raw(id="other"), raw(volumeMounts=None),
    raw(volumeMounts=[]), raw(volumeMounts=[ENTRY] * 2), raw(volumeMounts=[None]),
    raw(volumeMounts=[{**ENTRY, "volumeId": 1}]), raw(volumeMounts=[{**ENTRY, "volumeId": ""}]),
    raw(volumeMounts=[{**ENTRY, "volumeType": None}]), raw(volumeMounts=[{**ENTRY, "volumeType": "bad"}]),
    raw(volumeMounts=[{**ENTRY, "volumeType": []}]),
    raw(volumeMounts=[{**ENTRY, "mountPath": "workspace"}]), raw(volumeMounts=[{**ENTRY, "mountPath": "/workspace/"}]),
    raw(volumeMounts=[{**ENTRY, "mountPath": "/workspace/../escape"}]),
    raw(volumeMounts=[{**ENTRY, "mountPath": "/workspace//models"}]),
    raw(volumeMounts=[{**ENTRY, "readOnly": 0}]), raw(volumeMounts=[{k: v for k, v in ENTRY.items() if k != "readOnly"}]),
    raw(volumeMounts=[ENTRY, {**ENTRY, "volumeId": "vol-2"}]),
    raw(volumeMounts=[ENTRY, {**ENTRY, "mountPath": "/other"}]),
    raw(networkVolumeId="other"), raw(networkVolumeId=[]), raw(networkVolume={}),
    raw(networkVolumeId="vol-1", networkVolume={"id": "other"}), raw(volumeMountPath="/other"),
    raw(objectStores=[{"objectStoreId": "bucket", "mountPath": "/workspace", "readOnly": False}]),
])
def test_strict_normalizer_refuses_missing_malformed_duplicate_conflicting(payload):
    with pytest.raises(ValueError):
        discovery.normalize_pod_mounts(payload, "pod-1", NONCE)


def test_normalization_discards_raw_provider_and_secret_fields():
    source = raw(networkVolumeId="vol-1", networkVolume={"id": "vol-1", "secret": "sentinel"},
                 volumeMountPath="/workspace", objectStores=[], env={"SECRET": "sentinel"})
    source["volumeMounts"][0]["unexpected"] = "sentinel"
    result = discovery.normalize_pod_mounts(source, "pod-1", NONCE)
    assert result["legacy_evidence"] == {"present": True, "valid": True, "conflict": False}
    assert "sentinel" not in json.dumps(result)


def test_all_null_legacy_fields_are_absent_not_mount_authority():
    result = discovery.normalize_pod_mounts(raw(networkVolumeId=None, networkVolume=None,
        objectStores=None, volumeMountPath=None), "pod-1", NONCE)
    assert result["legacy_evidence"] == {"present": False, "valid": True, "conflict": False}


@pytest.mark.parametrize("nonce", [None, "", "x" * 32, "a" * 31, "A" * 32, [NONCE]])
def test_invalid_nonce_refuses_before_provider(monkeypatch, nonce):
    def unexpected(*a, **kw):
        raise AssertionError("provider called")
    monkeypatch.setattr(api, "get_pod_mounts", unexpected)
    with pytest.raises(ValueError, match="nonce"):
        asyncio.run(discovery.observe_pod_mounts("pod-1", "offline-test", nonce))


def test_missing_typed_sdk_field_has_no_legacy_fallback(monkeypatch, capsys):
    monkeypatch.setattr(api, "runpod", SimpleNamespace(get_pod=lambda *_: raw(networkVolumeId="vol-1")))
    monkeypatch.setattr(cli, "_resolve_api_key", lambda _: "offline-test")
    assert cli.main(["mounts", "pod-1", "--nonce", NONCE, "--json"]) == 1
    result = capsys.readouterr()
    assert not result.out
    assert result.err == "error: Typed provider mount observation failed\n"


def test_provider_failures_are_bounded_and_do_not_log_raw_details(monkeypatch, capsys, caplog):
    def fail(*args, **kwargs):
        print("secret-stdout")
        raise RuntimeError("raw-provider-secret")
    monkeypatch.setattr(api, "runpod", SimpleNamespace(get_pod_mounts=fail))
    monkeypatch.setattr(cli, "_resolve_api_key", lambda _: "offline-test")
    assert cli.main(["mounts", "pod-1", "--nonce", NONCE, "--json"]) == 1
    result = capsys.readouterr()
    assert not result.out
    assert result.err == "error: Typed provider mount observation failed\n"
    assert "raw-provider-secret" not in caplog.text


@pytest.mark.parametrize("legacy,present,valid,conflict", [
    ({}, False, True, False), ({"networkVolumeId": None, "networkVolume": None}, False, True, False),
    ({"networkVolumeId": "vol-1"}, True, True, False), ({"networkVolume": {"id": "vol-1"}}, True, True, False),
    ({"networkVolumeId": "vol-1", "networkVolume": {"id": "vol-2"}}, True, True, True),
    ({"networkVolumeId": []}, True, False, False), ({"networkVolume": {}}, True, False, False),
    ({"networkVolume": {"id": None}}, True, False, False),
])
def test_list_preserves_independent_legacy_verdict(legacy, present, valid, conflict):
    result = discovery._to_summary({"id": "pod-1", **legacy})
    assert result.network_volume_evidence == {"present": present, "valid": valid, "conflict": conflict}


def test_real_list_cli_preserves_conflict_verdict(monkeypatch, runpod_sdk_mock, capsys):
    runpod_sdk_mock.get_pods.return_value = [{"id": "pod-1", "networkVolumeId": "vol-1", "networkVolume": {"id": "other"}}]
    monkeypatch.setattr(cli, "_resolve_api_key", lambda _: "offline-test")
    assert cli.main(["list", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)[0]
    assert result["network_volume_id"] is None
    assert result["network_volume_evidence"] == {"present": True, "valid": True, "conflict": True}


@pytest.mark.parametrize("path", ["/workspace", "/other"])
def test_real_list_cli_preserves_legacy_mount_path(monkeypatch, runpod_sdk_mock, capsys, path):
    runpod_sdk_mock.get_pods.return_value = [{"id": "pod-1", "volumeMountPath": path, "networkVolume": None}]
    monkeypatch.setattr(cli, "_resolve_api_key", lambda _: "offline-test")
    assert cli.main(["list", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)[0]
    assert result["network_volume_mount_path"] == path
    assert result["network_volume_evidence"] == {"present": True, "valid": True, "conflict": False}


@pytest.mark.parametrize("path", ["relative", "/workspace/../escape", "/workspace/", "/work\nspace", 1, [], "x" * 513])
def test_list_marks_malformed_legacy_path_invalid(path):
    result = discovery._to_summary({"id": "pod-1", "volumeMountPath": path})
    assert result.network_volume_mount_path is None
    assert result.network_volume_evidence == {"present": True, "valid": False, "conflict": False}


def test_list_projects_only_safe_legacy_object_mount_facts():
    obj = {"objectStoreId": "bucket-1", "mountPath": "/objects", "readOnly": False,
           "secretAccessKey": "never-project", "accessKeyId": "never-project", "bucketName": "never-project"}
    result = discovery._to_summary({"id": "pod-1", "objectStores": [obj]})
    assert result.legacy_object_mounts == [{"volume_id": "bucket-1", "mount_path": "/objects", "read_only": False}]
    assert result.network_volume_evidence == {"present": True, "valid": True, "conflict": False}
    assert "never-project" not in json.dumps(result.__dict__)


@pytest.mark.parametrize("objects", [None, []])
def test_list_preserves_absent_or_empty_legacy_object_mounts(objects):
    result = discovery._to_summary({"id": "pod-1", "objectStores": objects, "volumeMountPath": None})
    assert result.legacy_object_mounts == objects
    assert result.network_volume_evidence == {"present": objects is not None, "valid": True, "conflict": False}


@pytest.mark.parametrize("objects", [{}, [None], [{"objectStoreId": "b", "mountPath": "/objects", "readOnly": 0}],
    [{"objectStoreId": [], "mountPath": "/objects", "readOnly": False}],
    [{"objectStoreId": "b", "mountPath": "relative", "readOnly": False}],
    [{"objectStoreId": "b", "mountPath": "/objects", "readOnly": False}] * 2])
def test_list_marks_malformed_duplicate_object_mounts_invalid(objects):
    result = discovery._to_summary({"id": "pod-1", "objectStores": objects})
    assert result.legacy_object_mounts is None
    assert result.network_volume_evidence == {"present": True, "valid": False, "conflict": False}
