from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from runpod_lifecycle import api, graphql
from runpod_lifecycle.config import RunPodConfig
from runpod_lifecycle.lifecycle import launch_when_available


BOUND = "2099-01-02T03:04:05Z"


def test_launch_forwards_deadline_into_exact_provider_mutation(monkeypatch):
    provider = MagicMock()
    monkeypatch.setattr(api, "_get_runpod", lambda: provider)
    run = MagicMock(return_value={"data": {"podFindAndDeployOnDemand": {"id": "pod-1"}}})
    monkeypatch.setattr(graphql, "_run", run)
    for name in ("find_gpu_type", "get_storage_volume_id", "get_network_volumes", "check_and_expand_storage"):
        monkeypatch.setattr("runpod_lifecycle.lifecycle." + name, lambda *a, **k: pytest.fail("unexpected provider call"))
    config = RunPodConfig(api_key="test", gpu_type_id="gpu-1", data_center_id="dc-1",
                          worker_image="image", template_id="", ram_tiers_enabled=False,
                          terminate_after=BOUND)
    pod = asyncio.run(launch_when_available(config, name="claim", max_wait_sec=0))
    assert pod.id == "pod-1"
    assert run.call_count == 1
    assert run.call_args.args == ('''
    mutation {
      podFindAndDeployOnDemand(
        input: {
          name: "claim", imageName: "image", cloudType: SECURE, startSsh: true, gpuTypeId: "gpu-1", terminateAfter: "2099-01-02T03:04:05Z", supportPublicIp: true, gpuCount: 1, volumeInGb: 200, minVcpuCount: 8, minMemoryInGb: 32, dataCenterId: "dc-1", containerDiskInGb: 200, ports: "22/tcp,8888/http", volumeMountPath: "/workspace"
        }
      ) {
        id
        imageName
        env
        machineId
        machine {
          podHostId
        }
      }
    }
    ''', "test")


@pytest.mark.parametrize("value", ["", "2099-01-02T03:04:05", "2099-01-02T03:04:05+01:00",
                                    "2000-01-01T00:00:00Z", "2099-02-30T00:00:00Z",
                                    '2099-01-02T03:04:05Z" }', 123])
def test_invalid_bounds_refused_before_provider_calls(value, monkeypatch):
    provider = MagicMock()
    monkeypatch.setattr(api, "_get_runpod", provider)
    runner = MagicMock()
    monkeypatch.setattr(graphql, "_run", runner)
    with pytest.raises(ValueError, match="terminate_after"):
        RunPodConfig(api_key="test", terminate_after=value)
    with pytest.raises(ValueError, match="terminate_after"):
        api.create_pod("test", "gpu", "image", terminate_after=value)
    with pytest.raises(ValueError, match="terminate_after"):
        graphql.create_pod("claim", "image", gpu_type_id="gpu", network_volume_id="vol",
                           terminate_after=value)
    provider.assert_not_called()
    runner.assert_not_called()


def test_expired_config_refused_before_launch_provider_calls(monkeypatch):
    config = RunPodConfig(api_key="test", terminate_after=BOUND)
    config.terminate_after = "2000-01-01T00:00:00Z"
    create = MagicMock()
    monkeypatch.setattr("runpod_lifecycle.lifecycle.find_gpu_type", create)
    with pytest.raises(ValueError, match="future"):
        asyncio.run(launch_when_available(config, max_wait_sec=0))
    create.assert_not_called()


def test_bound_is_optional_and_cpu_rejects_it_before_provider(monkeypatch):
    assert RunPodConfig(api_key="test").terminate_after is None
    provider = MagicMock()
    monkeypatch.setattr(api, "_get_runpod", provider)
    with pytest.raises(ValueError, match="GPU"):
        api.create_pod("test", None, "image", compute_type="CPU", terminate_after=BOUND)
    provider.assert_not_called()
