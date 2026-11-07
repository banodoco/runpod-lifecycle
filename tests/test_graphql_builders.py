"""Lifecycle-owned GraphQL builders (transport is the public SDK's runner)."""

import json
import re
from unittest.mock import patch

import pytest

from runpod_lifecycle import graphql
from runpod_lifecycle.graphql import generate_pod_deployment_mutation, generate_pod_mounts_query


def test_docker_args_are_a_single_escaped_graphql_string():
    command = "bash -lc \"exec python3 -c 'print(\\\"hello\\\")'\"\n"
    query = generate_pod_deployment_mutation(
        name="test", image_name="runpod/base:latest", gpu_type_id="gpu", docker_args=command
    )
    encoded = re.search(r'dockerArgs: ("(?:[^"\\]|\\.)*")', query)[1]
    assert json.loads(encoded) == command


def test_min_cuda_version_is_emitted_for_gpu_pods_and_validated():
    query = generate_pod_deployment_mutation(
        name="n", image_name="i", gpu_type_id="g", min_cuda_version="13.0"
    )
    assert 'minCudaVersion: "13.0"' in query and "allowedCudaVersions" not in query
    for bad in ("13", "x.y"):
        with pytest.raises(ValueError):
            generate_pod_deployment_mutation(name="n", image_name="i", gpu_type_id="g", min_cuda_version=bad)
    with pytest.raises(ValueError):
        generate_pod_deployment_mutation(
            name="n", image_name="i", gpu_type_id="g", min_cuda_version="13.0", allowed_cuda_versions=["12.4"]
        )


def test_explicit_cpu_image_does_not_insert_ubuntu_template():
    explicit = generate_pod_deployment_mutation(name="n", image_name="img:1")
    assert "templateId" not in explicit and "deployCpuPod" in explicit
    default = generate_pod_deployment_mutation(name="n", image_name="")
    assert 'templateId: "runpod-ubuntu"' in default


def test_observation_queries_select_machine_data_center():
    assert "dataCenterId" in graphql.QUERY_PODS
    assert "dataCenterId" in graphql.generate_pod_query("pod-1")


def test_minimal_query_and_response_are_exact_and_fresh():
    response = {"id": "pod-1", "volumeMounts": [
        {"volumeId": "vol-1", "volumeType": "NETWORK_VOLUME", "mountPath": "/workspace", "readOnly": False}
    ]}
    with patch("runpod.api.graphql.requests.post") as post:
        post.return_value.json.return_value = {"data": {"pod": response}}
        assert graphql.get_pod_mounts("pod-1", api_key="offline-test") == response
        assert graphql.get_pod_mounts("pod-1", api_key="offline-test") == response
        assert post.call_count == 2
        query = json.loads(post.call_args.kwargs["data"])["query"]
    assert query == generate_pod_mounts_query("pod-1")
    assert 'pod(input: {podId: "pod-1"})' in query
    for excluded in ("myself", "env", "networkVolumeId", "objectStores", "runtime", "name", "volumeMountPath"):
        assert excluded not in query


@pytest.mark.parametrize("pod", [None, [], {}, {"id": "other"}, [{"id": "pod-1"}]])
def test_refuses_unjoinable_exact_pod(pod):
    with patch("runpod.api.graphql.requests.post") as post:
        post.return_value.json.return_value = {"data": {"pod": pod}}
        with pytest.raises(ValueError, match="Exact-pod"):
            graphql.get_pod_mounts("pod-1", api_key="offline-test")


@pytest.mark.parametrize("pod_id", [None, "", " pod-1", "p\n", 'p"}) { myself { id } } #', ["pod-1"], "x" * 129])
def test_invalid_pod_id_refuses_before_transport(pod_id):
    with patch("runpod.api.graphql.requests.post") as post:
        with pytest.raises(ValueError, match="Invalid pod ID"):
            graphql.get_pod_mounts(pod_id, api_key="offline-test")
        post.assert_not_called()


def test_partial_graphql_errors_never_establish_observation():
    import runpod

    with patch("runpod.api.graphql.requests.post") as post:
        post.return_value.json.return_value = {
            "data": {"pod": {"id": "pod-1", "volumeMounts": []}},
            "errors": [{"message": "typed projection unsupported", }],
        }
        with pytest.raises(runpod.error.QueryError):
            graphql.get_pod_mounts("pod-1", api_key="offline-test")


def test_partial_errors_rejected_even_with_custom_transport():
    with patch("runpod_lifecycle.graphql._run", return_value={
        "data": {"pod": {"id": "pod-1", "volumeMounts": []}}, "errors": [{"message": "unsupported"}]
    }):
        with pytest.raises(ValueError, match="Exact-pod"):
            graphql.get_pod_mounts("pod-1", api_key="offline-test")
