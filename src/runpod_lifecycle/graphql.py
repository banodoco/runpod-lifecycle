"""Lifecycle-owned RunPod GraphQL builders.

The public ``runpod`` SDK only supplies the transport (``run_graphql_query``).
Query and mutation text is built here so lifecycle controls exactly what is
sent: an optional minimum CUDA host version, an explicit CPU image without the
default Ubuntu template, the host data centre in pod observations, a minimal
exact-pod typed mount projection, and ``dockerArgs`` escaped exactly once.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Sequence

_POD_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
_CUDA_VERSION = re.compile(r"\d+\.\d+")

_POD_FIELDS = """
            id
            containerDiskInGb
            costPerHr
            desiredStatus
            dockerArgs
            dockerId
            env
            gpuCount
            imageName
            lastStatusChange
            machineId
            memoryInGb
            name
            podType
            port
            ports
            uptimeSeconds
            vcpuCount
            volumeInGb
            volumeMountPath
            runtime {
                ports{
                    ip
                    isIpPublic
                    privatePort
                    publicPort
                    type
                }
            }
            machine {
                gpuDisplayName
                dataCenterId
            }
"""

# Account pod listing; ``machine.dataCenterId`` is selected so observations
# carry the host data centre.
QUERY_PODS = (
    "\nquery myPods {\n    myself {\n        pods {"
    + _POD_FIELDS
    + "        }\n    }\n}\n"
)

QUERY_NETWORK_VOLUMES = """
query networkVolumes {
    myself {
        networkVolumes {
            id
            dataCenterId
        }
    }
}
"""


def _run(query: str, api_key: Optional[str]) -> dict[str, Any]:
    # Imported lazily: the SDK import is what some sandboxes deny.
    from runpod.api.graphql import run_graphql_query

    return run_graphql_query(query, api_key=api_key)


def generate_pod_query(pod_id: str) -> str:
    """Single-pod observation query (includes ``machine.dataCenterId``)."""
    if not isinstance(pod_id, str) or not _POD_ID.fullmatch(pod_id):
        raise ValueError("Invalid pod ID for observation")
    return f'\n    query pod {{\n        pod(input: {{podId: "{pod_id}"}}) {{{_POD_FIELDS}        }}\n    }}\n'


def generate_pod_mounts_query(pod_id: str) -> str:
    """Select only typed storage evidence for one exact pod, without secrets."""
    if not isinstance(pod_id, str) or not _POD_ID.fullmatch(pod_id):
        raise ValueError("Invalid pod ID for mount observation")
    return f"""
    query podMounts {{
        pod(input: {{podId: "{pod_id}"}}) {{
            id
            volumeMounts {{
                volumeId
                volumeType
                mountPath
                readOnly
            }}
        }}
    }}
    """


def get_pod(pod_id: str, api_key: Optional[str] = None) -> Optional[dict[str, Any]]:
    """One pod, including ``machine.dataCenterId``."""
    return _run(generate_pod_query(pod_id), api_key)["data"]["pod"]


def get_pods(api_key: Optional[str] = None) -> list[dict[str, Any]]:
    """Account pods, each including ``machine.dataCenterId``."""
    return _run(QUERY_PODS, api_key)["data"]["myself"]["pods"]


def get_pod_mounts(pod_id: str, api_key: Optional[str] = None) -> dict[str, Any]:
    """Read a fresh, minimal exact-pod typed mount projection.

    Deliberately does not use account lists, a deployment request, or the broad
    status projection as a substitute for the provider observation.
    """
    raw = _run(generate_pod_mounts_query(pod_id), api_key)
    if isinstance(raw, dict) and raw.get("errors"):
        raise ValueError("Exact-pod mount observation is unavailable")
    data = raw.get("data") if isinstance(raw, dict) else None
    pod = data.get("pod") if isinstance(data, dict) else None
    if not isinstance(pod, dict) or pod.get("id") != pod_id:
        raise ValueError("Exact-pod mount observation is unavailable")
    return pod


def generate_pod_deployment_mutation(
    name: str,
    image_name: str,
    gpu_type_id: Optional[str] = None,
    cloud_type: str = "ALL",
    support_public_ip: bool = True,
    start_ssh: bool = True,
    data_center_id: Optional[str] = None,
    country_code: Optional[str] = None,
    gpu_count: Optional[int] = None,
    volume_in_gb: Optional[int] = None,
    container_disk_in_gb: Optional[int] = None,
    min_vcpu_count: Optional[int] = None,
    min_memory_in_gb: Optional[int] = None,
    docker_args: Optional[str] = None,
    ports: Optional[str] = None,
    volume_mount_path: Optional[str] = None,
    env: Optional[dict] = None,
    template_id: Optional[str] = None,
    network_volume_id: Optional[str] = None,
    allowed_cuda_versions: Optional[Sequence[str]] = None,
    min_download: Optional[int] = None,
    min_upload: Optional[int] = None,
    instance_id: Optional[str] = None,
    min_cuda_version: Optional[str] = None,
) -> str:
    """Build the on-demand pod deployment mutation (GPU or CPU)."""
    if min_cuda_version is not None and (
        not _CUDA_VERSION.fullmatch(min_cuda_version) or allowed_cuda_versions
    ):
        raise ValueError("minimum CUDA must be a version and cannot be combined with exact CUDA versions")

    input_fields = [f'name: "{name}"', f'imageName: "{image_name}"', f"cloudType: {cloud_type}"]
    if start_ssh:
        input_fields.append("startSsh: true")

    if gpu_type_id is not None:
        input_fields.append(f'gpuTypeId: "{gpu_type_id}"')
        input_fields.append(f"supportPublicIp: {str(support_public_ip).lower()}")
        if gpu_count is not None:
            input_fields.append(f"gpuCount: {gpu_count}")
        if volume_in_gb is not None:
            input_fields.append(f"volumeInGb: {volume_in_gb}")
        if min_vcpu_count is not None:
            input_fields.append(f"minVcpuCount: {min_vcpu_count}")
        if min_memory_in_gb is not None:
            input_fields.append(f"minMemoryInGb: {min_memory_in_gb}")
        if allowed_cuda_versions is not None:
            versions = ", ".join(f'"{v}"' for v in allowed_cuda_versions)
            input_fields.append(f"allowedCudaVersions: [{versions}]")
        if min_cuda_version is not None:
            input_fields.append(f'minCudaVersion: "{min_cuda_version}"')
    else:
        if instance_id is not None:
            input_fields.append(f'instanceId: "{instance_id}"')
        # An explicit image is respected; the Ubuntu template is only the
        # default when no image was given.
        if not image_name:
            template_id = template_id or "runpod-ubuntu"

    if data_center_id is not None:
        input_fields.append(f'dataCenterId: "{data_center_id}"')
    else:
        input_fields.append("dataCenterId: null")
    if docker_args is not None:
        # One escaping pass: json.dumps supplies the quotes and escapes.
        input_fields.append(f"dockerArgs: {json.dumps(docker_args)}")
    if country_code is not None:
        input_fields.append(f'countryCode: "{country_code}"')
    if container_disk_in_gb is not None:
        input_fields.append(f"containerDiskInGb: {container_disk_in_gb}")
    if ports is not None:
        input_fields.append(f'ports: "{ports.replace(" ", "")}"')
    if volume_mount_path is not None:
        input_fields.append(f'volumeMountPath: "{volume_mount_path}"')
    if env is not None:
        items = [f'{{ key: "{k}", value: "{v}" }}' for k, v in env.items()]
        input_fields.append(f"env: [{', '.join(items)}]")
    if template_id is not None:
        input_fields.append(f'templateId: "{template_id}"')
    if network_volume_id is not None:
        input_fields.append(f'networkVolumeId: "{network_volume_id}"')
    if min_download is not None:
        input_fields.append(f"minDownload: {min_download}")
    if min_upload is not None:
        input_fields.append(f"minUpload: {min_upload}")

    mutation_type = "podFindAndDeployOnDemand" if gpu_type_id else "deployCpuPod"
    input_string = ", ".join(input_fields)
    return f"""
    mutation {{
      {mutation_type}(
        input: {{
          {input_string}
        }}
      ) {{
        id
        imageName
        env
        machineId
        machine {{
          podHostId
        }}
      }}
    }}
    """


def create_pod(
    name: str,
    image_name: str = "",
    gpu_type_id: Optional[str] = None,
    cloud_type: str = "ALL",
    data_center_id: Optional[str] = None,
    network_volume_id: Optional[str] = None,
    container_disk_in_gb: Optional[int] = None,
    template_id: Optional[str] = None,
    api_key: Optional[str] = None,
    **fields: Any,
) -> dict[str, Any]:
    """Deploy a pod through the public SDK's GraphQL runner; return the pod dict."""
    if not image_name and not template_id:
        raise ValueError("Either image_name or template_id must be provided")
    if cloud_type not in ("ALL", "COMMUNITY", "SECURE"):
        raise ValueError("cloud_type must be one of ALL, COMMUNITY or SECURE")
    if network_volume_id and data_center_id is None:
        volumes = _run(QUERY_NETWORK_VOLUMES, api_key)["data"]["myself"]["networkVolumes"]
        for volume in volumes:
            if volume["id"] == network_volume_id:
                data_center_id = volume["dataCenterId"]
                break
    if container_disk_in_gb is None and template_id is None:
        container_disk_in_gb = 10
    if gpu_type_id is None:
        fields.pop("gpu_count", None)
    mutation = generate_pod_deployment_mutation(
        name=name, image_name=image_name, gpu_type_id=gpu_type_id, cloud_type=cloud_type,
        data_center_id=data_center_id, network_volume_id=network_volume_id,
        container_disk_in_gb=container_disk_in_gb, template_id=template_id, **fields,
    )
    raw = _run(mutation, api_key)
    return raw["data"]["podFindAndDeployOnDemand" if gpu_type_id else "deployCpuPod"]
