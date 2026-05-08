"""RunPod SDK and HTTP primitives for the standalone lifecycle package."""

from __future__ import annotations

import contextlib
import io
import logging
from typing import Any

import httpx

try:
    import runpod
except ImportError:  # pragma: no cover - exercised indirectly before deps install.
    runpod = None  # type: ignore[assignment]

logger = logging.getLogger("runpod_lifecycle.api")

GRAPHQL_URL = "https://api.runpod.io/graphql"
NETWORK_VOLUMES_URL = "https://api.runpod.io/v1/networkvolumes"


def _get_runpod() -> Any:
    if runpod is None:
        raise RuntimeError("runpod package is required for RunPod API calls")
    return runpod


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def get_network_volumes(api_key: str) -> list[dict[str, Any]]:
    """Return the account's RunPod network volumes."""
    sdk = _get_runpod()
    sdk.api_key = api_key

    try:
        if hasattr(sdk, "get_network_volumes"):
            volumes = sdk.get_network_volumes()
            return volumes if isinstance(volumes, list) else []
    except Exception as exc:
        logger.warning("RunPod SDK get_network_volumes failed: %s", exc)

    try:
        response = httpx.get(NETWORK_VOLUMES_URL, headers=_auth_headers(api_key), timeout=30)
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                return data
    except Exception as exc:
        logger.warning("RunPod REST network volume lookup failed: %s", exc)

    query = """
    query {
      myself {
        networkVolumes {
          id
          name
          size
          dataCenterId
        }
      }
    }
    """
    try:
        response = httpx.post(
            GRAPHQL_URL,
            json={"query": query},
            headers=_auth_headers(api_key),
            timeout=30,
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("data", {}).get("myself", {}).get("networkVolumes", [])
    except Exception as exc:
        logger.warning("RunPod GraphQL network volume lookup failed: %s", exc)

    logger.warning("Could not fetch network volumes from SDK, REST, or GraphQL")
    return []


def find_gpu_type(gpu_display_name: str, api_key: str) -> dict[str, Any] | None:
    """Find a GPU type by display name or ID."""
    sdk = _get_runpod()
    sdk.api_key = api_key

    try:
        gpus = sdk.get_gpus()
    except Exception as exc:
        logger.error("Error retrieving GPU list from RunPod: %s", exc)
        return None

    for gpu in gpus:
        if gpu_display_name in (gpu.get("displayName"), gpu.get("id")):
            return gpu
    return None


def create_pod(
    api_key: str,
    gpu_type_id: str,
    image_name: str,
    name: str = "worker-pod",
    network_volume_id: str | None = None,
    volume_mount_path: str = "/workspace",
    disk_in_gb: int = 20,
    container_disk_in_gb: int = 10,
    public_key_string: str | None = None,
    env_vars: dict[str, str] | None = None,
    min_vcpu_count: int = 8,
    min_memory_in_gb: int = 32,
    template_id: str | None = None,
) -> dict[str, Any]:
    """Create a RunPod pod and return provision metadata immediately."""
    sdk = _get_runpod()
    sdk.api_key = api_key

    params: dict[str, Any] = {
        "name": name,
        "image_name": image_name,
        "gpu_type_id": gpu_type_id,
        "gpu_count": 1,
        "cloud_type": "SECURE",
        "volume_in_gb": disk_in_gb,
        "container_disk_in_gb": container_disk_in_gb,
        "min_vcpu_count": min_vcpu_count,
        "min_memory_in_gb": min_memory_in_gb,
        "ports": "22/tcp,8888/http",
        "network_volume_id": network_volume_id,
    }

    if template_id:
        params["template_id"] = template_id

    if network_volume_id:
        params["volume_mount_path"] = volume_mount_path

    pod_env: dict[str, str] = {}
    if env_vars:
        pod_env.update(env_vars)
    if public_key_string:
        pod_env["PUBLIC_KEY"] = public_key_string
    if pod_env:
        params["env"] = pod_env

    sdk_stdout = io.StringIO()
    with contextlib.redirect_stdout(sdk_stdout):
        pod = sdk.create_pod(**params)
    leaked_stdout = sdk_stdout.getvalue().strip()
    if leaked_stdout:
        logger.debug("RunPod SDK create_pod wrote %d bytes to stdout; suppressed to avoid leaking pod env", len(leaked_stdout))

    pod_data = pod
    if isinstance(pod, dict) and "data" in pod:
        pod_data = pod.get("data", {}).get("podFindAndDeployOnDemand", {})

    pod_id = pod_data.get("id") if isinstance(pod_data, dict) else None
    if not pod_id:
        raise RuntimeError("Pod creation failed (no pod ID returned)")

    return {
        "id": pod_id,
        "desiredStatus": "PROVISIONING",
        "name": name,
        "gpu_type_id": gpu_type_id,
        "created": True,
    }


def _normalize_pod_status(runpod_id: str, status: dict[str, Any]) -> dict[str, Any]:
    runtime = status.get("runtime") if isinstance(status, dict) else None
    runtime = runtime if isinstance(runtime, dict) else {}
    return {
        "runpod_id": runpod_id,
        "desired_status": status.get("desiredStatus"),
        "actual_status": status.get("actualStatus"),
        "ip": runtime.get("ip"),
        "ports": runtime.get("ports", []),
        "ssh_password": runtime.get("sshPassword"),
        "created_at": status.get("createdAt"),
        "last_status_change": status.get("lastStatusChange"),
        "uptime_seconds": runtime.get("uptimeInSeconds", 0),
        "cost_per_hr": status.get("costPerHr"),
    }


def _get_pod_status_graphql(runpod_id: str, api_key: str) -> dict[str, Any] | None:
    queries = [
        """
        query PodStatus($podId: String!) {
          pod(input: {podId: $podId}) {
            id
            desiredStatus
            actualStatus
            createdAt
            lastStatusChange
            costPerHr
            runtime {
              ip
              sshPassword
              uptimeInSeconds
              ports {
                ip
                publicPort
                privatePort
                type
              }
            }
          }
        }
        """,
        """
        query PodStatus($podId: String!) {
          pod(input: {podId: $podId}) {
            id
            desiredStatus
            actualStatus
            runtime {
              ip
              ports {
                ip
                publicPort
                privatePort
                type
              }
            }
          }
        }
        """,
    ]
    for query in queries:
        try:
            response = httpx.post(
                GRAPHQL_URL,
                json={"query": query, "variables": {"podId": runpod_id}},
                headers=_auth_headers(api_key),
                timeout=30,
            )
            if response.status_code != 200:
                logger.warning(
                    "GraphQL pod status lookup query failed for %s: %s",
                    runpod_id,
                    response.status_code,
                )
                continue

            body = response.json()
            if body.get("errors"):
                continue

            pod = body.get("data", {}).get("pod")
            return _normalize_pod_status(runpod_id, pod) if isinstance(pod, dict) else None
        except Exception as exc:
            logger.warning("GraphQL pod status lookup failed for %s: %s", runpod_id, exc)
            return None

    logger.warning("GraphQL pod status lookup returned only errors for %s", runpod_id)
    return None


def get_pod_status(runpod_id: str, api_key: str) -> dict[str, Any] | None:
    """Return normalized pod status details using snake_case keys."""
    try:
        sdk = _get_runpod()
        sdk.api_key = api_key
        status = sdk.get_pod(runpod_id)
        if isinstance(status, dict) and status:
            return _normalize_pod_status(runpod_id, status)
        if status:
            logger.warning("RunPod SDK returned unexpected pod status for %s: %r", runpod_id, status)
    except Exception as exc:
        logger.warning("RunPod SDK pod status lookup failed for %s: %s", runpod_id, exc)

    return _get_pod_status_graphql(runpod_id, api_key)


def get_pod_ssh_details(pod_id: str, api_key: str) -> dict[str, Any] | None:
    """Return SSH details (ip, port, password) for a running pod."""
    sdk = _get_runpod()
    sdk.api_key = api_key

    try:
        status = sdk.get_pod(pod_id)
        if isinstance(status, dict):
            runtime = status.get("runtime", {})
            if isinstance(runtime, dict):
                for port_map in runtime.get("ports", []):
                    if port_map.get("privatePort") == 22:
                        return {
                            "ip": port_map.get("ip"),
                            "port": port_map.get("publicPort"),
                            "password": runtime.get("sshPassword", "runpod"),
                        }
    except Exception as exc:
        logger.warning("RunPod SDK get_pod failed for %s: %s", pod_id, exc)

    query = """
    query PodSshDetails($podId: String!) {
      pod(input: {podId: $podId}) {
        id
        desiredStatus
        runtime {
          ports {
            ip
            publicPort
            privatePort
            type
          }
        }
      }
    }
    """
    try:
        response = httpx.post(
            GRAPHQL_URL,
            json={"query": query, "variables": {"podId": pod_id}},
            headers=_auth_headers(api_key),
            timeout=30,
        )
        if response.status_code == 200:
            pod = response.json().get("data", {}).get("pod")
            if isinstance(pod, dict):
                runtime = pod.get("runtime", {})
                if isinstance(runtime, dict):
                    for port_map in runtime.get("ports", []):
                        if port_map.get("privatePort") == 22:
                            return {
                                "ip": port_map.get("ip"),
                                "port": port_map.get("publicPort"),
                                "password": "runpod",
                            }
        else:
            logger.warning("GraphQL API failed for pod %s: %s", pod_id, response.status_code)
    except Exception as exc:
        logger.warning("GraphQL fallback failed for pod %s: %s", pod_id, exc)

    logger.warning("Could not get SSH details for pod %s via SDK or GraphQL API", pod_id)
    return None


def terminate_pod(pod_id: str, api_key: str) -> None:
    """Terminate a RunPod pod to stop billing."""
    sdk = _get_runpod()
    sdk.api_key = api_key
    sdk.terminate_pod(pod_id)


__all__ = [
    "create_pod",
    "find_gpu_type",
    "get_network_volumes",
    "get_pod_ssh_details",
    "get_pod_status",
    "terminate_pod",
]
