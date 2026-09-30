"""RunPod SDK and HTTP primitives for the standalone lifecycle package."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import re
import time
from typing import Any, Callable, Sequence

import httpx

from .errors import CleanupPendingError

# Keep the provider SDK behind the call sites.  The Astrid executor sandbox
# permits the declared HTTP/SSH transport but can deny platform probes made by
# optional SDK dependencies during import (notably aiohttp -> os.uname).  The
# REST/GraphQL paths remain sufficient for attach/status/SSH discovery and the
# SDK is still loaded for launch-only operations when the host permits it.
runpod = None  # type: ignore[assignment]

logger = logging.getLogger("runpod_lifecycle.api")


def normalize_allowed_cuda_versions(
    versions: Sequence[str] | None,
) -> tuple[str, ...] | None:
    """Validate and normalize RunPod's allowed CUDA host versions."""
    if versions is None:
        return None
    if isinstance(versions, str):
        raise TypeError("allowed_cuda_versions must be a sequence of version strings")
    normalized: list[str] = []
    for version in versions:
        if not isinstance(version, str):
            raise TypeError("allowed_cuda_versions entries must be strings")
        value = version.strip()
        if not value or not re.fullmatch(r"\d+\.\d+", value):
            raise ValueError(
                "allowed_cuda_versions entries must be non-empty CUDA versions like '12.4'"
            )
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("allowed_cuda_versions must contain at least one version")
    return tuple(normalized)

GRAPHQL_URL = "https://api.runpod.io/graphql"
NETWORK_VOLUMES_URL = "https://rest.runpod.io/v1/networkvolumes"
PODS_URL = "https://rest.runpod.io/v1/pods"


def _get_runpod() -> Any:
    global runpod
    if runpod is None:
        try:
            import runpod as provider_sdk
        except Exception as exc:  # noqa: BLE001
            logger.warning("RunPod SDK unavailable; using HTTP fallbacks: %s", exc)
            return None
        runpod = provider_sdk
    if runpod is None:
        raise RuntimeError("runpod package is required for RunPod API calls")
    return runpod


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def get_network_volumes(api_key: str) -> list[dict[str, Any]]:
    """Return the account's RunPod network volumes."""
    sdk = _get_runpod()
    if sdk is not None:
        sdk.api_key = api_key

    try:
        if sdk is not None and hasattr(sdk, "get_network_volumes"):
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


def list_pods(api_key: str) -> list[dict[str, Any]]:
    """Return account pods, tolerating transient SDK GraphQL response gaps.

    The RunPod SDK indexes ``response["data"]`` directly. Immediately after a
    lifecycle mutation the GraphQL endpoint can transiently return a response
    without that key, causing an otherwise harmless empty/list refresh to fail
    with ``KeyError('data')``. The REST collection is the authoritative fallback
    and also gives termination verification a transport independent of the
    mutation call.
    """
    sdk = _get_runpod()
    if sdk is not None:
        sdk.api_key = api_key
    try:
        pods = sdk.get_pods() if sdk is not None else None
        if isinstance(pods, list):
            return pods
        if pods is not None:
            logger.warning("RunPod SDK returned unexpected pod collection: %r", type(pods).__name__)
    except Exception as exc:
        logger.warning("RunPod SDK get_pods failed; falling back to REST: %s", exc)

    response = httpx.get(PODS_URL, headers=_auth_headers(api_key), timeout=30)
    if response.status_code != 200:
        raise RuntimeError(
            f"RunPod pod listing failed: HTTP {response.status_code}: {response.text[:500]}"
        )
    payload = response.json()
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("pods"), list):
        return payload["pods"]
    raise RuntimeError(f"RunPod pod listing returned unexpected {type(payload).__name__} payload")


def find_gpu_type(gpu_display_name: str, api_key: str) -> dict[str, Any] | None:
    """Find a GPU type by display name or ID."""
    sdk = _get_runpod()
    if sdk is None:
        return None
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
    ports: str | None = None,
    allowed_cuda_versions: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Create a RunPod pod and return provision metadata immediately."""
    sdk = _get_runpod()
    if sdk is None:
        raise RuntimeError(
            "RunPod SDK is unavailable in this runtime; pod provisioning requires the provider SDK"
        )
    sdk.api_key = api_key

    cloud_type = os.getenv("RUNPOD_CLOUD_TYPE", "SECURE").strip().upper() or "SECURE"
    params: dict[str, Any] = {
        "name": name,
        "image_name": image_name,
        "gpu_type_id": gpu_type_id,
        "gpu_count": 1,
        "cloud_type": cloud_type,
        "volume_in_gb": disk_in_gb,
        "container_disk_in_gb": container_disk_in_gb,
        "min_vcpu_count": min_vcpu_count,
        "min_memory_in_gb": min_memory_in_gb,
        "ports": ports or "22/tcp,8888/http",
        "network_volume_id": network_volume_id,
    }

    normalized_cuda_versions = normalize_allowed_cuda_versions(allowed_cuda_versions)
    if normalized_cuda_versions is not None:
        # The SDK maps this snake_case argument to GraphQL's
        # ``allowedCudaVersions`` input on podFindAndDeployOnDemand.
        params["allowed_cuda_versions"] = list(normalized_cuda_versions)

    if template_id:
        params["template_id"] = template_id

    # RunPod uses ``volume_in_gb`` for the pod-local volume when no network
    # volume is attached.  The mount path still has to be sent explicitly;
    # omitting it leaves the requested local volume unmounted and makes
    # ``/workspace`` fall back to the container disk.
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
        **({"allowed_cuda_versions": list(normalized_cuda_versions)} if normalized_cuda_versions else {}),
    }


_CAPACITY_ERROR_MARKERS = (
    "no longer any instances available",
    "there are no machines available",
    "no longer have any instances",
    "not enough capacity",
    "not enough free gpu",
    "no free gpu",
    "out of stock",
)


def _is_capacity_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _CAPACITY_ERROR_MARKERS)


def create_pod_with_fallbacks(
    api_key: str,
    gpu_type_candidates: Sequence[str],
    image_name: str,
    *,
    name_prefix: str = "worker-pod",
    volume_candidates: Sequence[str] = (),
    volume_mount_path: str = "/workspace",
    disk_in_gb: int = 20,
    container_disk_in_gb: int = 10,
    public_key_string: str | None = None,
    env_vars: dict[str, str] | None = None,
    min_vcpu_count: int = 8,
    min_memory_in_gb: int = 32,
    template_id: str | None = None,
    ports: str | None = None,
    allowed_cuda_versions: Sequence[str] | None = None,
    max_full_passes: int = 5,
    retry_sleep_seconds: float = 60.0,
    on_attempt: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Create a pod, iterating (gpu_type, volume) combinations on capacity errors.

    Resolves each candidate gpu_type display name once. Pre-loads the account's
    network volumes once and matches them by name to volume_candidates. Then
    iterates the cartesian product (gpu × volume, plus one no-volume slot if
    nothing matched) until ``create_pod`` succeeds. RunPod capacity errors fall
    back to the next combo. After a full pass exhausts every combo, sleeps
    ``retry_sleep_seconds`` and retries the pass — up to ``max_full_passes``
    times. Non-capacity errors abort immediately.

    Returns the same dict shape as ``create_pod``, enriched with
    ``selected_gpu_display_name`` and ``selected_volume_name`` so callers can
    log which combo actually succeeded.
    """
    if not gpu_type_candidates:
        raise ValueError("gpu_type_candidates must contain at least one GPU type")
    normalized_cuda_versions = normalize_allowed_cuda_versions(allowed_cuda_versions)

    resolved_gpus: list[tuple[str, str]] = []  # (display_name, gpu_type_id)
    for candidate in gpu_type_candidates:
        candidate_str = str(candidate).strip()
        if not candidate_str:
            continue
        info = find_gpu_type(candidate_str, api_key)
        if not info or not info.get("id"):
            logger.warning("create_pod_with_fallbacks: GPU type not found in catalog: %r", candidate_str)
            continue
        resolved_gpus.append((str(info.get("displayName") or candidate_str), str(info["id"])))
    if not resolved_gpus:
        raise RuntimeError(f"None of the candidate GPU types resolved: {list(gpu_type_candidates)}")

    volume_lookup: list[tuple[str | None, str | None]] = []  # (volume_name, volume_id)
    if volume_candidates:
        try:
            available = get_network_volumes(api_key)
            by_name = {v.get("name"): v.get("id") for v in available if v.get("name")}
        except Exception as exc:
            logger.warning("create_pod_with_fallbacks: could not list network volumes (%s)", exc)
            by_name = {}
        seen_volume_names: set[str] = set()
        for name in volume_candidates:
            if not name or name in seen_volume_names:
                continue
            seen_volume_names.add(name)
            vid = by_name.get(name)
            if vid:
                volume_lookup.append((name, vid))
    if not volume_lookup:
        volume_lookup.append((None, None))

    last_capacity_error: BaseException | None = None
    total_attempts = 0
    for full_pass in range(1, max_full_passes + 1):
        for gpu_display, gpu_id in resolved_gpus:
            for vol_name, vol_id in volume_lookup:
                total_attempts += 1
                if on_attempt is not None:
                    try:
                        on_attempt({
                            "attempt": total_attempts,
                            "pass": full_pass,
                            "gpu_display_name": gpu_display,
                            "gpu_type_id": gpu_id,
                            "volume_name": vol_name,
                            "volume_id": vol_id,
                        })
                    except Exception as exc:
                        logger.debug("create_pod_with_fallbacks on_attempt callback failed: %s", exc)
                pod_name = f"{name_prefix}-{int(time.time() * 1000)}"
                try:
                    pod = create_pod(
                        api_key=api_key,
                        gpu_type_id=gpu_id,
                        image_name=image_name,
                        name=pod_name,
                        network_volume_id=vol_id,
                        volume_mount_path=volume_mount_path,
                        disk_in_gb=disk_in_gb,
                        container_disk_in_gb=container_disk_in_gb,
                        public_key_string=public_key_string,
                        env_vars=env_vars,
                        min_vcpu_count=min_vcpu_count,
                        min_memory_in_gb=min_memory_in_gb,
                        template_id=template_id,
                        ports=ports,
                        allowed_cuda_versions=normalized_cuda_versions,
                    )
                except Exception as exc:
                    if _is_capacity_error(exc):
                        last_capacity_error = exc
                        logger.info(
                            "create_pod_with_fallbacks: capacity miss attempt %d (gpu=%s vol=%s); trying next combo",
                            total_attempts,
                            gpu_display,
                            vol_name,
                        )
                        continue
                    raise
                pod["selected_gpu_display_name"] = gpu_display
                pod["selected_gpu_type_id"] = gpu_id
                pod["selected_volume_name"] = vol_name
                pod["selected_volume_id"] = vol_id
                pod["attempt"] = total_attempts
                pod["pass"] = full_pass
                if normalized_cuda_versions is not None:
                    pod["allowed_cuda_versions"] = list(normalized_cuda_versions)
                return pod
        if full_pass < max_full_passes and retry_sleep_seconds > 0:
            logger.info(
                "create_pod_with_fallbacks: pass %d exhausted all %d combos; sleeping %.1fs before retry",
                full_pass,
                len(resolved_gpus) * len(volume_lookup),
                retry_sleep_seconds,
            )
            time.sleep(retry_sleep_seconds)

    if last_capacity_error is not None:
        raise RuntimeError(
            f"create_pod_with_fallbacks: all {total_attempts} attempts hit capacity errors. "
            f"Last error: {last_capacity_error}"
        )
    raise RuntimeError(
        f"create_pod_with_fallbacks: exhausted {total_attempts} attempts without success"
    )


def _normalize_pod_status(runpod_id: str, status: dict[str, Any]) -> dict[str, Any]:
    runtime = status.get("runtime") if isinstance(status, dict) else None
    runtime = runtime if isinstance(runtime, dict) else {}
    ports = runtime.get("ports") or status.get("ports", [])
    ports = ports if isinstance(ports, list) else []
    ip = runtime.get("ip") or status.get("publicIp") or next(
        (port.get("ip") for port in ports if isinstance(port, dict) and port.get("ip")),
        None,
    )
    # REST currently returns compact port strings plus a separate mapping,
    # while GraphQL/SDK return runtime port objects. Normalize the REST shape
    # into the same object form consumed by SSH route and handle code.
    if ports and all(isinstance(port, str) for port in ports):
        mappings = status.get("portMappings")
        mappings = mappings if isinstance(mappings, dict) else {}
        normalized_ports: list[dict[str, Any]] = []
        for port in ports:
            match = re.match(r"^(\d+)/(tcp|http|https)$", port)
            if not match:
                continue
            private_port = int(match.group(1))
            public_port = mappings.get(str(private_port))
            normalized_ports.append({
                "ip": ip,
                "privatePort": private_port,
                "publicPort": public_port,
                "type": match.group(2),
            })
        ports = normalized_ports
    return {
        "runpod_id": runpod_id,
        "desired_status": status.get("desiredStatus"),
        "actual_status": status.get("actualStatus"),
        "ip": ip,
        "ports": ports,
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
            createdAt
            lastStatusChange
            costPerHr
            runtime {
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
        if sdk is not None:
            sdk.api_key = api_key
            status = sdk.get_pod(runpod_id)
            if isinstance(status, dict) and status:
                return _normalize_pod_status(runpod_id, status)
            if status:
                logger.warning("RunPod SDK returned unexpected pod status for %s: %r", runpod_id, status)
    except Exception as exc:
        logger.warning("RunPod SDK pod status lookup failed for %s: %s", runpod_id, exc)

    # REST retains cost_per_hr when the GraphQL compatibility query has to
    # fall back to its reduced schema. This is important for claim-handle
    # normalization: missing provider pricing must fail closed, never become a
    # guessed rate.
    try:
        response = httpx.get(
            f"{PODS_URL}/{runpod_id}",
            headers=_auth_headers(api_key),
            timeout=30,
        )
        if response.status_code == 200:
            payload = response.json()
            if isinstance(payload, dict) and payload:
                return _normalize_pod_status(runpod_id, payload)
    except Exception as exc:
        logger.warning("RunPod REST pod status lookup failed for %s: %s", runpod_id, exc)

    return _get_pod_status_graphql(runpod_id, api_key)


def get_pod_ssh_details(pod_id: str, api_key: str) -> dict[str, Any] | None:
    """Return SSH details (ip, port, password) for a running pod."""
    sdk = _get_runpod()
    if sdk is not None:
        sdk.api_key = api_key

    try:
        status = sdk.get_pod(pod_id) if sdk is not None else None
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

    # The SDK is optional in the Astrid worker sandbox.  REST pod status
    # includes the live runtime sshPassword on accounts/providers that expose
    # it, whereas the reduced GraphQL compatibility query does not.  Resolve
    # this before falling back to the historical ``runpod`` default; otherwise
    # reattaching to a claim handle can reach the right endpoint with a stale
    # password and fail authentication.
    try:
        status = get_pod_status(pod_id, api_key)
        if isinstance(status, dict):
            password = status.get("ssh_password") or "runpod"
            for port_map in status.get("ports", []):
                if isinstance(port_map, dict) and port_map.get("privatePort") == 22:
                    return {
                        "ip": port_map.get("ip") or status.get("ip"),
                        "port": port_map.get("publicPort"),
                        "password": password,
                    }
    except Exception as exc:
        logger.warning("RunPod REST SSH detail lookup failed for %s: %s", pod_id, exc)

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


def start_pod(pod_id: str, api_key: str) -> dict[str, Any]:
    """Request that a stopped Pod be started.

    RunPod exposes Pod start only through the REST API. The response is
    intentionally returned without logging it because provider payloads can
    contain environment and connection metadata.
    """
    response = httpx.post(
        f"{PODS_URL}/{pod_id}/start",
        headers=_auth_headers(api_key),
        timeout=30,
    )
    if response.status_code not in (200, 202):
        raise RuntimeError(
            f"RunPod pod start failed for {pod_id}: "
            f"HTTP {response.status_code}: {response.text[:500]}"
        )
    payload = response.json()
    return payload if isinstance(payload, dict) else {"response": payload}


def _wait_for_pod_absence(
    pod_id: str,
    api_key: str,
    *,
    verify_timeout_seconds: float,
    poll_interval_seconds: float,
) -> None:
    """Wait for this exact provider ID to disappear from account discovery."""
    deadline = time.monotonic() + max(0.0, verify_timeout_seconds)
    while True:
        pods = list_pods(api_key)
        if not any(str(pod.get("id") or "") == pod_id for pod in pods if isinstance(pod, dict)):
            return
        if time.monotonic() >= deadline:
            raise CleanupPendingError(
                pod_id,
                f"pod remained present after {verify_timeout_seconds:.1f}s",
            )
        time.sleep(max(0.0, poll_interval_seconds))


def reconcile_pod_cleanup(
    pod_id: str,
    api_key: str,
    *,
    verify_timeout_seconds: float = 90.0,
    poll_interval_seconds: float = 2.0,
) -> None:
    """Reconcile cleanup for an exact pod ID without issuing another delete.

    Callers may use this after :func:`terminate_pod` raises
    :class:`CleanupPendingError`.  A later confirmed absence returns normally;
    an unresolved resource remains explicitly ``cleanup_pending``.
    """
    _wait_for_pod_absence(
        pod_id,
        api_key,
        verify_timeout_seconds=verify_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )


def terminate_pod(
    pod_id: str,
    api_key: str,
    *,
    verify_timeout_seconds: float = 90.0,
    poll_interval_seconds: float = 2.0,
) -> None:
    """Terminate a pod and return only after provider-side deletion is verified.

    RunPod's supported termination endpoint is REST ``DELETE /v1/pods/{id}``.
    A successful mutation response is not itself a terminal state receipt, so
    poll the independent account collection until the pod is absent. HTTP 404
    is idempotent success. A pod that remains visible past the deadline is an
    error rather than a false ``terminated`` acknowledgement.
    """
    response = httpx.delete(
        f"{PODS_URL}/{pod_id}",
        headers=_auth_headers(api_key),
        timeout=30,
    )
    if response.status_code not in (200, 202, 204, 404):
        raise RuntimeError(
            f"RunPod pod termination failed: HTTP {response.status_code}: {response.text[:500]}"
        )
    if response.status_code == 404:
        return

    reconcile_pod_cleanup(
        pod_id,
        api_key,
        verify_timeout_seconds=verify_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )


def create_network_volume(
    api_key: str,
    name: str,
    size_gb: int,
    datacenter_id: str,
) -> dict[str, Any]:
    """Create a RunPod network volume via REST API.

    POSTs to ``NETWORK_VOLUMES_URL`` with payload ``{name, size, dataCenterId}``.
    Returns the full API response dict on success.
    """
    payload: dict[str, Any] = {
        "name": name,
        "size": size_gb,
        "dataCenterId": datacenter_id,
    }
    response = httpx.post(
        NETWORK_VOLUMES_URL,
        json=payload,
        headers=_auth_headers(api_key),
        timeout=30,
    )
    if response.status_code not in (200, 201):
        logger.error(
            "create_network_volume failed: status=%d body=%s",
            response.status_code,
            response.text[:500],
        )
        raise RuntimeError(
            f"Failed to create network volume '{name}': "
            f"HTTP {response.status_code}: {response.text[:200]}"
        )
    return response.json()


__all__ = [
    "create_pod",
    "create_network_volume",
    "find_gpu_type",
    "get_network_volumes",
    "get_pod_ssh_details",
    "get_pod_status",
    "reconcile_pod_cleanup",
    "start_pod",
    "terminate_pod",
]
