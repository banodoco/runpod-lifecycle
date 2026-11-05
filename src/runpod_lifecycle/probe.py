"""Probe RunPod GPU availability without provisioning a pod.

The :func:`probe` function ranks provider catalogue candidates and enriches
them with data-center stock and price evidence. Only confirmed stock satisfies
a requested data-center filter; unknown availability remains explicit. No pod
is provisioned during discovery.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from typing import Any

import httpx
import os
import sys

from .api import GRAPHQL_URL, _auth_headers, normalize_allowed_cuda_versions
from .availability import _QueryUnavailable, _constraints, _fetch_datacenters, _for_gpu, _query

# GraphQL query used by :func:`probe`. ``lowestPrice`` is parameterised at
# request time because RunPod's schema expects a literal boolean rather than
# a variable on the inner input object in some deployments.
_PROBE_QUERY_TEMPLATE = """
query GpuTypesProbe {
  gpuTypes {
    id
    displayName
    memoryInGb
    secureCloud
    communityCloud
    lowestPrice(input: {gpuCount: 1, secureCloud: __SECURE____CUDA__}) {
      uninterruptablePrice
    }
  }
}
"""


def _build_query(
    require_secure_cloud: bool,
    allowed_cuda_versions: tuple[str, ...] | None = None,
) -> str:
    cuda = ""
    if allowed_cuda_versions:
        cuda = ", allowedCudaVersions: [" + ", ".join(f'\"{v}\"' for v in allowed_cuda_versions) + "]"
    return _PROBE_QUERY_TEMPLATE.replace(
        "__SECURE__", "true" if require_secure_cloud else "false"
    ).replace("__CUDA__", cuda)


def _is_blackwell(gpu_id: str | None, display_name: str | None) -> bool:
    haystack = f"{gpu_id or ''} {display_name or ''}".lower()
    return "blackwell" in haystack


async def _fetch_gpu_types(
    api_key: str,
    require_secure_cloud: bool,
    allowed_cuda_versions: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    query = _build_query(require_secure_cloud, allowed_cuda_versions)

    def _post() -> httpx.Response:
        return httpx.post(
            GRAPHQL_URL,
            json={"query": query},
            headers=_auth_headers(api_key),
            timeout=30,
        )

    response = await asyncio.to_thread(_post)
    if response.status_code != 200:
        raise RuntimeError(
            f"RunPod GraphQL gpuTypes query failed: HTTP {response.status_code}: "
            f"{response.text[:200]}"
        )
    body = response.json()
    if body.get("errors"):
        raise RuntimeError(
            f"RunPod GraphQL gpuTypes query returned errors: {body['errors']}"
        )
    gpu_types = body.get("data", {}).get("gpuTypes") or []
    if not isinstance(gpu_types, list):
        raise RuntimeError(
            f"RunPod GraphQL gpuTypes query returned unexpected payload: {body!r}"
        )
    return gpu_types


async def probe(
    *,
    api_key: str,
    gpu_types: list[str] | None = None,
    min_memory_gb: int = 24,
    max_price_per_hour: float | None = None,
    require_secure_cloud: bool = True,
    exclude_blackwell: bool = False,
    container_disk_gb: int = 100,
    datacenter_ids: list[str] | None = None,
    allowed_cuda_versions: list[str] | tuple[str, ...] | None = None,
    min_cuda_version: str | None = None,
) -> list[dict[str, Any]]:
    """Return price-ranked catalogue candidates with data-center stock evidence.

    Each returned entry has the shape::

        {
          "gpu_type": "NVIDIA RTX 6000 Ada Generation",
          "memory_gb": 48,
          "price_per_hour": 0.77,
          "secure_cloud": True,
          "is_blackwell": False,
          "datacenters_available": ["EU-RO-1"],
          "availability_status": "available",
          "datacenter_availability": [{"datacenter_id": "EU-RO-1", ...}],
        }

    Parameters
    ----------
    api_key:
        RunPod API key used for the GraphQL request.
    gpu_types:
        Optional allow-list of GPU type ``id`` values (case-sensitive).
        ``None`` means "consider every type RunPod returns".
    min_memory_gb:
        Minimum VRAM (``memoryInGb`` from RunPod) the GPU must report.
    max_price_per_hour:
        Optional cap on the hourly uninterruptable price.
    require_secure_cloud:
        When ``True`` the ``lowestPrice`` lookup restricts to Secure Cloud
        instances (and the returned ``secure_cloud`` flag is always ``True``).
    exclude_blackwell:
        Filter out GPU types whose id/display name contains ``"Blackwell"``
        (case-insensitive). Banodoco hivemind reports a training-quality
        regression on Blackwell variants.
    container_disk_gb:
        Minimum free container disk required by the provider stock query.
    datacenter_ids:
        Optional exact DC allow-list. Only configurations with confirmed stock
        in this list are returned. An empty list matches no data centers.

    Results include DC-specific evidence in ``datacenter_availability`` and
    confirmed available IDs in ``datacenters_available``. Without a DC filter,
    catalogue candidates whose DC stock cannot be determined are retained with
    an explicit ``availability_status`` of ``unknown``; price alone is never
    proof of stock. CUDA constraints apply to both stock and price queries.
    No resource is allocated, and stock can change immediately after querying.
    """
    constraints = _constraints(require_secure_cloud, container_disk_gb, allowed_cuda_versions, min_cuda_version)
    normalized_cuda_versions = normalize_allowed_cuda_versions(allowed_cuda_versions)
    if datacenter_ids is not None:
        catalogue = await _query(api_key, "query GpuCatalogue { gpuTypes { id displayName memoryInGb } }", {})
        raw = catalogue.get("gpuTypes") or []
    else:
        raw = await _fetch_gpu_types(api_key, require_secure_cloud, normalized_cuda_versions)

    gpu_type_allowlist: set[str] | None = (
        set(gpu_types) if gpu_types is not None else None
    )

    results: list[dict[str, Any]] = []
    if os.environ.get("RUNPOD_PROBE_DIAGNOSTICS"):
        print(f"probe catalogue rows: {len(raw)}", file=sys.stderr)
    for entry in raw:
        if not isinstance(entry, dict):
            continue

        gpu_id = entry.get("id")
        display_name = entry.get("displayName")
        memory_gb_raw = entry.get("memoryInGb")
        try:
            memory_gb = int(memory_gb_raw) if memory_gb_raw is not None else 0
        except (TypeError, ValueError):
            memory_gb = 0

        if gpu_type_allowlist is not None and gpu_id not in gpu_type_allowlist:
            continue

        if memory_gb < min_memory_gb:
            continue

        blackwell = _is_blackwell(gpu_id, display_name)
        if exclude_blackwell and blackwell:
            continue

        lowest = entry.get("lowestPrice") or {}
        price_raw = lowest.get("uninterruptablePrice") if isinstance(lowest, dict) else None
        if price_raw is None and not datacenter_ids:
            # No availability under the requested cloud flag — drop it.
            continue
        try:
            price = float(price_raw) if price_raw is not None else 0.0
        except (TypeError, ValueError):
            continue

        if not datacenter_ids and max_price_per_hour is not None and price > max_price_per_hour:
            continue

        results.append(
            {
                "gpu_type": display_name or gpu_id or "",
                "memory_gb": memory_gb,
                "price_per_hour": price,
                "secure_cloud": bool(require_secure_cloud),
                "is_blackwell": blackwell,
                "datacenters_available": [],
                "gpu_type_id": gpu_id,
                **({"allowed_cuda_versions": list(normalized_cuda_versions)} if normalized_cuda_versions else {}),
            }
        )

    if results:
        try:
            centers = await _fetch_datacenters(api_key, constraints)
            for result in results:
                try:
                    dcs = await _for_gpu(
                        api_key, result["gpu_type_id"], centers, constraints,
                        datacenter_ids, True,
                    )
                except _QueryUnavailable as exc:
                    result["datacenter_availability"] = []
                    result["availability_status"] = "unknown"
                    result["availability_reason"] = str(exc)
                    continue
                # A price cap must also apply to DC-specific prices.
                available = [dc for dc in dcs if dc.status == "available" and
                             (max_price_per_hour is None or dc.price_per_hour <= max_price_per_hour)]
                result["datacenter_availability"] = [asdict(dc) for dc in dcs]
                if os.environ.get("RUNPOD_PROBE_DIAGNOSTICS") and any(dc.status != "unavailable" for dc in dcs):
                    print(result["gpu_type_id"], result["datacenter_availability"], file=sys.stderr)
                result["datacenters_available"] = [dc.datacenter_id for dc in available]
                result["availability_status"] = (
                    "available" if available else "unknown" if any(dc.status == "unknown" for dc in dcs) else "unavailable"
                )
                if available:
                    result["price_per_hour"] = min(dc.price_per_hour for dc in available)
        except _QueryUnavailable as exc:
            for result in results:
                result["datacenter_availability"] = []
                result["availability_status"] = "unknown"
                result["availability_reason"] = str(exc)
    if datacenter_ids is not None:
        results = [result for result in results if result["datacenters_available"]]

    results.sort(key=lambda r: r["price_per_hour"])
    return results


__all__ = ["probe"]
