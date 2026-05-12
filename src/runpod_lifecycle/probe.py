"""Probe RunPod GPU availability without provisioning a pod.

The :func:`probe` function answers the question "what configuration could
actually launch right now, given my constraints?" — by querying RunPod's
GraphQL ``gpuTypes`` schema and returning a price-ranked list of viable
candidates. This avoids the trial-and-error provisioning loop where every
failed attempt costs 20s+ of RAM-tier iteration.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from .api import GRAPHQL_URL, _auth_headers

logger = logging.getLogger("runpod_lifecycle.probe")

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
    lowestPrice(input: {gpuCount: 1, secureCloud: __SECURE__}) {
      uninterruptablePrice
    }
  }
}
"""


def _build_query(require_secure_cloud: bool) -> str:
    return _PROBE_QUERY_TEMPLATE.replace(
        "__SECURE__", "true" if require_secure_cloud else "false"
    )


def _is_blackwell(gpu_id: str | None, display_name: str | None) -> bool:
    haystack = f"{gpu_id or ''} {display_name or ''}".lower()
    return "blackwell" in haystack


async def _fetch_gpu_types(
    api_key: str, require_secure_cloud: bool
) -> list[dict[str, Any]]:
    query = _build_query(require_secure_cloud)

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
) -> list[dict[str, Any]]:
    """Return a price-ranked list of viable pod configurations.

    Each returned entry has the shape::

        {
          "gpu_type": "NVIDIA RTX 6000 Ada Generation",
          "memory_gb": 48,
          "price_per_hour": 0.77,
          "secure_cloud": True,
          "is_blackwell": False,
          "datacenters_available": [],
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
        Reserved for the eventual datacenter-availability lookup; currently
        unused but accepted for forward compatibility with the brief.
    datacenter_ids:
        Optional restriction list. The ``datacenters_available`` field is
        returned as ``[]`` for now (TODO below); when this argument is set
        and the field is empty we still return the entry so callers can rank
        and try them — actual DC-level capacity must be inferred by attempting
        to launch.

    Returns
    -------
    list[dict]
        Configurations sorted by ``price_per_hour`` ascending. GPU types with
        no ``lowestPrice`` (i.e. no current availability under the
        ``secureCloud`` flag requested) are filtered out.

    Notes
    -----
    TODO: RunPod's public GraphQL schema does not expose a clean per-GPU
    datacenter availability list. ``datacenters_available`` is therefore a
    best-effort empty list ``[]`` for the first cut; a future revision can
    fill it in once we settle on whether to scrape the ``Stockless``
    detection endpoint or the ``dataCenters { compute }`` resolver (the
    latter is admin-gated as of this writing).
    """
    # ``container_disk_gb`` and ``datacenter_ids`` accepted but unused for
    # the first cut — see TODO above.
    del container_disk_gb

    raw = await _fetch_gpu_types(api_key, require_secure_cloud)

    gpu_type_allowlist: set[str] | None = (
        set(gpu_types) if gpu_types is not None else None
    )

    results: list[dict[str, Any]] = []
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
        if price_raw is None:
            # No availability under the requested cloud flag — drop it.
            continue
        try:
            price = float(price_raw)
        except (TypeError, ValueError):
            continue

        if max_price_per_hour is not None and price > max_price_per_hour:
            continue

        results.append(
            {
                "gpu_type": display_name or gpu_id or "",
                "memory_gb": memory_gb,
                "price_per_hour": price,
                "secure_cloud": bool(require_secure_cloud),
                "is_blackwell": blackwell,
                "datacenters_available": [],  # TODO: see docstring.
            }
        )

    # ``datacenter_ids`` is accepted today purely as a no-op annotation; once
    # availability data is plumbed through we will filter ``results`` by it.
    if datacenter_ids:
        logger.debug(
            "probe: datacenter_ids=%s requested but availability data is not yet "
            "plumbed; returning unfiltered results",
            datacenter_ids,
        )

    results.sort(key=lambda r: r["price_per_hour"])
    return results


__all__ = ["probe"]
