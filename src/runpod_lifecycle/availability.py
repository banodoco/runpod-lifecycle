"""Read-only, provider-derived GPU stock and network-storage discovery.

No allocation is attempted. ``available`` is a snapshot, not a reservation.
CUDA constraints filter host capability at the provider; they do not identify
an individual host, driver version, or container toolkit.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Literal

import httpx

from .api import GRAPHQL_URL, _auth_headers, normalize_allowed_cuda_versions

AvailabilityStatus = Literal["available", "unavailable", "unknown"]


@dataclass(frozen=True)
class DataCenterAvailability:
    datacenter_id: str
    name: str | None
    location: str | None
    status: AvailabilityStatus
    stock_level: str | None
    price_per_hour: float | None
    network_volume_support: bool | None
    reason: str | None = None
    # These resolvers do not expose individual host CUDA/driver versions.
    cuda_version: str | None = None
    driver_version: str | None = None


@dataclass(frozen=True)
class GpuAvailability:
    requested_gpu: str
    gpu_type_id: str | None
    gpu_name: str | None
    status: AvailabilityStatus
    observed_at: str
    constraints: dict[str, Any]
    datacenters: list[DataCenterAvailability] = field(default_factory=list)
    reason: str | None = None
    source: str = GRAPHQL_URL

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _QueryUnavailable(RuntimeError):
    """A sanitized provider failure, safe to include in discovery output."""


async def _query(api_key: str, query: str, variables: dict[str, Any]) -> dict[str, Any]:
    try:
        response = await asyncio.to_thread(
            httpx.post,
            GRAPHQL_URL,
            json={"query": query, "variables": variables},
            headers=_auth_headers(api_key),
            timeout=30,
        )
        if response.status_code != 200:
            raise _QueryUnavailable(f"RunPod GraphQL HTTP {response.status_code}")
        body = response.json()
    except (httpx.HTTPError, ValueError):
        raise _QueryUnavailable(
            "RunPod GraphQL transport or JSON response unavailable"
        ) from None
    # Do not echo response bodies, errors or exception text: they may contain
    # credentials. Reject partial GraphQL data rather than infer missing stock.
    if not isinstance(body, dict) or body.get("errors"):
        raise _QueryUnavailable("RunPod GraphQL returned errors or an invalid response")
    if not isinstance(body.get("data"), dict):
        raise _QueryUnavailable("RunPod GraphQL data missing")
    return body["data"]


_DC_FIELDS = """
  dataCenters {
    id name location storageSupport
    gpuAvailability(input: $availability) { gpuTypeId available stockStatus }
  }
"""


async def _fetch_datacenters(
    api_key: str, constraints: dict[str, Any]
) -> list[dict[str, Any]]:
    data = await _query(
        api_key,
        "query DataCenterStock($availability: GpuAvailabilityInput) {"
        + _DC_FIELDS
        + "}",
        {"availability": constraints},
    )
    centers = data.get("dataCenters")
    if (
        not isinstance(centers, list)
        or not centers
        or any(
            not isinstance(dc, dict)
            or not isinstance(dc.get("id"), str)
            or not dc["id"]
            for dc in centers
        )
    ):
        raise _QueryUnavailable("RunPod data-center catalogue missing or invalid")
    return centers


def _constraints(
    secure: bool,
    disk: int,
    cuda: list[str] | tuple[str, ...] | None,
    min_cuda_version: str | None = None,
) -> dict[str, Any]:
    if disk < 0:
        raise ValueError("container_disk_gb must be non-negative")
    normalized = normalize_allowed_cuda_versions(cuda)
    result: dict[str, Any] = {"gpuCount": 1, "secureCloud": secure, "minDisk": disk}
    if normalized:
        result["allowedCudaVersions"] = list(normalized)
    if min_cuda_version is not None:
        result["minCudaVersion"] = normalize_allowed_cuda_versions([min_cuda_version])[
            0
        ]
    return result


def _price(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


async def _for_gpu(
    api_key: str,
    gpu_id: str,
    centers: list[dict[str, Any]],
    constraints: dict[str, Any],
    datacenter_ids: list[str] | None,
    require_network_volume: bool,
) -> list[DataCenterAvailability]:
    selected = set(datacenter_ids) if datacenter_ids is not None else None
    rows: list[tuple[dict[str, Any], dict[str, Any] | None, str | None]] = []
    for dc in centers:
        if selected is not None and dc["id"] not in selected:
            continue
        entries = dc.get("gpuAvailability")
        if not isinstance(entries, list) or any(
            not isinstance(g, dict) or not isinstance(g.get("gpuTypeId"), str)
            for g in entries
        ):
            rows.append((dc, None, "GPU availability missing"))
            continue
        matches = [g for g in entries if g.get("gpuTypeId") == gpu_id]
        if not matches and selected is None:
            continue  # Provider lists GPU offerings per DC; this GPU is not listed.
        rows.append(
            (
                dc,
                matches[0] if len(matches) == 1 else None,
                "GPU not offered in this data center"
                if not matches
                else "Ambiguous provider GPU availability"
                if len(matches) > 1
                else None,
            )
        )
    known = {dc["id"] for dc in centers}
    for dc_id in sorted((selected or set()) - known):
        rows.append(({"id": dc_id}, None, "Data center absent from provider catalogue"))

    # Fetch DC-specific prices in one query per GPU. A global catalogue price
    # cannot be used as a substitute for a particular data center's price.
    priced = [(dc, stock) for dc, stock, reason in rows if stock is not None]
    prices: dict[str, dict[str, Any] | None] = {}
    price_error: str | None = None
    if priced:
        definitions = ["$gpu: GpuTypeFilter"]
        variables: dict[str, Any] = {"gpu": {"id": gpu_id}}
        fields = []
        for index, (dc, _) in enumerate(priced):
            definitions.append(f"$p{index}: GpuLowestPriceInput")
            variables[f"p{index}"] = {**constraints, "dataCenterId": dc["id"]}
            fields.append(
                f"dc{index}: lowestPrice(input: $p{index}) "
                "{ stockStatus uninterruptablePrice }"
            )
        query = (
            "query DataCenterPrices("
            + ", ".join(definitions)
            + ") { gpuTypes(input: $gpu) { id "
            + " ".join(fields)
            + " } }"
        )
        try:
            data = await _query(api_key, query, variables)
            gpus = data.get("gpuTypes")
            if (
                not isinstance(gpus, list)
                or len(gpus) != 1
                or not isinstance(gpus[0], dict)
                or gpus[0].get("id") != gpu_id
            ):
                raise _QueryUnavailable("RunPod data-center pricing missing")
            prices = {
                dc["id"]: gpus[0].get(f"dc{index}")
                for index, (dc, _) in enumerate(priced)
            }
        except _QueryUnavailable as exc:
            price_error = str(exc)

    results = []
    for dc, stock, reason in rows:
        support = dc.get("storageSupport")
        support = support if isinstance(support, bool) else None
        status: AvailabilityStatus = "unknown"
        if reason == "GPU not offered in this data center":
            status = "unavailable"
        if stock is not None:
            available = stock.get("available")
            status = (
                "available"
                if available is True
                else "unavailable"
                if available is False
                else "unknown"
            )
            reason = (
                None
                if available is True
                else "Provider reports no matching stock"
                if available is False
                else "Provider availability flag missing"
            )
        if require_network_volume:
            if support is False:
                status, reason = "unavailable", "Network volumes are not supported"
            elif support is None and status != "unavailable":
                status, reason = "unknown", "Network-volume support unknown"
        price_info = prices.get(dc["id"])
        price = (
            _price(price_info.get("uninterruptablePrice"))
            if isinstance(price_info, dict)
            else None
        )
        if status == "available" and (price is None or price_error):
            status, reason = "unknown", price_error or "Data-center price unavailable"
        # Conflicting snapshots must not be advertised as available.
        stock_level = stock.get("stockStatus") if stock else None
        if (
            status == "available"
            and isinstance(price_info, dict)
            and price_info.get("stockStatus") != stock_level
        ):
            status, reason = "unknown", "Provider stock snapshots disagree; retry"
        results.append(
            DataCenterAvailability(
                datacenter_id=dc["id"],
                name=dc.get("name"),
                location=dc.get("location"),
                status=status,
                stock_level=stock_level if isinstance(stock_level, str) else None,
                price_per_hour=price,
                network_volume_support=support,
                reason=reason,
            )
        )
    return sorted(
        results,
        key=lambda dc: (
            dc.price_per_hour is None,
            dc.price_per_hour or 0,
            dc.datacenter_id,
        ),
    )


async def gpu_availability(
    *,
    api_key: str,
    gpu: str,
    datacenter_ids: list[str] | None = None,
    require_secure_cloud: bool = True,
    require_network_volume: bool = True,
    container_disk_gb: int = 100,
    allowed_cuda_versions: list[str] | tuple[str, ...] | None = None,
    min_cuda_version: str | None = None,
) -> GpuAvailability:
    """Resolve an exact catalogue ID/display name (case insensitive) and query stock.

    Returns typed unknown results on provider/schema/transport failure. An
    unknown GPU name is unavailable; an ambiguous name is unknown. No region
    is selected automatically. ``None`` means all DCs; an empty allow-list
    means none. Only single-GPU, on-demand availability is checked.
    """
    constraints = _constraints(
        require_secure_cloud, container_disk_gb, allowed_cuda_versions, min_cuda_version
    )
    evidence = {
        **constraints,
        "requireNetworkVolume": require_network_volume,
        "datacenterIds": datacenter_ids,
    }
    observed_at = datetime.now(timezone.utc).isoformat()
    gpu_id = gpu_name = None
    try:
        data = await _query(
            api_key, "query GpuCatalogue { gpuTypes { id displayName memoryInGb } }", {}
        )
        catalogue = data.get("gpuTypes")
        if (
            not isinstance(catalogue, list)
            or not catalogue
            or any(not isinstance(g, dict) or not g.get("id") for g in catalogue)
        ):
            raise _QueryUnavailable("RunPod GPU catalogue missing or invalid")
        matches = [
            g
            for g in catalogue
            if gpu.strip().casefold()
            in (
                str(g.get("id", "")).casefold(),
                str(g.get("displayName", "")).casefold(),
            )
        ]
        if len(matches) != 1:
            return GpuAvailability(
                gpu,
                None,
                None,
                "unavailable" if not matches else "unknown",
                observed_at,
                evidence,
                reason="GPU not found in provider catalogue"
                if not matches
                else "Ambiguous GPU name; use a catalogue ID",
            )
        gpu_id, gpu_name = matches[0]["id"], matches[0].get("displayName")
        centers = await _fetch_datacenters(api_key, constraints)
        results = await _for_gpu(
            api_key,
            gpu_id,
            centers,
            constraints,
            datacenter_ids,
            require_network_volume,
        )
        status: AvailabilityStatus = (
            "available"
            if any(dc.status == "available" for dc in results)
            else "unknown"
            if any(dc.status == "unknown" for dc in results)
            else "unavailable"
        )
        return GpuAvailability(
            gpu,
            gpu_id,
            gpu_name,
            status,
            observed_at,
            evidence,
            results,
            None
            if status == "available"
            else "No confirmed available data center matches the constraints",
        )
    except _QueryUnavailable as exc:
        return GpuAvailability(
            gpu, gpu_id, gpu_name, "unknown", observed_at, evidence, reason=str(exc)
        )


__all__ = [
    "AvailabilityStatus",
    "DataCenterAvailability",
    "GpuAvailability",
    "gpu_availability",
]
