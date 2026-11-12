"""REST v2 wire shapes projected onto lifecycle's SDK-compatible records.

Source: https://github.com/runpod/docs/blob/main/api-reference-v2/openapi.json
GraphQL projections stay unchanged. These aliases describe provider configuration;
they do not establish the typed, fulfilled mount evidence used by storage probes.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _alias(record: dict[str, Any], key: str, value: Any) -> None:
    if key in record and record[key] != value:
        raise ValueError(f"Conflicting provider field: {key}")
    record[key] = value


def normalize_volume(raw: dict[str, Any]) -> dict[str, Any]:
    result = dict(raw)
    if "dataCenter" in raw:
        _alias(result, "dataCenterId", raw["dataCenter"])
    return result


def normalize_pod(raw: dict[str, Any]) -> dict[str, Any]:
    result = dict(raw)
    for wire, internal in (("image", "imageName"), ("status", "desiredStatus"),
                           ("cost", "costPerHr"), ("disk", "containerDiskInGb")):
        if wire in raw:
            _alias(result, internal, raw[wire])
    gpu = raw.get("gpu")
    if isinstance(gpu, dict):
        if "count" in gpu:
            _alias(result, "gpuCount", gpu["count"])
        if "id" in gpu:
            _alias(result, "machineType", gpu["id"])
        result.setdefault("computeType", "GPU")
    elif isinstance(raw.get("cpu"), dict):
        _alias(result, "gpuCount", 0)
        result.setdefault("computeType", "CPU")
    if "mounts" in raw:
        mounts = raw["mounts"]
        if not isinstance(mounts, dict):
            raise ValueError("Malformed provider mounts")
        network = mounts.get("network", [])
        if (not isinstance(network, list) or any(not isinstance(m, dict)
                or not isinstance(m.get("volumeId"), str) or not m["volumeId"]
                or not isinstance(m.get("path"), str) or not m["path"].startswith("/")
                for m in network)):
            raise ValueError("Malformed provider network mounts")
        # A multi-volume pod has no unambiguous single-volume legacy alias.
        if len(network) > 1 and any(key in raw for key in ("networkVolume", "networkVolumeId", "volumeMountPath")):
            raise ValueError("Conflicting provider volume aliases for multiple mounts")
        if len(network) <= 1:
            volume = {"id": network[0]["volumeId"]} if network else None
            _alias(result, "networkVolume", volume)
            if "networkVolumeId" in result and result["networkVolumeId"] != (volume or {}).get("id"):
                raise ValueError("Conflicting provider volume identity")
            persistent = mounts.get("persistent")
            path = network[0]["path"] if network else (
                persistent.get("path") if isinstance(persistent, dict) else None)
            _alias(result, "volumeMountPath", path)
    runtime = raw.get("runtime")
    if isinstance(runtime, dict):
        runtime = dict(runtime)
        if "uptime" in runtime:
            _alias(runtime, "uptimeInSeconds", runtime["uptime"])
        ports = runtime.get("ports")
        if isinstance(ports, list):
            normalized = []
            for port in ports:
                if not isinstance(port, dict):
                    raise ValueError("Malformed provider runtime port")
                port = dict(port)
                for wire, internal in (("private", "privatePort"), ("public", "publicPort")):
                    if wire in port:
                        _alias(port, internal, port.pop(wire))
                normalized.append(port)
            runtime["ports"] = normalized
        result["runtime"] = runtime
    return result


def collect_pages(fetch: Callable[[dict[str, str]], Any], collection: str) -> list[dict[str, Any]]:
    """Walk v2 cursor pages; incomplete/malformed inventory must never mean absent."""
    params: dict[str, str] = {}
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    while True:
        body = fetch(dict(params))
        if not isinstance(body, dict) or not isinstance(body.get(collection), list):
            raise ValueError("Malformed provider collection")
        items = body[collection]
        if any(not isinstance(item, dict) for item in items):
            raise ValueError("Malformed provider collection item")
        records.extend(items)
        pagination = body.get("pagination")
        if not isinstance(pagination, dict) or type(pagination.get("hasNextPage")) is not bool:
            raise ValueError("Malformed provider pagination")
        if not pagination["hasNextPage"]:
            return records
        cursor = pagination.get("nextCursor")
        if not isinstance(cursor, str) or not cursor or cursor in seen:
            raise ValueError("Invalid or repeated provider cursor")
        seen.add(cursor)
        params["cursor"] = cursor
