"""Account-wide RunPod discovery, attach-by-id, orphan detection, and module-level termination."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from . import api
from .config import RunPodConfig
from .errors import CleanupPendingError, LaunchFailure, TerminateError
from .events import EventHooks, PodState, _emit_error, _emit_state
from .pod import Pod

logger = logging.getLogger("runpod_lifecycle.discovery")


def _mount_text(value: Any, *, path: bool = False) -> str:
    if not isinstance(value, str) or not value or len(value) > 512 or value.strip() != value:
        raise ValueError("Malformed provider mount evidence")
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("Malformed provider mount evidence")
    if path:
        if not value.startswith("/") or value == "/" or any(
            part in {"", ".", ".."} for part in value[1:].split("/")
        ) or "\\" in value:
            raise ValueError("Malformed provider mount path")
    elif not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
        raise ValueError("Malformed provider mount identity")
    return value


def normalize_pod_mounts(raw: Any, pod_id: str, nonce: str) -> dict[str, Any]:
    """Project strict typed evidence, preserving only legacy conflict verdicts.

    The nonce binds the output to this invocation. It does not establish a
    provider-side observation timestamp or fulfilled-mount semantics; the
    authenticated live filesystem probe remains independently required.
    """
    _mount_text(pod_id)
    if not isinstance(nonce, str) or not re.fullmatch(r"[a-f0-9]{32}", nonce):
        raise ValueError("Invalid provider mount observation nonce")
    if not isinstance(raw, dict) or raw.get("id") != pod_id:
        raise ValueError("Exact-pod mount observation is unavailable")
    entries = raw.get("volumeMounts")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 16:
        raise ValueError("Typed provider mounts are unavailable or ambiguous")
    mounts = []
    paths: set[str] = set()
    identities: set[tuple[str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Malformed provider mount evidence")
        volume_id = _mount_text(entry.get("volumeId"))
        volume_type = entry.get("volumeType")
        mount_path = _mount_text(entry.get("mountPath"), path=True)
        read_only = entry.get("readOnly")
        if not isinstance(volume_type, str) or volume_type not in {"NETWORK_VOLUME", "OBJECT_STORE_VOLUME"} or type(read_only) is not bool:
            raise ValueError("Malformed provider mount policy")
        identity = (volume_type, volume_id)
        if mount_path in paths or identity in identities:
            raise ValueError("Duplicate provider mount evidence")
        paths.add(mount_path)
        identities.add(identity)
        mounts.append({"volume_id": volume_id, "volume_type": volume_type,
                       "mount_path": mount_path, "read_only": read_only})

    legacy_present = any(raw.get(k) is not None for k in ("networkVolumeId", "networkVolume", "objectStores", "volumeMountPath"))
    network_ids = {m["volume_id"] for m in mounts if m["volume_type"] == "NETWORK_VOLUME"}
    legacy_ids = []
    if raw.get("networkVolumeId") is not None:
        legacy_ids.append(_mount_text(raw["networkVolumeId"]))
    if raw.get("networkVolume") is not None:
        volume = raw["networkVolume"]
        if not isinstance(volume, dict):
            raise ValueError("Malformed legacy provider mount evidence")
        legacy_ids.append(_mount_text(volume.get("id")))
    if legacy_ids and (len(set(legacy_ids)) != 1 or legacy_ids[0] not in network_ids):
        raise ValueError("Conflicting legacy provider mount evidence")
    if raw.get("volumeMountPath") is not None:
        legacy_path = _mount_text(raw["volumeMountPath"], path=True)
        if not any(m["mount_path"] == legacy_path and m["volume_type"] == "NETWORK_VOLUME" for m in mounts):
            raise ValueError("Conflicting legacy provider mount evidence")
    if raw.get("objectStores") is not None:
        objects = raw["objectStores"]
        if not isinstance(objects, list) or len(objects) > 16:
            raise ValueError("Malformed legacy provider mount evidence")
        observed_objects = set()
        for obj in objects:
            if not isinstance(obj, dict) or type(obj.get("readOnly")) is not bool:
                raise ValueError("Malformed legacy provider mount evidence")
            facts = (_mount_text(obj.get("objectStoreId")),
                     _mount_text(obj.get("mountPath"), path=True), obj["readOnly"])
            if facts in observed_objects:
                raise ValueError("Duplicate legacy provider mount evidence")
            observed_objects.add(facts)
        typed_objects = {(m["volume_id"], m["mount_path"], m["read_only"]) for m in mounts
                         if m["volume_type"] == "OBJECT_STORE_VOLUME"}
        if observed_objects != typed_objects:
            raise ValueError("Conflicting legacy provider mount evidence")
    return {"schema": "runpod.provider-mounts.v1", "pod_id": pod_id, "nonce": nonce,
            "volume_mounts": mounts,
            "legacy_evidence": {"present": legacy_present, "valid": True, "conflict": False}}


async def observe_pod_mounts(pod_id: str, api_key: str, nonce: str) -> dict[str, Any]:
    """Obtain a new exact-pod SDK observation for each invocation."""
    _mount_text(pod_id)
    if not isinstance(nonce, str) or not re.fullmatch(r"[a-f0-9]{32}", nonce):
        raise ValueError("Invalid provider mount observation nonce")
    raw = await asyncio.to_thread(api.get_pod_mounts, pod_id, api_key)
    return normalize_pod_mounts(raw, pod_id, nonce)


@dataclass(frozen=True)
class PodSummary:
    id: str
    name: str | None
    desired_status: str | None
    actual_status: str | None
    gpu_type: str | None
    image: str | None
    created_at: str | None
    cost_per_hr: float
    uptime_seconds: int | None
    ports: list[dict[str, Any]]
    network_volume_id: str | None
    network_volume_evidence: dict[str, bool] | None = None
    network_volume_mount_path: str | None = None
    legacy_object_mounts: list[dict[str, Any]] | None = None


def _legacy_mount_facts(raw: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]] | None, bool, bool]:
    """Keep only bounded legacy path/policy facts; never project store secrets."""
    path = None
    objects = None
    present = raw.get("volumeMountPath") is not None or raw.get("objectStores") is not None
    try:
        if raw.get("volumeMountPath") is not None:
            path = _mount_text(raw["volumeMountPath"], path=True)
        if raw.get("objectStores") is not None:
            entries = raw["objectStores"]
            if not isinstance(entries, list) or len(entries) > 16:
                raise ValueError("Malformed legacy provider mount evidence")
            objects = []
            identities = set()
            paths = set()
            for entry in entries:
                if not isinstance(entry, dict) or type(entry.get("readOnly")) is not bool:
                    raise ValueError("Malformed legacy provider mount evidence")
                identity = _mount_text(entry.get("objectStoreId"))
                object_path = _mount_text(entry.get("mountPath"), path=True)
                if identity in identities or object_path in paths:
                    raise ValueError("Duplicate legacy provider mount evidence")
                identities.add(identity)
                paths.add(object_path)
                objects.append({"volume_id": identity, "mount_path": object_path, "read_only": entry["readOnly"]})
    except ValueError:
        return None, None, False, present
    return path, objects, True, present


def _network_volume_evidence(raw: dict[str, Any]) -> dict[str, bool]:
    """Retain malformed/conflicting independent legacy facts in CLI lists.

    All-null fields supply no independent evidence. Once one field supplies a
    value, null/malformed companion fields cannot silently erase a conflict.
    """
    values = []
    malformed = False
    if "networkVolumeId" in raw:
        values.append(raw["networkVolumeId"])
    if "networkVolume" in raw:
        volume = raw["networkVolume"]
        if volume is None:
            values.append(None)
        elif isinstance(volume, dict):
            values.append(volume.get("id"))
            malformed = "id" not in volume or volume.get("id") is None
        else:
            malformed = True
    _, _, valid_mounts, mount_present = _legacy_mount_facts(raw)
    identity_present = any(v is not None for v in values)
    present = malformed or identity_present or mount_present
    if not present:
        return {"present": False, "valid": True, "conflict": False}
    valid_ids = [v for v in values if isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", v)]
    valid = not malformed and valid_mounts and (not identity_present or len(valid_ids) == len(values))
    conflict = len(set(valid_ids)) > 1
    return {"present": True, "valid": valid, "conflict": conflict}


def _network_volume_id(raw: dict[str, Any]) -> str | None:
    """Normalize observed SDK/REST identities, rejecting malformed/conflicting evidence.

    REST v2 mounts.network references are normalized to networkVolume.id.
    It is an attached-volume object, not the similarly named request filter.
    """
    identities = []
    if "networkVolumeId" in raw:
        identities.append(raw["networkVolumeId"])
    if "networkVolume" in raw:
        volume = raw["networkVolume"]
        identities.append(volume.get("id") if isinstance(volume, dict) else None)
    if not identities or any(not isinstance(v, str) or not v or v.strip() != v for v in identities):
        return None
    return identities[0] if all(v == identities[0] for v in identities) else None


def _to_summary(raw: dict[str, Any]) -> PodSummary:
    raw = api.normalize_pod(raw)
    runtime = raw.get("runtime") if isinstance(raw.get("runtime"), dict) else {}
    machine = raw.get("machine") if isinstance(raw.get("machine"), dict) else {}
    cost = raw.get("costPerHr")
    try:
        cost_val = float(cost) if cost is not None else 0.0
    except (TypeError, ValueError):
        cost_val = 0.0
    legacy_path, legacy_objects, _, _ = _legacy_mount_facts(raw)
    return PodSummary(
        id=str(raw.get("id") or ""),
        name=raw.get("name"),
        desired_status=raw.get("desiredStatus"),
        actual_status=raw.get("actualStatus"),
        gpu_type=raw.get("machineType") or machine.get("gpuDisplayName") or machine.get("gpuTypeId"),
        image=raw.get("imageName"),
        created_at=raw.get("createdAt"),
        cost_per_hr=cost_val,
        uptime_seconds=runtime.get("uptimeInSeconds") if runtime else None,
        ports=runtime.get("ports", []) if runtime else [],
        network_volume_id=_network_volume_id(raw),
        network_volume_evidence=_network_volume_evidence(raw),
        network_volume_mount_path=legacy_path,
        legacy_object_mounts=legacy_objects,
    )


def _list_pods_sync(api_key: str) -> list[dict[str, Any]]:
    return api.list_pods(api_key)


async def list_pods(api_key: str, *, name_prefix: str | None = None) -> list[PodSummary]:
    """Return summaries for every pod on the RunPod account, optionally filtered by name prefix."""
    raw = await asyncio.to_thread(_list_pods_sync, api_key)
    summaries = [_to_summary(p) for p in raw if isinstance(p, dict)]
    if name_prefix:
        summaries = [s for s in summaries if s.name and s.name.startswith(name_prefix)]
    return summaries


async def find_pods(
    api_key: str,
    predicate: Callable[[PodSummary], bool],
    *,
    name_prefix: str | None = None,
) -> list[PodSummary]:
    """Return account pods matching the caller-supplied predicate."""
    summaries = await list_pods(api_key, name_prefix=name_prefix)
    return [s for s in summaries if predicate(s)]


_ACTIVE_STATUSES = {"RUNNING", "PROVISIONING", "STARTING"}


async def find_orphans(
    api_key: str,
    known_pod_ids: Iterable[str],
    *,
    name_prefix: str | None = None,
    older_than_seconds: int | None = None,
) -> list[PodSummary]:
    """Return active pods on the RunPod account that the caller doesn't know about.

    Mirrors the orphan partition logic from
    reigh-worker-orchestrator/scripts/debug/commands/runpod.py:46-70.
    `known_pod_ids` is supplied by the caller (e.g. orchestrator's DB query);
    this package never touches Supabase.
    """
    known = {str(pid) for pid in known_pod_ids if pid}
    summaries = await list_pods(api_key, name_prefix=name_prefix)
    orphans = [s for s in summaries if s.desired_status in _ACTIVE_STATUSES and s.id not in known]
    if older_than_seconds is not None:
        orphans = [
            s for s in orphans if s.uptime_seconds is not None and s.uptime_seconds >= older_than_seconds
        ]
    return orphans


async def get_pod(
    pod_id: str,
    config: RunPodConfig,
    *,
    hooks: EventHooks | None = None,
    name: str | None = None,
) -> Pod:
    """Attach to an existing pod by id. Raises LaunchFailure if it doesn't exist."""
    status = await asyncio.to_thread(api.get_pod_status, pod_id, config.api_key)
    if not status:
        raise LaunchFailure(f"Pod {pod_id} not found on RunPod account")
    pod = Pod(
        pod_id=pod_id,
        name=name or pod_id,
        config=config,
        hooks=hooks,
        ram_tier=0,
        storage_volume=None,
    )
    return pod


async def terminate(
    pod_id: str,
    api_key: str,
    *,
    hooks: EventHooks | None = None,
) -> None:
    """Terminate a pod by id without needing a Pod handle. Raises TerminateError on failure."""
    try:
        await asyncio.to_thread(api.terminate_pod, pod_id, api_key)
    except CleanupPendingError as exc:
        await _emit_error(
            hooks,
            exc,
            {"pod_id": pod_id, "status": exc.status, "cleanup_pending": True},
        )
        raise
    except Exception as exc:
        await _emit_error(hooks, exc, {"pod_id": pod_id, "operation": "terminate"})
        raise TerminateError(f"Failed to terminate pod {pod_id}: {exc}") from exc
    await _emit_state(hooks, pod_id, PodState.TERMINATED, {"source": "discovery.terminate"})


def cost_summary(pods: Iterable[PodSummary]) -> dict[str, float]:
    """Aggregate $/hr across pods. Mirrors _print_orphaned_pods cost math."""
    total = sum(p.cost_per_hr for p in pods)
    return {
        "total_per_hr": round(total, 4),
        "daily": round(total * 24, 2),
        "monthly": round(total * 24 * 30, 2),
    }
