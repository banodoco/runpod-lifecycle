"""Account-wide RunPod discovery, attach-by-id, orphan detection, and module-level termination."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from . import api
from .config import RunPodConfig
from .errors import LaunchFailure, TerminateError
from .events import EventHooks, PodState, _emit_error, _emit_state
from .pod import Pod

logger = logging.getLogger("runpod_lifecycle.discovery")


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


def _to_summary(raw: dict[str, Any]) -> PodSummary:
    runtime = raw.get("runtime") if isinstance(raw.get("runtime"), dict) else {}
    machine = raw.get("machine") if isinstance(raw.get("machine"), dict) else {}
    cost = raw.get("costPerHr")
    try:
        cost_val = float(cost) if cost is not None else 0.0
    except (TypeError, ValueError):
        cost_val = 0.0
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
        network_volume_id=raw.get("networkVolumeId"),
    )


def _list_pods_sync(api_key: str) -> list[dict[str, Any]]:
    sdk = api._get_runpod()
    sdk.api_key = api_key
    pods = sdk.get_pods()
    return list(pods) if pods else []


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


_ACTIVE_STATUSES = {"RUNNING", "PROVISIONING"}


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
