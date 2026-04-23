"""Event hook primitives for consumers that persist pod state externally."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable


class PodState(str, Enum):
    PROVISIONING = "PROVISIONING"
    STARTING = "STARTING"
    READY = "READY"
    STOPPED = "STOPPED"
    FAILED = "FAILED"
    TERMINATED = "TERMINATED"


@dataclass(slots=True)
class PodEvent:
    pod_id: str | None
    state: PodState
    detail: dict[str, Any] = field(default_factory=dict)


StateHook = Callable[[PodEvent], Awaitable[None] | None]
ErrorHook = Callable[[Exception, dict[str, Any]], Awaitable[None] | None]


@dataclass(slots=True)
class EventHooks:
    on_state_change: StateHook | None = None
    on_error: ErrorHook | None = None


async def _maybe_await(result: object) -> None:
    if inspect.iscoroutine(result):
        await result


async def _emit_state(
    hooks: EventHooks | None,
    pod_id: str | None,
    state: PodState,
    detail: dict[str, Any] | None = None,
) -> None:
    if hooks is None or hooks.on_state_change is None:
        return
    result = hooks.on_state_change(PodEvent(pod_id=pod_id, state=state, detail=detail or {}))
    await _maybe_await(result)


async def _emit_error(
    hooks: EventHooks | None,
    error: Exception,
    detail: dict[str, Any] | None = None,
) -> None:
    if hooks is None or hooks.on_error is None:
        return
    result = hooks.on_error(error, detail or {})
    await _maybe_await(result)
