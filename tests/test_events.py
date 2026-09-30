from __future__ import annotations

import asyncio

from runpod_lifecycle.errors import LaunchFailure
from runpod_lifecycle.events import EventHooks, PodState, _emit_error, _emit_state


class _MarkerAwaitable:
    def __init__(self, marker: list[str], value: str) -> None:
        self.marker = marker
        self.value = value

    def __await__(self):
        async def mark() -> None:
            self.marker.append(self.value)

        return mark().__await__()


def test_event_dispatch_awaits_generic_awaitables() -> None:
    marker: list[str] = []

    def on_state(event):  # type: ignore[no-untyped-def]
        return _MarkerAwaitable(marker, event.state.value)

    def on_error(error, detail):  # type: ignore[no-untyped-def]
        return _MarkerAwaitable(marker, str(error))

    async def exercise() -> None:
        hooks = EventHooks(on_state_change=on_state, on_error=on_error)
        await _emit_state(hooks, "pod-1", PodState.READY)
        await _emit_error(hooks, LaunchFailure("hook test"))

    asyncio.run(exercise())

    assert marker == ["READY", "hook test"]
