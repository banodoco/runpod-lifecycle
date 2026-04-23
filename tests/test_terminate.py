from __future__ import annotations

import asyncio

import pytest

from runpod_lifecycle.errors import TerminateError
from runpod_lifecycle.events import EventHooks, PodState
from runpod_lifecycle.pod import Pod


def test_pod_terminate_emits_terminated_event(base_config, monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict[str, object]]] = []

    async def on_state(event) -> None:  # type: ignore[no-untyped-def]
        events.append((event.state.value, event.detail))

    pod = Pod("pod-1", "worker", base_config, hooks=EventHooks(on_state_change=on_state))
    monkeypatch.setattr("runpod_lifecycle.pod.api.terminate_pod", lambda pod_id, api_key: None)

    asyncio.run(pod.terminate())

    assert events == [(PodState.TERMINATED.value, {"pod_id": "pod-1"})]


def test_pod_terminate_raises_and_emits_error_once(base_config, monkeypatch: pytest.MonkeyPatch) -> None:
    error_calls: list[tuple[str, dict[str, object]]] = []
    call_order: list[str] = []

    def on_error(error: Exception, detail: dict[str, object]) -> None:
        call_order.append("on_error")
        error_calls.append((str(error), detail))

    pod = Pod("pod-1", "worker", base_config, hooks=EventHooks(on_error=on_error))

    def fail_terminate(pod_id: str, api_key: str) -> None:
        raise RuntimeError("terminate boom")

    monkeypatch.setattr("runpod_lifecycle.pod.api.terminate_pod", fail_terminate)

    with pytest.raises(TerminateError):
        asyncio.run(pod.terminate())

    assert error_calls == [("terminate boom", {"pod_id": "pod-1"})]
    assert call_order == ["on_error"]
