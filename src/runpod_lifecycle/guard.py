"""PodGuard watchdog and signal-handler setup for RunPod lifecycle."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pod import Pod


class PodGuard:
    """Watchdog that auto-terminates a pod after max_runtime_seconds.

    When *auto_terminate* is ``False`` the watchdog still detects the
    breach but emits a warning and appends to ``breach_log`` instead of
    terminating – the caller is responsible for teardown.
    """

    def __init__(
        self,
        *,
        name_prefix: str,
        max_runtime_seconds_env: str = "VIBECOMFY_RUNPOD_MAX_RUNTIME_SECONDS",
        default_max_runtime_seconds: int = 7200,
        auto_terminate: bool = True,
    ) -> None:
        self.name_prefix = name_prefix
        self.max_runtime_seconds_env = max_runtime_seconds_env
        self.default_max_runtime_seconds = default_max_runtime_seconds
        self.auto_terminate = auto_terminate
        self.breach_log: list[dict] = []
        self.pod: Pod | None = None
        self._watchdog: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def attach(self, pod: Pod) -> None:
        """Bind *pod* and start the runtime watchdog."""
        self.pod = pod
        self._start_watchdog()

    async def terminate(self) -> None:
        """Cancel the watchdog and terminate the bound pod (idempotent)."""
        if self._watchdog is not None:
            self._watchdog.cancel()
            self._watchdog = None
        if self.pod is not None:
            try:
                await self.pod.terminate()
            except Exception as exc:
                if "not found" in str(exc).lower():
                    print(f"pod_already_terminated={self.pod.id}", flush=True)
                else:
                    raise

    # ------------------------------------------------------------------
    # Internal watchdog
    # ------------------------------------------------------------------

    def _start_watchdog(self) -> None:
        seconds = int(
            os.getenv(self.max_runtime_seconds_env, str(self.default_max_runtime_seconds))
        )
        self._watchdog = asyncio.create_task(self._terminate_after(seconds))

    async def _terminate_after(self, seconds: int) -> None:
        try:
            await asyncio.sleep(seconds)
            if self.pod is not None:
                if self.auto_terminate:
                    print(f"watchdog_terminating_pod={self.pod.id}", flush=True)
                    await self.pod.terminate()
                else:
                    breach_entry = {
                        "pod_id": self.pod.id,
                        "max_runtime_seconds": seconds,
                        "breached_at": datetime.now(timezone.utc).isoformat(),
                    }
                    self.breach_log.append(breach_entry)
                    print(
                        f"watchdog_runtime_breach pod={self.pod.id} "
                        f"max_runtime_seconds={seconds} auto_terminate=false",
                        flush=True,
                    )
        except asyncio.CancelledError:
            return


def install_signal_handlers(loop) -> asyncio.Event:
    """Install SIGINT/SIGTERM handlers that set an event + cancel current task.

    Returns an ``asyncio.Event`` that callers can ``await`` to detect
    early-exit requests.
    """
    cancel_event = asyncio.Event()
    task = asyncio.current_task(loop=loop)
    signal_count = 0

    def _handle_signal(sig: signal.Signals) -> None:
        nonlocal signal_count
        signal_count += 1
        print(f"interrupt_requested signal={sig.name} cleanup=true", flush=True)
        cancel_event.set()
        if task is not None and not task.done():
            task.cancel()
        if signal_count > 1:
            print("interrupt_repeated=true cleanup_still_in_progress=true", flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal, sig)
        except (NotImplementedError, RuntimeError):
            previous = signal.getsignal(sig)

            def _fallback(signum, frame, *, previous=previous) -> None:
                loop.call_soon_threadsafe(_handle_signal, signal.Signals(signum))
                if callable(previous) and previous not in {signal.SIG_DFL, signal.SIG_IGN}:
                    previous(signum, frame)

            signal.signal(sig, _fallback)
    return cancel_event