"""PodGuard watchdog and signal-handler setup for RunPod lifecycle."""

from __future__ import annotations

import asyncio
import os
import re
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Awaitable, Callable, Iterable

if TYPE_CHECKING:
    from .pod import Pod


_DEFAULT_STALE_POD_AGE_SEC = 6 * 60 * 60


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


@dataclass(frozen=True)
class StalePodCleanupResult:
    """Outcome of pruning stale live-test pods grouped by name prefix."""

    inspected: int
    stale: tuple[str, ...]
    terminated: tuple[str, ...]
    failed: tuple[tuple[str, str], ...]


def _compile_prefix_patterns(prefixes: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    return tuple(
        re.compile(rf"^{re.escape(prefix)}(\d{{8}})t(\d{{6}})z$") for prefix in prefixes
    )


def _pod_id_of(pod: object) -> str:
    return str(getattr(pod, "id", "") or "")


def _pod_name_of(pod: object) -> str | None:
    name = getattr(pod, "name", None)
    return str(name) if name else None


def _age_from_iso_timestamp(value: str | None, now: datetime) -> int | None:
    if not value:
        return None
    try:
        created = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return max(0, int((now - created).total_seconds()))


def _age_from_pod_name(
    name: str | None,
    patterns: tuple[re.Pattern[str], ...],
    now: datetime,
) -> int | None:
    if not name:
        return None
    for pattern in patterns:
        match = pattern.fullmatch(name)
        if not match:
            continue
        created = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(
            tzinfo=timezone.utc
        )
        return max(0, int((now - created).total_seconds()))
    return None


def _pod_age_seconds(
    pod: object, patterns: tuple[re.Pattern[str], ...], now: datetime
) -> int:
    uptime = getattr(pod, "uptime_seconds", None)
    if isinstance(uptime, int) and uptime >= 0:
        return uptime
    name_age = _age_from_pod_name(_pod_name_of(pod), patterns, now)
    if name_age is not None:
        return name_age
    created_age = _age_from_iso_timestamp(getattr(pod, "created_at", None), now)
    return created_age if created_age is not None else 0


async def _default_list_pods(api_key: str, prefix: str) -> Iterable[object]:
    from . import discovery

    return await discovery.list_pods(api_key, name_prefix=prefix)


async def _default_terminate(api_key: str, pod_id: str) -> None:
    from . import discovery

    await discovery.terminate(pod_id, api_key)


async def _prune_pods_by_prefix_async(
    prefixes: tuple[str, ...],
    api_key: str,
    *,
    stale_age_sec: int,
    list_pods_fn: Callable[[str, str], Awaitable[Iterable[object]]],
    terminate_fn: Callable[[str, str], Awaitable[None]],
    now: datetime,
) -> StalePodCleanupResult:
    patterns = _compile_prefix_patterns(prefixes)
    inspected_total = 0
    stale_ids: list[str] = []
    seen_ids: set[str] = set()
    for prefix in prefixes:
        pods = list(await list_pods_fn(api_key, prefix))
        inspected_total += len(pods)
        for pod in pods:
            pod_id = _pod_id_of(pod)
            if not pod_id or pod_id in seen_ids:
                continue
            if _pod_age_seconds(pod, patterns, now) >= stale_age_sec:
                seen_ids.add(pod_id)
                stale_ids.append(pod_id)
    terminated: list[str] = []
    failed: list[tuple[str, str]] = []
    for pod_id in stale_ids:
        try:
            await terminate_fn(api_key, pod_id)
            terminated.append(pod_id)
        except Exception as exc:  # noqa: BLE001 — caller surfaces aggregated errors
            failed.append((pod_id, str(exc)))
    return StalePodCleanupResult(
        inspected=inspected_total,
        stale=tuple(stale_ids),
        terminated=tuple(terminated),
        failed=tuple(failed),
    )


def prune_pods_by_prefix(
    prefixes: tuple[str, ...],
    api_key: str | None,
    *,
    stale_age_sec: int = _DEFAULT_STALE_POD_AGE_SEC,
    list_pods_fn: Callable[[str, str], Awaitable[Iterable[object]]] | None = None,
    terminate_fn: Callable[[str, str], Awaitable[None]] | None = None,
    now: datetime | None = None,
) -> StalePodCleanupResult:
    """Terminate live-test pods whose names match any of *prefixes* and have aged past *stale_age_sec*.

    Returns the same shape as the reigh-worker ``prune_stale_live_test_pods``
    helper that wraps this function. Honors the ``REIGH_LIVE_TEST_SKIP_STALE_POD_CLEANUP``
    opt-out and missing-credential cases by returning an empty result.
    """
    if not api_key or not prefixes:
        return StalePodCleanupResult(0, (), (), ())
    if os.getenv("REIGH_LIVE_TEST_SKIP_STALE_POD_CLEANUP") == "1":
        return StalePodCleanupResult(0, (), (), ())
    return asyncio.run(
        _prune_pods_by_prefix_async(
            prefixes,
            api_key,
            stale_age_sec=stale_age_sec,
            list_pods_fn=list_pods_fn or _default_list_pods,
            terminate_fn=terminate_fn or _default_terminate,
            now=now or datetime.now(timezone.utc),
        )
    )


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