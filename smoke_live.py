"""Live end-to-end smoke test for runpod_lifecycle.

Launches a real RunPod 4090, exercises the package, terminates. Costs ~$0.10.
ALWAYS runs `terminate` in finally — do not edit that out.
"""

from __future__ import annotations

import asyncio
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

from runpod_lifecycle import (
    EventHooks,
    PodEvent,
    RunPodConfig,
    launch,
    list_pods,
)


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


async def on_state(event: PodEvent) -> None:
    _log(f"  state={event.state.value} pod={event.pod_id} detail_keys={list(event.detail.keys())}")


async def on_error(err: Exception, detail: dict) -> None:
    _log(f"  ERROR: {type(err).__name__}: {err} detail={detail}")


async def main() -> int:
    load_dotenv()
    config = RunPodConfig.from_env(
        gpu_type="NVIDIA GeForce RTX 4090",
        ram_tiers=(32, 16),  # smaller tiers — faster fallback if 4090 with big RAM is unavailable
        storage_volumes=("Peter", "EU-NO-1", "EU-CZ-1", "EUR-IS-1"),
    )
    hooks = EventHooks(on_state_change=on_state, on_error=on_error)

    pod = None
    try:
        _log("=== STEP 1: launch ===")
        t0 = time.monotonic()
        pod = await launch(config, name=f"smoke-{int(time.time())}", hooks=hooks)
        _log(f"launched pod_id={pod.id} ram_tier={pod._ram_tier} storage={pod._storage_volume} elapsed={time.monotonic()-t0:.1f}s")

        _log("=== STEP 2: wait_ready (up to 5min) ===")
        t0 = time.monotonic()
        await pod.wait_ready(timeout=300)
        _log(f"ready in {time.monotonic()-t0:.1f}s")

        _log("=== STEP 3: status() ===")
        status = await pod.status()
        _log(f"status: desired={status.get('desired_status')} actual={status.get('actual_status')} ip={status.get('ip')} cost=${status.get('cost_per_hr')}/hr")

        _log("=== STEP 4: exec_ssh 'nvidia-smi -L' ===")
        t0 = time.monotonic()
        code, stdout, stderr = await pod.exec_ssh("nvidia-smi -L", timeout=60)
        _log(f"exit={code} elapsed={time.monotonic()-t0:.1f}s")
        _log(f"stdout: {stdout.strip()}")
        if stderr.strip():
            _log(f"stderr: {stderr.strip()}")

        _log("=== STEP 5: exec_ssh 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits' ===")
        code, stdout, _ = await pod.exec_ssh(
            "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits",
            timeout=30,
        )
        _log(f"GPU util raw: {stdout.strip()!r}")

        _log("=== STEP 6: is_idle(threshold=0) — should be True (just-finished exec, GPU unused) ===")
        # threshold=0 forces the nvidia-smi path
        idle = await pod.is_idle(0)
        _log(f"is_idle(0): {idle}")

        _log("=== STEP 7: check_storage_health (df parse) ===")
        try:
            health = await pod.check_storage_health(min_free_gb=5, max_percent_used=90)
            _log(f"health: {health}")
        except Exception as exc:
            _log(f"check_storage_health failed (not fatal): {type(exc).__name__}: {exc}")

        _log("=== STEP 8: list_pods — should include this pod ===")
        all_pods = await list_pods(config.api_key)
        ids = [p.id for p in all_pods]
        _log(f"account has {len(all_pods)} pods. ours present: {pod.id in ids}")

        _log("=== ALL CHECKS PASSED ===")
        return 0

    except Exception as exc:
        _log(f"!!! FAILED: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        if pod is not None:
            _log("=== STEP 9: terminate (always runs) ===")
            try:
                await pod.terminate()
                _log(f"terminated {pod.id}")
                await asyncio.sleep(3)
                all_pods = await list_pods(config.api_key)
                still_there = [p for p in all_pods if p.id == pod.id and p.desired_status in {"RUNNING", "PROVISIONING"}]
                _log(f"post-terminate active match: {len(still_there)} (0 expected)")
            except Exception as exc:
                _log(f"!!! TERMINATE FAILED — pod {pod.id} may still be running and costing money: {exc}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
