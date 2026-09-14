---
name: runpod-lifecycle
description: Provision, manage, and tear down RunPod GPU instances from Python via the reusable `runpod_lifecycle` package (v0.3.0). The single source of truth for launching pods, waiting for readiness, SSH exec, shipping code + running it + collecting artifacts (`ship_and_run` / `ship_and_run_detached`), capacity probing, multi-GPU/multi-datacenter fan-out, orphan cleanup, and shutdown. Use whenever a tool, script, or agent needs ephemeral GPU compute on RunPod — model inference, training, batch jobs, benchmarks, live test/acceptance suites. ALSO use when the user asks to "spin up a RunPod / 4090 / A100", "shut down pod X", "list my pods", "run this on a GPU", or when extending a service that manages pods. Do not call the raw `runpod` SDK directly in new code — use this package.
---

# RunPod Lifecycle (v0.3.0)

Async RunPod lifecycle primitives — launch a pod, wait until SSH-ready, run commands, ship code + collect artifacts, monitor storage, discover/clean orphans, and terminate cleanly. No orchestrator-specific state; consumers attach `EventHooks` if they want persistence.

Package: `/Users/peteromalley/Documents/reigh-workspace/runpod-lifecycle/` (editable install). It is the **single source of truth** for RunPod provisioning, readiness, idle detection, SSH exec, storage health, and shutdown.

## When to use

- Any new script/service/agent that needs a RunPod GPU.
- "Ship this repo to a pod, run a script, bring back the outputs" → `ship_and_run_detached`.
- Migrating off hand-rolled `runpod` SDK calls.
- Debugging a stuck, orphaned, or storage-full pod.

## When NOT to use

- Non-RunPod providers — use that provider's package.
- You only need to *know* what's launchable right now — use `probe` (creates no pod).

## Install

```bash
pip install -e /Users/peteromalley/Documents/reigh-workspace/runpod-lifecycle
# or, from another project:
pip install "runpod-lifecycle @ git+https://github.com/banodoco/runpod-lifecycle.git@v0.3.0"
```

### Credentials / `.env` (cross-repo gotcha)

Defaults are read from env vars (`RUNPOD_API_KEY`, `RUNPOD_GPU_TYPE`, `RUNPOD_WORKER_IMAGE`, `RUNPOD_SSH_*`, `RUNPOD_STORAGE_NAME`, …). Seed from `runpod-lifecycle/env.example`.

`RunPodConfig.from_env()` calls **bare `load_dotenv()`**, which searches upward from the CWD. If you call it from a *different* repo (e.g. vibecomfy), it will **not** find `runpod-lifecycle/.env`. Load it explicitly with python-dotenv first:

```python
from dotenv import load_dotenv
load_dotenv("/Users/peteromalley/Documents/reigh-workspace/runpod-lifecycle/.env", override=True)
```

Do **not** shell-`source` that file — its unquoted spaced values (e.g. `NVIDIA GeForce RTX 4090`) and multi-line inline SSH keys are mangled by `source`. python-dotenv parses both correctly.

## High-level workhorse: ship → run → collect

For "upload this repo, run a script on a fresh pod, download the outputs, terminate," use the runner — it wraps launch + upload + exec + poll + artifact download + teardown.

```python
from pathlib import Path
from runpod_lifecycle import RunPodConfig, ship_and_run_detached

config = RunPodConfig.from_env()
result = await ship_and_run_detached(
    config,
    remote_script="cd /workspace/job && python train.py",  # bash; run detached, polled to completion
    local_root=Path("."),           # uploaded to remote_root (tarball or sftp_walk)
    remote_root="/workspace/job",
    exclude={".git", ".venv", "__pycache__", "out"},
    upload_mode="tarball",
    timeout=3600,
    name_prefix="myjob",
    terminate_after_exec=True,      # terminate the pod when the script exits
    poll_interval=30,
)
print(result.returncode, result.artifact_root)  # artifacts downloaded under local_root/"artifacts"
```

`ship_and_run_detached` signature (v0.3.0): `config, remote_script, *, pod, local_root, remote_root, exclude, upload_mode, timeout, name_prefix, terminate_after_exec, poll_interval`.

**v0.3.0 API change — do not pass these (they were accepted in v0.2, now rejected):** `guard_factory`, `poll_command_template`, `poll_exit_marker`, `artifact_paths`. The detached path hardcodes sensible defaults — `artifact_paths=["out", "output"]`, exit marker `/tmp/runpod-lifecycle-exit-code`, download to `local_root/"artifacts"` — which are a functional superset of what callers used to pass.

`ship_and_run(...)` is the synchronous (blocking-until-done) sibling; it still accepts `guard_factory`.

### `ShipAndRunResult` fields

`returncode`, `stdout`, `stderr`, `pod` (the `Pod`), `artifact_root` (`Path` to downloaded artifacts, or `None`), `cost_per_hr`, `terminated` (`bool`), `breach_log`, `upload_info`.

### Reattaching to an existing pod

Pass `pod=existing_pod` to `ship_and_run_detached` to skip launch/storage/RAM fallback; `config` is then used only as an API-key source.

## Low-level: launch + Pod API

```python
from runpod_lifecycle import RunPodConfig, launch

pod = await launch(RunPodConfig.from_env(gpu_type="NVIDIA GeForce RTX 4090"))
try:
    await pod.wait_ready(timeout=600)
    code, stdout, stderr = await pod.exec_ssh("nvidia-smi -L", timeout=60)
    if await pod.is_idle(threshold_sec=300):
        ...
finally:
    await pod.terminate()           # ALWAYS terminate in finally
```

- `pod.exec_ssh(cmd, timeout=...)` → `(code, stdout, stderr)`.
- `pod.open_ssh_client()` → connected paramiko client; **caller closes it**.
- `pod.wait_ready(timeout=...)`, `pod.is_idle(threshold_sec=...)`, `pod.terminate()`.
- **Always wrap in `try/finally` and `terminate()`.** RunPod bills by the second; a leaked pod is a real money leak.

## Capacity: find a pod that will actually launch

Supply fluctuates; a single GPU × single volume × single datacenter often shows "no instances available." Cast a wider net.

**Multi-GPU fallback** — `gpu_type` accepts a string or an ordered tuple/list; `launch()` tries each and returns the first that provisions. `RUNPOD_GPU_TYPE` accepts a comma-separated list.

**Multi-storage / multi-datacenter** — `storage_name` is tried first, then each of `storage_volumes`. Because a network volume pins the pod to its datacenter, listing volumes across DCs dramatically widens the GPU pool. `RUNPOD_STORAGE_VOLUMES` is comma-separated.

```python
cfg = RunPodConfig.from_env(
    gpu_type=["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3090", "NVIDIA A40", "NVIDIA L40"],
    storage_name="Peter",
    storage_volumes=("Training", "EU-NO-1", "EU-CZ-1", "EUR-IS-1"),
    ram_tiers=(32, 16),
)
pod = await launch(cfg)             # raises LaunchFailure aggregating every failed combo
```

**Wait for capacity** — `launch_when_available(config, max_wait_sec=, retry_interval_sec=)` retries the exact GPU×RAM×storage matrix for a bounded period (vs. `launch()`, which is one-shot). The detached runner does **not** self-retry on capacity — wrap it in a retry loop catching only `LaunchFailure` (real job failures return an exit code, not an exception).

**Probe without launching** — `runpod-lifecycle probe --min-memory 48 --exclude-blackwell` (or `probe(...)` in Python) returns price-ranked launchable GPU types; creates no pod.

## Discovery & orphan cleanup

```python
from runpod_lifecycle import list_pods, find_orphans, find_pods, get_pod, terminate, cost_summary

for p in await list_pods():
    ...                            # PodSummary: id, name, gpu, status, age_minutes, last_activity_minutes, hourly_rate
print(await cost_summary())        # account-wide $/hr, daily, monthly

for orphan in await find_orphans(max_age_minutes=60):
    # cross-reference your own DB before terminating — a pod with no row
    # may be in-flight from another agent
    ...
```

## CLI

```bash
runpod-lifecycle list
runpod-lifecycle launch --gpu-type "RTX 4090,A40,L40" --storage-volumes "Peter,EU-NO-1" --wait-capacity 900
runpod-lifecycle status <pod-id>
runpod-lifecycle terminate <pod-id>
runpod-lifecycle probe --min-memory 48 --exclude-blackwell --format table
runpod-lifecycle launch --gpu-type "A5000,L4,A40" --storage-volumes "Peter,EU-NO-1" --probe-only   # claim+release to test capacity
```

## Reigh prebuilt validation environment

`rl prebuilt ...` (CLI) manages the reusable RunPod volume used by Reigh/VibeComfy live validation (portable RTX 4090 profile, optional `sage` attention profile). Build/check/status/cleanup a prebuilt env; `check` writes `env.health.json`. See the package README for the full sequence — out of scope for this quick reference.

## Config reference

| field | env var | default | notes |
|---|---|---|---|
| `api_key` | `RUNPOD_API_KEY` | required | used by all SDK/HTTP calls |
| `gpu_type` | `RUNPOD_GPU_TYPE` | `NVIDIA GeForce RTX 4090` | display name or CSV/tuple of candidates |
| `worker_image` | `RUNPOD_WORKER_IMAGE` | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | container image |
| `template_id` | `RUNPOD_TEMPLATE_ID` | `runpod-torch-v240` | RunPod template |
| `volume_mount_path` | `RUNPOD_VOLUME_MOUNT_PATH` | `/workspace` | network-volume mount path |
| `disk_size_gb` | `RUNPOD_DISK_SIZE_GB` | `20` | pod root disk |
| `container_disk_gb` | `RUNPOD_CONTAINER_DISK_GB` | `50` | container/docker disk |
| `min_vcpu_count` | `RUNPOD_MIN_VCPU_COUNT` | `8` | min vCPU |
| `min_memory_gb` | `RUNPOD_MIN_MEMORY_GB` | `32` | floor for RAM fallback |
| `ram_tiers_enabled` | `RUNPOD_RAM_TIERS_ENABLED` | `True` | enable RAM-tier fallback |
| `ram_tiers` | `RUNPOD_RAM_TIERS` | `(72,60,48,32,16)` | ordered tiers; below `min_memory_gb` filtered |
| `storage_name` | `RUNPOD_STORAGE_NAME` | `None` | preferred volume (tried first) |
| `storage_volumes` | `RUNPOD_STORAGE_VOLUMES` | `()` | ordered fallback volumes (multi-DC) |
| `ssh_public_key` / `ssh_private_key` | `RUNPOD_SSH_*` | `None` | inline keys |
| `ssh_public_key_path` / `ssh_private_key_path` | `RUNPOD_SSH_*_PATH` | `None` | key file paths |
| `env_vars` | `RUNPOD_ENV_VARS` | `{}` | JSON object of extra pod env |
| `name_prefix` | `RUNPOD_NAME_PREFIX` | `pod` | pod-name prefix |

If `storage_name` is unset and `storage_volumes` is empty, `launch()` creates a **volumeless** pod (`network_volume_id=None`) — intentional. Per-call overrides win over `from_env()` defaults.

## Cost guardrails

1. **Always `terminate()` in `finally`** (or `terminate_after_exec=True` on the runner). Non-negotiable.
2. **Set a timeout** — `wait_ready(timeout=...)`, runner `timeout=...`. Default 1800/3600.
3. **Prefer the cheapest GPU that fits** — don't default to A100 when a 4090/3090/A40 will do; broaden with multi-GPU fallback rather than jumping tiers.
4. **Fan across DCs** when one datacenter is dry — `storage_volumes` across datacenters.
5. **Use `is_idle()` for long-running services** — don't keep a pod alive between jobs unless warm-start savings beat idle cost.
6. **Watch storage** — the package monitors free disk and can auto-expand; frequent expansions mean the job is leaking files.

## Scope / extraction notes

Extracted from `reigh-worker-orchestrator/gpu_orchestrator/runpod/` (RAM-tier fallback, dual SDK + GraphQL SSH-detail fetch, storage expansion, startup-script SSH injection). Behavior parity is the contract — file divergences as bugs against the package, not workarounds in callers. Does not include `startup_script.py`, `check_worker_startup_status`, or any persistence layer; use `EventHooks(on_state_change=..., on_error=...)` to persist state to your own store.
