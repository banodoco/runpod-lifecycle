# runpod-lifecycle

`runpod_lifecycle` is a small async package for the full RunPod pod lifecycle: launch a pod, wait until SSH is ready, run commands, inspect status and idleness, monitor storage health, and terminate cleanly without pulling in orchestrator-specific state management.

## Install

```bash
pip install -e .[dev]
```

From another project, install directly from GitHub instead of relying on a
sibling checkout:

```bash
pip install "runpod-lifecycle @ git+https://github.com/banodoco/runpod-lifecycle.git@v0.1.1"
```

## Environment Variables

`RunPodConfig.from_env()` reads these variables:

- `RUNPOD_API_KEY`
- `RUNPOD_GPU_TYPE`
- `RUNPOD_WORKER_IMAGE`
- `RUNPOD_TEMPLATE_ID`
- `RUNPOD_VOLUME_MOUNT_PATH`
- `RUNPOD_DISK_SIZE_GB`
- `RUNPOD_CONTAINER_DISK_GB`
- `RUNPOD_MIN_VCPU_COUNT`
- `RUNPOD_MIN_MEMORY_GB`
- `RUNPOD_RAM_TIERS_ENABLED`
- `RUNPOD_RAM_TIER_FALLBACK` (legacy alias for `RUNPOD_RAM_TIERS_ENABLED`)
- `RUNPOD_RAM_TIERS`
- `RUNPOD_STORAGE_VOLUMES`
- `RUNPOD_STORAGE_NAME`
- `RUNPOD_SSH_PUBLIC_KEY`
- `RUNPOD_SSH_PRIVATE_KEY`
- `RUNPOD_SSH_PUBLIC_KEY_PATH`
- `RUNPOD_SSH_PRIVATE_KEY_PATH`
- `RUNPOD_ENV_VARS`
- `RUNPOD_NAME_PREFIX`

## Quick Start

```python
import asyncio

from runpod_lifecycle import RunPodConfig, launch, EventHooks


async def on_state(event):
    print(f"{event.state}: pod_id={event.pod_id} detail={event.detail}")


async def main() -> None:
    cfg = RunPodConfig.from_env(
        storage_name="my-network-volume",
    )
    hooks = EventHooks(on_state_change=on_state)

    pod = await launch(cfg, hooks=hooks)
    await pod.wait_ready(timeout=600)

    exit_code, stdout, stderr = await pod.exec_ssh("nvidia-smi -L", timeout=60)
    print(exit_code)
    print(stdout)
    print(stderr)

    await pod.terminate()


asyncio.run(main())
```

If `storage_name` is unset and `storage_volumes` is empty, `launch()` creates a volumeless pod by passing `network_volume_id=None`. That is intentional new behavior in this package.

### Multi-GPU fallback

`gpu_type` accepts a single string (backwards-compatible) or an ordered list of
candidates. `launch()` tries each in turn and returns the first that provisions;
if every candidate is exhausted, a `LaunchFailure` aggregates the per-candidate
reasons. The env var `RUNPOD_GPU_TYPE` accepts a comma-separated list.

```python
cfg = RunPodConfig.from_env(
    gpu_type=[
        "NVIDIA RTX 6000 Ada Generation",
        "NVIDIA RTX A6000",
        "NVIDIA L40S",
    ],
)
pod = await launch(cfg)
```

For direct file transport or other low-level SSH work, `Pod.open_ssh_client()` returns a connected `paramiko`-compatible client. Callers are responsible for closing the returned client when they are done with it.

### Detached runs on an existing pod

`ship_and_run_detached()` can either provision a fresh pod from `config` or
reattach to a `Pod` that was launched earlier. Supplying `pod=` skips launch,
storage attach, and RAM-tier fallback entirely; `config`, when also supplied,
is treated only as an API-key source for operations on that pod.

```python
result = await ship_and_run_detached(
    pod=existing_pod,
    remote_script="python train.py --resume",
    local_root=Path("payload"),
    remote_root="/workspace/job",
    exclude={".git", "__pycache__"},
    terminate_after_exec=False,
)
```

## Probing availability

Before launching, ask RunPod what is actually launchable right now — no pod is created:

```bash
runpod-lifecycle probe --min-memory 48 --exclude-blackwell --format table
```

```python
import asyncio, os
from runpod_lifecycle import probe

async def main():
    options = await probe(
        api_key=os.environ["RUNPOD_API_KEY"],
        min_memory_gb=48,
        exclude_blackwell=True,
    )
    for o in options[:3]:
        print(o["gpu_type"], o["price_per_hour"])

asyncio.run(main())
```

Results are price-ranked; GPU types without current secure-cloud availability are dropped.

## Reigh Prebuilt Validation Environment

`rl prebuilt ...` manages the reusable RunPod volume used by Reigh VibeComfy
live validation. The default target is a portable RTX 4090 profile. Larger GPU
types can be requested with `--gpu-type`, and `--attention-profile sage` is an
explicit opt-in profile; SageAttention failures must not block the portable
profile.

Canonical volume names are derived from the actual RunPod `dataCenterId`:

```text
reigh-livetest-prebuilt-<attention-profile>-<normalized-data-center-id>
```

For example, a RunPod `dataCenterId` of `EUR-NO-1` maps to
`reigh-livetest-prebuilt-portable-eur-no-1`. Treat that as an example, not a
hard-coded region.

Common sequence:

```bash
# Build or refresh a portable volume in a chosen RunPod data center.
rl prebuilt build --data-center <DATA_CENTER_ID> --attention-profile portable

# Cheap no-credential contract checks. These must not launch pods.
rl prebuilt check --data-center <DATA_CENTER_ID> --dry-run
rl prebuilt status --dry-run
rl prebuilt cleanup --dry-run

# Emit selected targets in reigh-worker, then enrich them in VibeComfy.
python -m scripts.live_test.main --variant fresh --backend vibecomfy \
  --case z_image_turbo --emit-targets-json /tmp/reigh-targets.json
python -m vibecomfy.cli workflows enrich-targets \
  --targets-json /tmp/reigh-targets.json \
  --output /tmp/reigh-targets.enriched.json \
  --models-root /workspace/reigh-livetest-prebuilt/models

# Check the real prebuilt environment before expensive workflow runs.
rl prebuilt check --data-center <DATA_CENTER_ID> \
  --enriched-targets-json /tmp/reigh-targets.enriched.json
```

`check` writes `env.health.json` on the prebuilt volume. Its grouped issues
separate environment setup, custom nodes, workflow source, schema, assets, and
runtime-deferred work. Missing model diagnostics include the asset name,
category/path, paths checked, URL when known, and remediation text.

Reconcile behavior is deliberately conservative:

```bash
# Plain target JSON does not contain enough asset metadata; this prints the
# enrichment command instead of guessing.
rl prebuilt reconcile --data-center <DATA_CENTER_ID> --dry-run \
  --targets-json /tmp/reigh-targets.json

# Enriched targets or an explicit asset manifest may produce a fetch plan.
rl prebuilt reconcile --data-center <DATA_CENTER_ID> --dry-run \
  --enriched-targets-json /tmp/reigh-targets.enriched.json
rl prebuilt reconcile --data-center <DATA_CENTER_ID> \
  --enriched-targets-json /tmp/reigh-targets.enriched.json
```

`status` and `cleanup` only operate on validation pod prefixes
`reigh-livetest-builder-` and `reigh-livetest-prebuilt-`. `cleanup` requires
`--yes` outside `--dry-run` and must never be used for unrelated user pods.
Use `rl prebuilt list` to inspect matching network volumes and
`rl prebuilt invalidate --data-center <DATA_CENTER_ID> --attention-profile portable`
when the bundle contract should be rebuilt from scratch.

## Config Reference

| field | env var | default | description |
| --- | --- | --- | --- |
| `api_key` | `RUNPOD_API_KEY` | required | RunPod API key used by all SDK and HTTP calls. |
| `gpu_type` | `RUNPOD_GPU_TYPE` | `NVIDIA GeForce RTX 4090` | Display name used by `find_gpu_type()` before pod creation. |
| `worker_image` | `RUNPOD_WORKER_IMAGE` | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | Container image passed to RunPod at launch time. |
| `template_id` | `RUNPOD_TEMPLATE_ID` | `runpod-torch-v240` | RunPod template identifier used when creating the pod. |
| `volume_mount_path` | `RUNPOD_VOLUME_MOUNT_PATH` | `/workspace` | Mount path for an attached network volume inside the container. |
| `disk_size_gb` | `RUNPOD_DISK_SIZE_GB` | `20` | Root disk size requested for the pod. |
| `container_disk_gb` | `RUNPOD_CONTAINER_DISK_GB` | `50` | Container disk size requested for the pod. |
| `min_vcpu_count` | `RUNPOD_MIN_VCPU_COUNT` | `8` | Minimum vCPU count passed to RunPod when launching. |
| `min_memory_gb` | `RUNPOD_MIN_MEMORY_GB` | `32` | Lowest RAM target allowed for launch fallback. |
| `ram_tiers_enabled` | `RUNPOD_RAM_TIERS_ENABLED` | `True` | Enables RAM-tier fallback instead of launching only at `min_memory_gb`. |
| `ram_tiers` | `RUNPOD_RAM_TIERS` | `(72, 60, 48, 32, 16)` | Ordered RAM fallback tiers; values below `min_memory_gb` are filtered out. |
| `storage_volumes` | `RUNPOD_STORAGE_VOLUMES` | `()` | Ordered fallback list of storage names to resolve and try after `storage_name`. |
| `storage_name` | `RUNPOD_STORAGE_NAME` | `None` | Preferred storage name to try first before `storage_volumes`. |
| `ssh_public_key` | `RUNPOD_SSH_PUBLIC_KEY` | `None` | Inline SSH public key string injected into the pod environment. |
| `ssh_private_key` | `RUNPOD_SSH_PRIVATE_KEY` | `None` | Inline private key used by `Pod.exec_ssh()`. |
| `ssh_public_key_path` | `RUNPOD_SSH_PUBLIC_KEY_PATH` | `None` | Filesystem path to the public key if not provided inline. |
| `ssh_private_key_path` | `RUNPOD_SSH_PRIVATE_KEY_PATH` | `None` | Filesystem path to the private key if not provided inline. |
| `env_vars` | `RUNPOD_ENV_VARS` | `{}` | JSON object of extra environment variables sent when the pod is created. |
| `name_prefix` | `RUNPOD_NAME_PREFIX` | `pod` | Prefix used for generated pod names when `launch(..., name=...)` is not provided. |

## Scope Notes

This package does not include `startup_script.py`, `check_worker_startup_status`, or any orchestrator-owned persistence layer. Consumers that need to persist lifecycle state should attach `EventHooks(on_state_change=..., on_error=...)` and write to their own database or control plane there.
