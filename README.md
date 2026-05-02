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

For direct file transport or other low-level SSH work, `Pod.open_ssh_client()` returns a connected `paramiko`-compatible client. Callers are responsible for closing the returned client when they are done with it.

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
