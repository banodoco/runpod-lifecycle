"""Command-line interface for runpod-lifecycle: launch, exec, ship, fetch, run, volumes, prebuilt, and legacy list/status/terminate/find-orphans/gpu-types."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
from dataclasses import asdict, is_dataclass
import json
import os
import re
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from . import api, config as cfg, discovery
from .config import RunPodConfig
from .guard import PodGuard, install_signal_handlers
from .lifecycle import find_gpu_type, get_network_volumes, launch as _launch, launch_when_available as _launch_when_available
from .pod import Pod
from .prebuilt import (
    PrebuiltEnvContract,
    PrebuiltManifest,
    acquire_build_lock,
    build_error_health_report,
    build_missing_manifest_health_report,
    compute_lockfile_hash,
    compute_pyproject_hash,
    health_path,
    manifest_path,
    prebuilt_volume_name_for_profile,
    read_manifest,
    run_prebuilt_health_probes,
    select_prebuilt_volume,
    write_health_report,
    write_manifest,
)
from .probe import probe as _probe
from .ssh import SSHClient


def _resolve_api_key(args: argparse.Namespace) -> str:
    key = getattr(args, "api_key", None) or os.getenv("RUNPOD_API_KEY")
    if not key:
        print("error: RUNPOD_API_KEY not set (use --api-key or .env)", file=sys.stderr)
        sys.exit(2)
    return key


def _coalesce_blank(value: str | None) -> str | None:
    """Return ``None`` for missing-or-blank strings; pass real values through.

    ``os.getenv`` returns ``""`` when an env var is set to the empty string,
    which then falsely propagates as "set" through the rest of the launch
    pipeline (e.g. ``RUNPOD_STORAGE_NAME=""`` would attempt to resolve a
    blank storage name). Coalesce here so downstream code can keep using
    ``if storage_name`` truthiness.
    """
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _parse_csv_values(value: str | None) -> tuple[str, ...]:
    if value is None or value.strip() == "":
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_csv_ints(value: str | None) -> tuple[int, ...]:
    if value is None or value.strip() == "":
        return ()
    values: list[int] = []
    for part in value.split(","):
        stripped = part.strip()
        if stripped:
            values.append(int(stripped))
    return tuple(values)


def _unique_values(values: list[str | None] | tuple[str | None, ...]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _resolve_storage_volumes(args: argparse.Namespace) -> tuple[str, ...]:
    cli_values = _parse_csv_values(getattr(args, "storage_volumes", None))
    env_values = _parse_csv_values(os.getenv("RUNPOD_STORAGE_VOLUMES"))
    return cli_values or env_values


def _resolve_gpu_type(args: argparse.Namespace) -> str | tuple[str, ...]:
    cli_values = _parse_csv_values(getattr(args, "gpu_type", None))
    if len(cli_values) == 1:
        return cli_values[0]
    if cli_values:
        return cli_values
    return os.getenv("RUNPOD_GPU_TYPE", cfg.DEFAULT_GPU_TYPE)


def _resolve_config(args: argparse.Namespace) -> RunPodConfig:
    api_key = _resolve_api_key(args)
    arg_storage = _coalesce_blank(getattr(args, "storage_name", None))
    env_storage = _coalesce_blank(os.getenv("RUNPOD_STORAGE_NAME"))
    storage_volumes = _resolve_storage_volumes(args)
    ram_tiers = _parse_csv_ints(getattr(args, "ram_tiers", None)) or _parse_csv_ints(
        os.getenv("RUNPOD_RAM_TIERS")
    ) or cfg.DEFAULT_RAM_TIERS
    return RunPodConfig(
        api_key=api_key,
        gpu_type=_resolve_gpu_type(args),
        worker_image=getattr(args, "image", None)
        or os.getenv("RUNPOD_WORKER_IMAGE", cfg.DEFAULT_WORKER_IMAGE),
        container_disk_gb=getattr(args, "container_disk_gb", None)
        or int(os.getenv("RUNPOD_CONTAINER_DISK_GB", "200")),
        name_prefix=getattr(args, "name_prefix", None)
        or os.getenv("RUNPOD_NAME_PREFIX", "pod"),
        disk_size_gb=getattr(args, "disk_size_gb", None)
        or int(os.getenv("RUNPOD_DISK_SIZE_GB", "200")),
        min_memory_gb=getattr(args, "min_memory_gb", None)
        or int(os.getenv("RUNPOD_MIN_MEMORY_GB", "32")),
        ram_tiers=ram_tiers,
        storage_name=arg_storage or env_storage,
        storage_volumes=storage_volumes,
    )


def _parse_duration(text: str) -> int:
    m = re.fullmatch(r"\s*(\d+)\s*([smhd]?)\s*", text)
    if not m:
        raise argparse.ArgumentTypeError(f"invalid duration: {text!r}")
    n, unit = int(m.group(1)), m.group(2) or "s"
    return n * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]


def _summary_to_row(s: discovery.PodSummary) -> list[str]:
    return [
        s.id,
        s.name or "-",
        s.desired_status or "-",
        s.gpu_type or "-",
        f"{s.uptime_seconds}s" if s.uptime_seconds is not None else "-",
        f"${s.cost_per_hr:.3f}/hr",
    ]


def _print_table(rows: list[list[str]], headers: list[str]) -> None:
    if not rows:
        print("(no pods)")
        return
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for r in rows:
        print(fmt.format(*r))


def _print_cost(summaries: list[discovery.PodSummary]) -> None:
    cost = discovery.cost_summary(summaries)
    print(
        f"\nTotal: ${cost['total_per_hr']:.3f}/hr  "
        f"(daily ${cost['daily']:.2f}, monthly ${cost['monthly']:.2f})"
    )


# ---------------------------------------------------------------------------
# Async handlers for each subcommand
# ---------------------------------------------------------------------------


async def _cmd_list(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    pods = await discovery.list_pods(api_key, name_prefix=args.name_prefix)
    if args.json:
        print(json.dumps([p.__dict__ for p in pods], default=str, indent=2))
        return 0
    _print_table(
        [_summary_to_row(p) for p in pods],
        ["ID", "NAME", "STATUS", "GPU", "UPTIME", "COST"],
    )
    _print_cost(pods)
    return 0


async def _cmd_status(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    status = await asyncio.to_thread(api.get_pod_status, args.pod_id, api_key)
    if not status:
        print(f"pod {args.pod_id} not found", file=sys.stderr)
        return 1
    print(json.dumps(status, default=str, indent=2))
    return 0


async def _cmd_terminate(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    if not args.yes:
        confirm = input(f"terminate pod {args.pod_id}? [y/N] ").strip().lower()
        if confirm != "y":
            print("aborted")
            return 1
    await discovery.terminate(args.pod_id, api_key)
    print(f"terminated {args.pod_id}")
    return 0


async def _cmd_find_orphans(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    known: list[str] = []
    if args.known_ids_file:
        with open(args.known_ids_file) as f:
            known = [line.strip() for line in f if line.strip()]
    orphans = await discovery.find_orphans(
        api_key,
        known,
        name_prefix=args.name_prefix,
        older_than_seconds=args.older_than,
    )
    _print_table(
        [_summary_to_row(p) for p in orphans],
        ["ID", "NAME", "STATUS", "GPU", "UPTIME", "COST"],
    )
    _print_cost(orphans)
    if args.terminate and orphans:
        if not args.yes:
            confirm = input(f"\nterminate {len(orphans)} orphan(s)? [y/N] ").strip().lower()
            if confirm != "y":
                print("aborted")
                return 1
        for p in orphans:
            try:
                await discovery.terminate(p.id, api_key)
                print(f"terminated {p.id}")
            except Exception as exc:
                print(f"failed to terminate {p.id}: {exc}", file=sys.stderr)
    return 0


async def _cmd_gpu_types(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    sdk = api._get_runpod()
    sdk.api_key = api_key
    gpus = await asyncio.to_thread(sdk.get_gpus)
    if args.json:
        print(json.dumps(gpus, default=str, indent=2))
    else:
        for g in gpus or []:
            print(f"{g.get('displayName','-')}  ({g.get('id','-')})")
    return 0


# -- Sprint 4 verbs --------------------------------------------------------


async def _cmd_launch(args: argparse.Namespace) -> int:
    """Launch a new RunPod pod. With --detach, prints details and exits 0."""
    config = _resolve_config(args)
    name = getattr(args, "name", None)
    pod: Pod | None = None
    if getattr(args, "wait_capacity", None):
        pod = await _launch_when_available(
            config,
            name=name,
            max_wait_sec=args.wait_capacity,
            retry_interval_sec=args.retry_interval,
        )
    else:
        pod = await _launch(config, name=name)

    if getattr(args, "probe_only", False):
        info = {
            "pod_id": pod.id,
            "name": pod.name,
            "selected_gpu_type": getattr(pod, "_gpu_type", None),
            "selected_ram_tier_gb": getattr(pod, "_ram_tier", None),
            "selected_storage_name": getattr(pod, "_storage_name", None),
            "selected_storage_volume_id": getattr(pod, "_storage_volume", None),
            "gpu_type_candidates": list(config.gpu_type_candidates),
            "storage_candidates": _unique_values([config.storage_name, *config.storage_volumes]),
            "terminated": False,
        }
        try:
            await pod.terminate()
            info["terminated"] = True
            print(json.dumps(info, indent=2))
            return 0
        except Exception as exc:
            info["terminate_error"] = str(exc)
            print(json.dumps(info, indent=2), file=sys.stderr)
            return 1

    await pod.wait_ready(timeout=getattr(args, "timeout", 600))

    ssh_details = await pod._ensure_ssh_details()
    info = {
        "pod_id": pod.id,
        "name": pod.name,
        "ssh": f"root@{ssh_details['ip']} -p {ssh_details['port']}",
        "gpu_type": config.gpu_type,
    }
    print(json.dumps(info, indent=2))

    if not getattr(args, "detach", False):
        print(f"\nPod {pod.id} is running. Press Ctrl-C to terminate.")
        try:
            while True:
                await asyncio.sleep(10)
        except KeyboardInterrupt:
            print("\nTerminating pod...")
            await pod.terminate()
            print(f"terminated {pod.id}")
    return 0


async def _cmd_exec(args: argparse.Namespace) -> int:
    """Execute a command on an existing pod via SSH."""
    config = _resolve_config(args)
    pod = await discovery.get_pod(args.pod_id, config)
    await pod.wait_ready(timeout=60)
    # REMAINDER captures the raw command tokens after '--'; join them back
    remote_cmd = " ".join(args.exec_cmd) if args.exec_cmd else ""
    if not remote_cmd:
        print("error: no command provided", file=sys.stderr)
        return 2
    code, stdout, stderr = await pod.exec_ssh(remote_cmd, timeout=getattr(args, "timeout", 600))
    if stdout:
        print(stdout, end="")
    if stderr:
        print(stderr, end="", file=sys.stderr)
    return code


async def _cmd_ship(args: argparse.Namespace) -> int:
    """Upload a local directory tree to a pod."""
    config = _resolve_config(args)
    pod = await discovery.get_pod(args.pod_id, config)
    await pod.wait_ready(timeout=60)
    exclude = set(getattr(args, "exclude", "").split(",")) if getattr(args, "exclude", None) else set()
    mode = getattr(args, "upload_mode", "sftp_walk") or "sftp_walk"
    await pod.upload_path(Path(args.local).resolve(), args.remote, exclude=exclude, mode=mode)
    print(f"shipped {args.local} -> {args.pod_id}:{args.remote}")
    return 0


async def _cmd_fetch(args: argparse.Namespace) -> int:
    """Download artifact directories from a pod."""
    config = _resolve_config(args)
    pod = await discovery.get_pod(args.pod_id, config)
    await pod.wait_ready(timeout=60)
    local = Path(args.local).resolve()
    result = await pod.download_archive(args.remote, local)
    if result:
        print(f"fetched artifacts -> {result}")
    else:
        print("no artifacts found", file=sys.stderr)
        return 1
    return 0


async def _cmd_run(args: argparse.Namespace) -> int:
    """Ship a script and run it on a pod (sync composite)."""
    from .runner import ship_and_run

    config = _resolve_config(args)
    script_path = Path(args.script).resolve()
    if not script_path.exists():
        print(f"error: script not found: {args.script}", file=sys.stderr)
        return 1
    remote_script = script_path.read_text()
    local_root = script_path.parent
    remote_root = getattr(args, "remote_root", "/workspace")

    result = await ship_and_run(
        config,
        remote_script,
        local_root=local_root,
        remote_root=remote_root,
        exclude=set(),
        upload_mode=getattr(args, "upload_mode", "sftp_walk") or "sftp_walk",
        timeout=getattr(args, "timeout", 600),
        name_prefix=getattr(args, "name_prefix", None) or config.name_prefix,
        terminate_after_exec=not getattr(args, "keep_pod", False),
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode


async def _cmd_volumes_ls(args: argparse.Namespace) -> int:
    """List all RunPod network volumes."""
    volumes = await Pod.list_storages()
    if args.json:
        print(json.dumps(volumes, default=str, indent=2))
    else:
        if not volumes:
            print("(no volumes)")
            return 0
        headers = ["ID", "NAME", "SIZE", "DATACENTER"]
        rows: list[list[str]] = []
        for v in volumes:
            rows.append([
                v.get("id", "-"),
                v.get("name", "-"),
                f"{v.get('size', '-')} GB",
                v.get("dataCenterId", "-"),
            ])
        width_id = max(len("ID"), max(len(r[0]) for r in rows))
        width_name = max(len("NAME"), max(len(r[1]) for r in rows))
        width_size = max(len("SIZE"), max(len(r[2]) for r in rows))
        width_dc = max(len("DATACENTER"), max(len(r[3]) for r in rows))
        fmt = f"{{:<{width_id}}}  {{:<{width_name}}}  {{:<{width_size}}}  {{:<{width_dc}}}"
        print(fmt.format(*headers))
        print(fmt.format(*["-" * w for w in [width_id, width_name, width_size, width_dc]]))
        for r in rows:
            print(fmt.format(*r))
    return 0


async def _cmd_volume_create(args: argparse.Namespace) -> int:
    """Create a RunPod network volume."""
    if not args.datacenter:
        print("error: --datacenter is required", file=sys.stderr)
        return 2
    try:
        vol = await Pod.create_storage(args.name, args.size_gb, args.datacenter)
        print(json.dumps(vol, default=str, indent=2))
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


async def _cmd_probe(args: argparse.Namespace) -> int:
    """Query RunPod for currently-launchable GPU configs (no pod created)."""
    api_key = _resolve_api_key(args)

    gpu_types_arg: list[str] | None = None
    if getattr(args, "gpu_types", None):
        gpu_types_arg = [g.strip() for g in args.gpu_types.split(",") if g.strip()] or None

    datacenter_ids: list[str] | None = None
    if getattr(args, "datacenter_ids", None):
        datacenter_ids = [
            d.strip() for d in args.datacenter_ids.split(",") if d.strip()
        ] or None

    results = await _probe(
        api_key=api_key,
        gpu_types=gpu_types_arg,
        min_memory_gb=args.min_memory,
        max_price_per_hour=args.max_price,
        require_secure_cloud=not args.allow_community_cloud,
        exclude_blackwell=args.exclude_blackwell,
        container_disk_gb=args.container_disk_gb,
        datacenter_ids=datacenter_ids,
    )

    fmt = getattr(args, "format", "json") or "json"
    if fmt == "json":
        print(json.dumps(results, indent=2))
        return 0

    # table
    if not results:
        print("(no viable configurations)")
        return 0
    headers = ["GPU TYPE", "MEM GB", "$/HR", "SECURE", "BLACKWELL"]
    rows: list[list[str]] = []
    for r in results:
        rows.append([
            str(r.get("gpu_type", "-")),
            str(r.get("memory_gb", "-")),
            f"${float(r.get('price_per_hour', 0.0)):.3f}",
            "yes" if r.get("secure_cloud") else "no",
            "yes" if r.get("is_blackwell") else "no",
        ])
    _print_table(rows, headers)
    return 0


# ---------------------------------------------------------------------------
# Prebuilt validation-environment CLI verb
# ---------------------------------------------------------------------------


PREBUILT_VOLUME_NAME_PREFIX = "reigh-livetest-prebuilt-"
_BUILDER_POD_PREFIX = "reigh-livetest-builder-"
_PREBUILT_POD_PREFIX = "reigh-livetest-prebuilt-"
_PREBUILT_RECONCILE_GPU_FALLBACKS = (
    "NVIDIA L4,"
    "NVIDIA RTX A4000,"
    "NVIDIA RTX 4000 Ada Generation,"
    "NVIDIA A40,"
    "NVIDIA GeForce RTX 4090"
)
_PREBUILT_CLEANUP_PREFIXES = (_BUILDER_POD_PREFIX, _PREBUILT_POD_PREFIX)
_BUILDER_REIGH_WORKER_DIR = "/opt/build/reigh-worker"
_BUILDER_VIBECOMFY_DIR = "/opt/build/vibecomfy"
_BUILDER_VENV_PATH = "/opt/reigh-worker-live-test-venv"
_BUILDER_VOLUME_MOUNT_PATH = "/workspace"
_REIGH_WORKER_REPO_URL = "https://github.com/banodoco/Reigh-Worker.git"
_VIBECOMFY_REPO_URL = "https://github.com/peteromallet/VibeComfy.git"
_RUNPOD_BASE_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
_VENV_BUNDLE_NAME = "venv.cuda124.tar.zst"
_VIBECOMFY_BUNDLE_NAME = "vibecomfy.tar.zst"
_MANIFEST_BUNDLE_FORMAT_VERSION = 1
_MANIFEST_SCHEMA_VERSION = 1


@contextlib.contextmanager
def _prebuilt_phase(phase_name: str, **fields: Any):
    """Lightweight phase logger for the prebuilt CLI (no reigh-worker import)."""
    started = time.monotonic()
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    print(f"phase_start name={phase_name} {extra}".rstrip(), flush=True)
    try:
        yield
    except Exception as exc:
        elapsed = round(time.monotonic() - started, 1)
        print(
            f"phase_fail name={phase_name} elapsed_sec={elapsed} error={type(exc).__name__}: {exc}",
            flush=True,
        )
        raise
    else:
        elapsed = round(time.monotonic() - started, 1)
        print(f"phase_done name={phase_name} elapsed_sec={elapsed}", flush=True)


def _builder_timestamp_label() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ").lower()


def _builder_contract(args: argparse.Namespace) -> PrebuiltEnvContract:
    volume_name = args.volume_name or prebuilt_volume_name_for_profile(
        args.attention_profile, args.data_center
    )
    return PrebuiltEnvContract(
        volume_name=volume_name,
        data_center_id=args.data_center,
        attention_profile=args.attention_profile,
        comfyui_pin=getattr(args, "comfyui_pin", "fix/latentupscale-model-mmap-residency"),
        python_version=args.python_version,
        bundle_format_version=_MANIFEST_BUNDLE_FORMAT_VERSION,
    )


def _quote(value: str) -> str:
    return shlex.quote(str(value))


def _stderr_excerpt(text: str) -> str:
    lines = (text or "").splitlines()
    if len(lines) <= 40:
        return "\n".join(lines)
    return "\n".join(lines[:20] + ["..."] + lines[-20:])


def _redact_sensitive_text(text: str) -> str:
    patterns = [
        r"RUNPOD_API_KEY=[^\s]+",
        r"SUPABASE_SERVICE_ROLE_KEY=[^\s]+",
        r"hf_[A-Za-z0-9_=-]+",
        r"rp_[A-Za-z0-9_=-]+",
    ]
    redacted = text or ""
    for pattern in patterns:
        redacted = re.sub(pattern, "[REDACTED]", redacted)
    return redacted


def _exec_check(ssh: SSHClient, command: str, *, timeout: int = 600) -> tuple[str, str]:
    exit_code, stdout, stderr = ssh.execute_command(command, timeout)
    if exit_code != 0:
        raise RuntimeError(
            f"Remote command failed with exit {exit_code}: {command}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return stdout, stderr


def _uv_sync_builder_shell(
    workdir: str,
    *,
    env_path: str,
    python_version: str,
    extras: tuple[str, ...],
) -> str:
    """Render the same uv-sync shell that reigh-worker run_install emits.

    Kept in-line here so runpod-lifecycle stays self-contained (no reigh-worker
    import). T14's golden-string test locks the equivalent reigh-worker version.
    """
    if not extras:
        raise ValueError("_uv_sync_builder_shell requires a non-empty extras tuple")
    if not python_version:
        raise ValueError("_uv_sync_builder_shell requires python_version")
    extras_args = " ".join(f"--extra {e}" for e in extras)
    python_arg = shlex.quote(str(python_version))
    return (
        f"cd {shlex.quote(workdir)}\n"
        'export PATH="$HOME/.local/bin:$PATH"\n'
        f"export UV_PROJECT_ENVIRONMENT={env_path}\n"
        "export UV_LINK_MODE=copy\n"
        'rm -rf "$UV_PROJECT_ENVIRONMENT"\n'
        f'uv venv --seed --python {python_arg} "$UV_PROJECT_ENVIRONMENT"\n'
        "for attempt in 1 2 3; do\n"
        f"  if uv sync --python {python_arg} {extras_args}; then\n"
        "    break\n"
        "  fi\n"
        '  echo "uv sync attempt $attempt failed; cleaning partial venv and retrying"\n'
        '  rm -rf .venv "$UV_PROJECT_ENVIRONMENT"\n'
        "  sleep 5\n"
        "  if [ $attempt -eq 3 ]; then exit 1; fi\n"
        "done\n"
    )


def _vibecomfy_install_builder_shell(
    workdir: str, *, python_path: str, attention_profile: str
) -> str:
    """Render the post-clone VibeComfy install body the builder pod will execute."""
    venv_path = python_path.rsplit("/bin/python", 1)[0]
    py = _quote(python_path)
    if attention_profile == "sage":
        torch_index = "https://download.pytorch.org/whl/cu128"
        torch_packages = "torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1"
        expected_cuda = "12.8"
    else:
        torch_index = "https://download.pytorch.org/whl/cu124"
        torch_packages = "torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0"
        expected_cuda = "12.4"
    torch_cuda = (
        f"uv pip install --python {py} --index-url {torch_index} --force-reinstall "
        f"{torch_packages}\n"
        f"uv pip install --python {py} 'numpy<2.3,>=1.24.4'\n"
        f"{py} - <<'PY'\n"
        "import torch\n"
        f"if torch.version.cuda != {expected_cuda!r}:\n"
        f"    raise RuntimeError(f'expected VibeComfy torch CUDA {expected_cuda}, got {{torch.version.cuda!r}}')\n"
        "print('vibecomfy torch cuda', torch.version.cuda)\n"
        "PY\n"
    )
    sage_block = ""
    if attention_profile == "sage":
        sage_block = (
            "rm -rf /tmp/sageattention\n"
            "git clone --depth 1 https://github.com/fblissjr/SageAttention-ada.git /tmp/sageattention\n"
            f"uv pip install --python {py} --no-build-isolation /tmp/sageattention\n"
            f"{py} - <<'PY'\n"
            "import sageattention\n"
            "import sageattention.core as core\n"
            "if not callable(getattr(sageattention, 'sageattn', None)):\n"
            "    raise RuntimeError('sageattention import succeeded but sageattn is missing')\n"
            "if core.get_cuda_version() < (12, 8):\n"
            "    raise RuntimeError(f'SageAttention-ada fast path requires CUDA >= 12.8, got {core.get_cuda_version()}')\n"
            "print('sageattention verified', core.get_cuda_version())\n"
            "PY\n"
        )
    return (
        f"uv venv --seed --python 3.11 {_quote(venv_path)}\n"
        f"uv pip install --python {py} -e {_quote(workdir)}\n"
        f"uv pip install --python {py} "
        "'comfyui@git+https://github.com/peteromallet/ComfyUI.git@fix/latentupscale-model-mmap-residency' "
        "'comfy-script[default]'\n"
        f"{torch_cuda}"
        f"{sage_block}"
        f"cd {_quote(workdir)}\n"
        "test -f custom_nodes.lock\n"
        f"{py} -m vibecomfy.cli nodes restore --lockfile custom_nodes.lock\n"
        f"{torch_cuda}"
        f"test -f {_quote(workdir)}/template_index.json\n"
        f"test -f {_quote(workdir)}/workflow_corpus/manifests/coverage.json\n"
    )


def _worker_python_version_probe_shell(python_path: str, expected_version: str) -> str:
    """Render a strict major.minor Python-version check for the worker venv."""
    return (
        "set -euo pipefail\n"
        f"{_quote(python_path)} - <<'PY'\n"
        "import sys\n"
        f"expected = {expected_version!r}\n"
        "observed = f'{sys.version_info.major}.{sys.version_info.minor}'\n"
        "if observed != expected:\n"
        "    raise SystemExit(f'worker python version mismatch: observed {observed}, expected {expected}')\n"
        "print(observed)\n"
        "PY\n"
    )


def _bundle_directory_shell(*, source_parent: str, source_name: str, bundle_path: str) -> str:
    staging = f"{bundle_path}.staging"
    return (
        "set -euo pipefail\n"
        f"mkdir -p {_quote(bundle_path.rsplit('/', 1)[0] or '/')}\n"
        f"rm -f {_quote(staging)}\n"
        f"tar --use-compress-program 'zstd -1 --threads=0' "
        f"-cf {_quote(staging)} -C {_quote(source_parent)} {_quote(source_name)}\n"
        f"sha256sum {_quote(staging)} | awk '{{print $1}}'\n"
        f"mv {_quote(staging)} {_quote(bundle_path)}\n"
    )


async def _connect_builder_ssh(pod: Pod) -> SSHClient:
    """Block on pod readiness, then return a connected SSHClient instance."""
    details = await pod._ensure_ssh_details()
    client = SSHClient(
        hostname=str(details["ip"]),
        port=int(details["port"]),
        username="root",
        password=details.get("password"),
        private_key_path=os.environ.get("REIGH_LIVE_TEST_SSH_KEY") or "~/.ssh/id_ed25519",
    )

    def _connect() -> None:
        client.connect()

    await asyncio.to_thread(_connect)
    return client


async def _cmd_prebuilt(args: argparse.Namespace) -> int:
    dispatch = {
        "build": _cmd_prebuilt_build,
        "check": _cmd_prebuilt_check,
        "cleanup": _cmd_prebuilt_cleanup,
        "inspect": _cmd_prebuilt_inspect,
        "invalidate": _cmd_prebuilt_invalidate,
        "list": _cmd_prebuilt_list,
        "reconcile": _cmd_prebuilt_reconcile,
        "status": _cmd_prebuilt_status,
    }
    handler = dispatch[args.prebuilt_cmd]
    return await handler(args)


def _prebuilt_target_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "targets_json": str(args.targets_json) if getattr(args, "targets_json", None) else None,
        "asset_manifest": str(args.asset_manifest) if getattr(args, "asset_manifest", None) else None,
        "enriched_targets_json": str(args.enriched_targets_json)
        if getattr(args, "enriched_targets_json", None)
        else None,
        "local_vibecomfy_dir": str(args.local_vibecomfy_dir)
        if getattr(args, "local_vibecomfy_dir", None)
        else None,
        "models_root": str(args.models_root) if getattr(args, "models_root", None) else None,
    }


def _read_json_path(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pod_summary_dict(pod: discovery.PodSummary) -> dict[str, Any]:
    if is_dataclass(pod):
        return asdict(pod)
    return dict(getattr(pod, "__dict__", {}))


def _prefixes_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    prefixes = tuple(getattr(args, "prefix", None) or _PREBUILT_CLEANUP_PREFIXES)
    disallowed = [p for p in prefixes if p not in _PREBUILT_CLEANUP_PREFIXES]
    if disallowed:
        raise ValueError(
            "prebuilt status/cleanup is restricted to validation pod prefixes: "
            + ", ".join(_PREBUILT_CLEANUP_PREFIXES)
        )
    return prefixes


async def _list_prebuilt_pods(api_key: str, prefixes: tuple[str, ...]) -> list[discovery.PodSummary]:
    pods_by_id: dict[str, discovery.PodSummary] = {}
    for prefix in prefixes:
        for pod in await discovery.list_pods(api_key, name_prefix=prefix):
            pods_by_id[pod.id] = pod
    return sorted(pods_by_id.values(), key=lambda p: (p.name or "", p.id))


async def _selected_prebuilt_volume_id(api_key: str, contract: PrebuiltEnvContract) -> str:
    volumes = await asyncio.to_thread(get_network_volumes, api_key)
    selected_volume = select_prebuilt_volume(
        volumes or [],
        profile=contract.attention_profile,
        data_center_id=contract.data_center_id,
        volume_name=contract.volume_name,
    )
    volume_id = str((selected_volume or {}).get("id") or "")
    if not volume_id:
        raise RuntimeError(
            f"Network volume {contract.volume_name!r} not found in datacenter "
            f"{contract.data_center_id!r}."
        )
    return volume_id


async def _launch_prebuilt_probe_pod(
    args: argparse.Namespace,
    *,
    api_key: str,
    contract: PrebuiltEnvContract,
    action: str,
) -> Pod:
    await _selected_prebuilt_volume_id(api_key, contract)
    gpu_type_arg = getattr(args, "gpu_type", None) or "NVIDIA GeForce RTX 4090"
    gpu_types = tuple(part.strip() for part in str(gpu_type_arg).split(",") if part.strip())
    if not gpu_types:
        raise RuntimeError("At least one GPU type is required")
    gpu_type: str | tuple[str, ...] = gpu_types[0] if len(gpu_types) == 1 else gpu_types
    pod_name = f"{_PREBUILT_POD_PREFIX}{action}-{_builder_timestamp_label()}"
    config = RunPodConfig(
        api_key=api_key,
        gpu_type=gpu_type,
        worker_image=_RUNPOD_BASE_IMAGE,
        container_disk_gb=max(100, int(getattr(args, "container_disk_gb", 100) or 100)),
        min_memory_gb=max(16, int(getattr(args, "min_memory_gb", 16) or 16)),
        name_prefix=_PREBUILT_POD_PREFIX,
        disk_size_gb=500,
        storage_name=contract.volume_name,
    )
    pod = await _launch(config, name=pod_name)
    await pod.wait_ready(timeout=900)
    return pod


def _extract_prebuilt_runtime_shell(contract: PrebuiltEnvContract) -> str:
    venv_bundle = f"{contract.cache_root}/{_VENV_BUNDLE_NAME}"
    vibecomfy_bundle = f"{contract.cache_root}/{_VIBECOMFY_BUNDLE_NAME}"
    runtime_parent = contract.runtime_vibecomfy_path.rsplit("/", 1)[0]
    return (
        "set -euo pipefail\n"
        "export DEBIAN_FRONTEND=noninteractive\n"
        "apt-get update\n"
        "apt-get install -y --no-install-recommends ffmpeg git curl zstd tar\n"
        f"test -f {_quote(venv_bundle)}\n"
        f"test -f {_quote(vibecomfy_bundle)}\n"
        f"rm -rf {_quote(contract.runtime_venv_path)} {_quote(contract.runtime_vibecomfy_path)}\n"
        f"mkdir -p {_quote(runtime_parent)} {_quote(contract.runtime_worker_path)}\n"
        f"tar --use-compress-program zstd -xf {_quote(venv_bundle)} -C /opt\n"
        f"tar --use-compress-program zstd -xf {_quote(vibecomfy_bundle)} -C {_quote(runtime_parent)}\n"
        f"cat > {_quote(contract.runtime_vibecomfy_path + '/extra_model_paths.yaml')} <<'YAML'\n"
        "reigh_prebuilt:\n"
        f"  base_path: {_quote(contract.models_path)}\n"
        "  checkpoints: checkpoints\n"
        "  clip: clip\n"
        "  clip_vision: clip_vision\n"
        "  configs: configs\n"
        "  controlnet: controlnet\n"
        "  diffusion_models: diffusion_models\n"
        "  embeddings: embeddings\n"
        "  latent_upscale_models: latent_upscale_models\n"
        "  loras: loras\n"
        "  style_models: style_models\n"
        "  text_encoders: text_encoders\n"
        "  unet: unet\n"
        "  upscale_models: upscale_models\n"
        "  vae: vae\n"
        "YAML\n"
    )


def _asset_entries_from_enriched_manifest(payload: Any) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not isinstance(payload, dict):
        return entries
    for target in payload.get("targets", []):
        if not isinstance(target, dict):
            continue
        template_id = target.get("template_id")
        for asset in target.get("assets", []):
            if isinstance(asset, dict):
                entries.append({"template_id": template_id, **asset})
    return entries


def _asset_entries_from_manifest(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("assets", "model_assets"):
        value = payload.get(key)
        if isinstance(value, list):
            return [entry for entry in value if isinstance(entry, dict)]
    return _asset_entries_from_enriched_manifest(payload)


def _asset_fetch_plan(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for entry in entries:
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            continue
        expected_path = entry.get("expected_path")
        url = entry.get("url")
        key = (name, str(expected_path or ""), str(url or ""))
        if key in seen:
            continue
        seen.add(key)
        plan.append(
            {
                "name": name,
                "template_id": entry.get("template_id"),
                "category": entry.get("category") or entry.get("subdir") or entry.get("directory"),
                "expected_path": expected_path,
                "url": url,
                "present": bool(entry.get("present")),
                "remediation": entry.get("remediation")
                or (
                    f"curl -L {url} -o {expected_path}"
                    if isinstance(url, str) and isinstance(expected_path, str)
                    else None
                ),
            }
        )
    return plan


def _selected_enriched_manifest_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if getattr(args, "enriched_targets_json", None):
        payload = _read_json_path(args.enriched_targets_json)
        return payload if isinstance(payload, dict) else None
    if getattr(args, "asset_manifest", None):
        entries = _asset_entries_from_manifest(_read_json_path(args.asset_manifest))
        return {
            "schema_version": 1,
            "producer": "runpod-lifecycle.asset-manifest-adapter",
            "targets": [
                {
                    "template_id": "asset_manifest",
                    "assets": entries,
                    "issues": [
                        {
                            "group": "runtime_deferred",
                            "code": "workflow_source_schema_not_enriched",
                            "severity": "warning",
                            "message": "Asset manifest supplied without VibeComfy source/schema enrichment.",
                        }
                    ],
                }
            ],
        }
    return None


def _health_summary(report) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for issue in report.issues:
        grouped.setdefault(issue.group, []).append(asdict(issue))
    return {
        "ok": report.ok,
        "health_path": None,
        "issue_count": len(report.issues),
        "issues_by_group": grouped,
    }


async def _cmd_prebuilt_check(args: argparse.Namespace) -> int:
    contract = _builder_contract(args)
    payload = {
        "action": "prebuilt check",
        "dry_run": bool(args.dry_run),
        "volume_name": contract.volume_name,
        "data_center_id": contract.data_center_id,
        "attention_profile": contract.attention_profile,
        "gpu_type": args.gpu_type,
        "container_disk_gb": args.container_disk_gb,
        "min_memory_gb": args.min_memory_gb,
        "python_version": contract.python_version,
        "manifest_path": manifest_path(contract),
        "target_handoff": _prebuilt_target_summary(args),
    }
    if args.dry_run:
        payload["no_credentials_required"] = True
        payload["would_probe"] = [
            "manifest",
            "worker_env",
            "vibecomfy_env",
            "custom_nodes",
            "extra_model_paths",
            "workflow_source",
            "schema",
            "assets",
        ]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    api_key = _resolve_api_key(args)
    pod_obj: Pod | None = None
    ssh: SSHClient | None = None
    try:
        with _prebuilt_phase("provision_prebuilt_check_pod", volume=contract.volume_name):
            pod_obj = await _launch_prebuilt_probe_pod(
                args, api_key=api_key, contract=contract, action="check"
            )
        with _prebuilt_phase("open_ssh", pod_id=pod_obj.id):
            ssh = await _connect_builder_ssh(pod_obj)
        manifest = await asyncio.to_thread(read_manifest, ssh, contract)
        enriched_manifest = _selected_enriched_manifest_from_args(args)
        if manifest is None:
            report = build_missing_manifest_health_report(
                contract,
                reason=(
                    f"No prebuilt manifest found at {manifest_path(contract)}. "
                    "Build or repair the prebuilt volume before running validation."
                ),
                targets_path=str(args.targets_json) if args.targets_json else None,
                enriched_path=str(args.enriched_targets_json) if args.enriched_targets_json else None,
            )
        else:
            with _prebuilt_phase("extract_prebuilt_runtime", pod_id=pod_obj.id):
                try:
                    await asyncio.to_thread(
                        _exec_check,
                        ssh,
                        "bash -lc " + _quote(_extract_prebuilt_runtime_shell(contract)),
                        timeout=1800,
                    )
                except Exception as exc:  # noqa: BLE001 - persist health evidence for early failures
                    report = build_error_health_report(
                        contract,
                        group="environment",
                        code="bundle_extract_failed",
                        reason=f"Failed to extract prebuilt bundles: {type(exc).__name__}: {exc}",
                        targets_path=str(args.targets_json) if args.targets_json else None,
                        enriched_path=str(args.enriched_targets_json) if args.enriched_targets_json else None,
                        detail={"manifest_path": manifest_path(contract)},
                    )
                else:
                    with _prebuilt_phase("run_prebuilt_health_probes", pod_id=pod_obj.id):
                        report = await asyncio.to_thread(
                            run_prebuilt_health_probes,
                            ssh,
                            contract,
                            manifest,
                            targets_path=str(args.targets_json) if args.targets_json else None,
                            enriched_path=str(args.enriched_targets_json) if args.enriched_targets_json else None,
                            enriched_manifest=enriched_manifest,
                        )
        with _prebuilt_phase("write_health_report", path=health_path(contract)):
            await asyncio.to_thread(write_health_report, ssh, contract, report)
        result = _health_summary(report)
        result.update(
            {
                "action": "prebuilt check",
                "pod_id": pod_obj.id,
                "volume_name": contract.volume_name,
                "health_path": health_path(contract),
            }
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if report.ok else 1
    finally:
        if ssh is not None:
            try:
                ssh.disconnect()
            except Exception:
                pass
        if pod_obj is not None:
            try:
                with _prebuilt_phase("terminate_prebuilt_check_pod", pod_id=pod_obj.id):
                    await pod_obj.terminate()
            except Exception as exc:
                print(f"warning: failed to terminate prebuilt check pod {pod_obj.id}: {exc}", file=sys.stderr)


async def _cmd_prebuilt_status(args: argparse.Namespace) -> int:
    prefixes = _prefixes_from_args(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "action": "prebuilt status",
                    "dry_run": True,
                    "no_credentials_required": True,
                    "prefixes": list(prefixes),
                    "json": bool(args.json),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    api_key = _resolve_api_key(args)
    pods = await _list_prebuilt_pods(api_key, prefixes)
    if args.json:
        print(json.dumps([_pod_summary_dict(pod) for pod in pods], default=str, indent=2))
        return 0
    _print_table(
        [_summary_to_row(pod) for pod in pods],
        ["ID", "NAME", "STATUS", "GPU", "UPTIME", "COST"],
    )
    _print_cost(pods)
    return 0


async def _cmd_prebuilt_cleanup(args: argparse.Namespace) -> int:
    prefixes = _prefixes_from_args(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "action": "prebuilt cleanup",
                    "dry_run": True,
                    "no_credentials_required": True,
                    "prefixes": list(prefixes),
                    "older_than_seconds": args.older_than,
                    "would_terminate_only_allowlisted_prefixes": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.yes:
        print("error: prebuilt cleanup requires --yes outside --dry-run", file=sys.stderr)
        return 2
    api_key = _resolve_api_key(args)
    pods = await _list_prebuilt_pods(api_key, prefixes)
    if args.older_than is not None:
        pods = [
            pod
            for pod in pods
            if pod.uptime_seconds is not None and pod.uptime_seconds >= args.older_than
        ]
    terminated: list[str] = []
    failed: list[dict[str, str]] = []
    for pod in pods:
        if not any((pod.name or "").startswith(prefix) for prefix in prefixes):
            failed.append({"pod_id": pod.id, "error": "name did not match allowlisted prefix"})
            continue
        try:
            await discovery.terminate(pod.id, api_key)
            terminated.append(pod.id)
        except Exception as exc:  # noqa: BLE001 - continue cleanup and report all failures
            failed.append({"pod_id": pod.id, "error": str(exc)})
    result = {
        "action": "prebuilt cleanup",
        "prefixes": list(prefixes),
        "terminated": terminated,
        "failed": failed,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for pod_id in terminated:
            print(f"terminated {pod_id}")
        for item in failed:
            print(f"failed {item['pod_id']}: {item['error']}", file=sys.stderr)
    return 1 if failed else 0


async def _cmd_prebuilt_reconcile(args: argparse.Namespace) -> int:
    contract = _builder_contract(args)
    handoff = _prebuilt_target_summary(args)
    payload: dict[str, Any] = {
        "action": "prebuilt reconcile",
        "dry_run": bool(args.dry_run),
        "volume_name": contract.volume_name,
        "data_center_id": contract.data_center_id,
        "attention_profile": contract.attention_profile,
        "target_handoff": handoff,
        "fetch_plan": [],
    }
    if args.enriched_targets_json:
        payload["fetch_plan"] = _asset_fetch_plan(
            _asset_entries_from_enriched_manifest(_read_json_path(args.enriched_targets_json))
        )
    elif args.asset_manifest:
        payload["fetch_plan"] = _asset_fetch_plan(
            _asset_entries_from_manifest(_read_json_path(args.asset_manifest))
        )
    elif args.targets_json:
        output_hint = str(Path(args.targets_json).with_suffix(".enriched.json"))
        payload["requires_enrichment"] = True
        payload["remediation"] = (
            "Generate enriched metadata before reconciling assets: "
            f"vibecomfy workflows enrich-targets --targets-json {args.targets_json} "
            f"--output {output_hint}"
            + (f" --models-root {args.models_root}" if args.models_root else "")
        )
        if args.local_vibecomfy_dir:
            payload["local_enrichment_command"] = (
                f"cd {args.local_vibecomfy_dir} && python -m vibecomfy.cli workflows enrich-targets "
                f"--targets-json {args.targets_json} --output {output_hint}"
                + (f" --models-root {args.models_root}" if args.models_root else "")
            )
    if args.dry_run:
        payload["no_credentials_required"] = True
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if not payload["fetch_plan"]:
        payload["status"] = "blocked"
        payload["message"] = (
            "Non-dry-run reconcile requires --enriched-targets-json or --asset-manifest with explicit assets. "
            "Plain --targets-json must be enriched first."
        )
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    api_key = _resolve_api_key(args)
    pod_obj: Pod | None = None
    ssh: SSHClient | None = None
    terminate_pod = True
    reconciled: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    try:
        if args.pod_id:
            terminate_pod = False
            with _prebuilt_phase("attach_prebuilt_reconcile_pod", pod_id=args.pod_id):
                pod_obj = await discovery.get_pod(args.pod_id, _resolve_config(args))
                await pod_obj.wait_ready(timeout=60)
        else:
            with _prebuilt_phase("provision_prebuilt_reconcile_pod", volume=contract.volume_name):
                pod_obj = await _launch_prebuilt_probe_pod(
                    args, api_key=api_key, contract=contract, action="reconcile"
                )
        with _prebuilt_phase("open_ssh", pod_id=pod_obj.id):
            ssh = await _connect_builder_ssh(pod_obj)
        total_assets = len(payload["fetch_plan"])
        for index, asset in enumerate(payload["fetch_plan"], start=1):
            expected_path = asset.get("expected_path")
            url = asset.get("url")
            name = asset.get("name")
            if not expected_path or not url:
                failed.append(
                    {
                        "name": name,
                        "error": "asset is missing expected_path or url; cannot reconcile without guessing",
                    }
                )
                continue
            print(
                json.dumps(
                    {
                        "event": "prebuilt_reconcile_fetch_start",
                        "index": index,
                        "total": total_assets,
                        "name": name,
                        "expected_path": expected_path,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            script = (
                "set -euo pipefail\n"
                f"if [ -f {_quote(str(expected_path))} ]; then\n"
                "  echo present\n"
                "  exit 0\n"
                "fi\n"
                f"mkdir -p {_quote(str(Path(str(expected_path)).parent))}\n"
                f"curl -L --fail --retry 3 --retry-delay 5 --continue-at - {_quote(str(url))} "
                f"-o {_quote(str(expected_path))}\n"
                "echo fetched\n"
            )
            exit_code, stdout, stderr = await asyncio.to_thread(
                ssh.execute_command,
                "bash -lc " + _quote(script),
                7200,
            )
            if exit_code == 0:
                reconciled.append(
                    {
                        "name": name,
                        "expected_path": expected_path,
                        "status": (stdout or "").strip().splitlines()[-1] if stdout.strip() else "ok",
                    }
                )
                print(
                    json.dumps(
                        {
                            "event": "prebuilt_reconcile_fetch_done",
                            "index": index,
                            "total": total_assets,
                            "name": name,
                            "status": reconciled[-1]["status"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            else:
                failed.append(
                    {
                        "name": name,
                        "expected_path": expected_path,
                        "error": _redact_sensitive_text(_stderr_excerpt(stderr)),
                    }
                )
                print(
                    json.dumps(
                        {
                            "event": "prebuilt_reconcile_fetch_failed",
                            "index": index,
                            "total": total_assets,
                            "name": name,
                            "error": failed[-1]["error"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        result = {
            "action": "prebuilt reconcile",
            "pod_id": pod_obj.id,
            "volume_name": contract.volume_name,
            "reconciled": reconciled,
            "failed": failed,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1 if failed else 0
    finally:
        if ssh is not None:
            try:
                ssh.disconnect()
            except Exception:
                pass
        if pod_obj is not None and terminate_pod:
            try:
                with _prebuilt_phase("terminate_prebuilt_reconcile_pod", pod_id=pod_obj.id):
                    await pod_obj.terminate()
            except Exception as exc:
                print(f"warning: failed to terminate prebuilt reconcile pod {pod_obj.id}: {exc}", file=sys.stderr)


async def _cmd_prebuilt_build(args: argparse.Namespace) -> int:
    if args.container_disk_gb < 100:
        print(
            f"error: --container-disk-gb must be >= 100 (got {args.container_disk_gb})",
            file=sys.stderr,
        )
        return 2
    contract = _builder_contract(args)
    pod_name = f"{_BUILDER_POD_PREFIX}{_builder_timestamp_label()}"
    build_gpu_type = _resolve_gpu_type(args)
    build_ram_tiers = _parse_csv_ints(getattr(args, "ram_tiers", None)) or cfg.DEFAULT_RAM_TIERS
    build_min_memory_gb = int(getattr(args, "min_memory_gb", 32) or 32)
    build_storage_volumes = _resolve_storage_volumes(args)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "action": "prebuilt build",
                    "dry_run": True,
                    "pod_name": pod_name,
                    "volume_name": contract.volume_name,
                    "data_center_id": contract.data_center_id,
                    "attention_profile": contract.attention_profile,
                    "worker_ref": args.worker_ref,
                    "vibecomfy_ref": args.vibecomfy_ref,
                    "python_version": contract.python_version,
                    "gpu_type": build_gpu_type,
                    "min_memory_gb": build_min_memory_gb,
                    "ram_tiers": list(build_ram_tiers),
                    "storage_volumes": list(build_storage_volumes),
                    "capacity_wait_sec": int(getattr(args, "capacity_wait_sec", 0) or 0),
                    "container_disk_gb": args.container_disk_gb,
                    "volume_disk_gb": args.volume_disk_gb,
                    "cache_root": contract.cache_root,
                    "venv_bundle": f"{contract.cache_root}/{_VENV_BUNDLE_NAME}",
                    "vibecomfy_bundle": f"{contract.cache_root}/{_VIBECOMFY_BUNDLE_NAME}",
                },
                indent=2,
            )
        )
        return 0

    api_key = _resolve_api_key(args)
    pod_obj: Pod | None = None
    ssh: SSHClient | None = None
    lock_release = None
    try:
        with _prebuilt_phase(
            "provision_builder_pod", name=pod_name, gpu_type=build_gpu_type
        ):
            volumes = await asyncio.to_thread(get_network_volumes, api_key)
            selected_volume = None
            if not build_storage_volumes:
                selected_volume = select_prebuilt_volume(
                    volumes or [],
                    profile=contract.attention_profile,
                    data_center_id=contract.data_center_id,
                    volume_name=contract.volume_name,
                )
                volume_id = str((selected_volume or {}).get("id") or "")
                if not volume_id:
                    raise RuntimeError(
                        f"Network volume {contract.volume_name!r} not found in datacenter "
                        f"{contract.data_center_id!r}. Use `runpod-lifecycle volumes create` first."
                    )
            config = RunPodConfig(
                api_key=api_key,
                gpu_type=build_gpu_type,
                worker_image=_RUNPOD_BASE_IMAGE,
                container_disk_gb=args.container_disk_gb,
                name_prefix=_BUILDER_POD_PREFIX,
                disk_size_gb=args.volume_disk_gb,
                min_memory_gb=build_min_memory_gb,
                ram_tiers=build_ram_tiers,
                storage_name=contract.volume_name if not build_storage_volumes else args.volume_name,
                storage_volumes=build_storage_volumes,
            )
            if getattr(args, "capacity_wait_sec", 0):
                pod_obj = await _launch_when_available(
                    config,
                    name=pod_name,
                    max_wait_sec=int(args.capacity_wait_sec),
                    retry_interval_sec=int(args.capacity_retry_interval_sec),
                )
            else:
                pod_obj = await _launch(config, name=pod_name)
            await pod_obj.wait_ready(timeout=900)
            selected_storage_name = getattr(pod_obj, "_storage_name", None)
            if selected_storage_name and selected_storage_name != contract.volume_name:
                volumes_by_name = {str(v.get("name") or ""): v for v in (volumes or [])}
                selected_volume = volumes_by_name.get(str(selected_storage_name))
                selected_data_center = str((selected_volume or {}).get("dataCenterId") or contract.data_center_id)
                contract = PrebuiltEnvContract(
                    volume_name=str(selected_storage_name),
                    data_center_id=selected_data_center,
                    attention_profile=args.attention_profile,
                    comfyui_pin=getattr(args, "comfyui_pin", "fix/latentupscale-model-mmap-residency"),
                    python_version=args.python_version,
                    bundle_format_version=_MANIFEST_BUNDLE_FORMAT_VERSION,
                )

        with _prebuilt_phase("open_ssh", pod_id=pod_obj.id):
            ssh = await _connect_builder_ssh(pod_obj)

        with _prebuilt_phase("acquire_lock", lock=contract.cache_root):
            lock_release = acquire_build_lock(
                ssh, contract, holder_id=pod_obj.id, ttl_sec=7200
            )

        with _prebuilt_phase("clone_repos"):
            clone_script = (
                "set -euo pipefail\n"
                "mkdir -p /opt/build\n"
                f"rm -rf {_quote(_BUILDER_REIGH_WORKER_DIR)} {_quote(_BUILDER_VIBECOMFY_DIR)}\n"
                f"git clone --branch {_quote(args.worker_ref)} --single-branch --recurse-submodules "
                f"{_quote(_REIGH_WORKER_REPO_URL)} {_quote(_BUILDER_REIGH_WORKER_DIR)}\n"
                f"git clone --branch {_quote(args.vibecomfy_ref)} --single-branch "
                f"{_quote(_VIBECOMFY_REPO_URL)} {_quote(_BUILDER_VIBECOMFY_DIR)}\n"
            )
            await asyncio.to_thread(
                _exec_check, ssh, "bash -lc " + _quote(clone_script), timeout=1800
            )

        with _prebuilt_phase("install_worker", workdir=_BUILDER_REIGH_WORKER_DIR):
            apt_python = f"python{contract.python_version}"
            apt_packages = f"{apt_python}-venv {apt_python}-dev build-essential ffmpeg git curl wget zstd pv"
            sync_body = _uv_sync_builder_shell(
                _BUILDER_REIGH_WORKER_DIR,
                env_path=_BUILDER_VENV_PATH,
                python_version=contract.python_version,
                extras=("cuda124",),
            )
            script = (
                "set -euo pipefail\n"
                "apt-get update\n"
                f"apt-get install -y {apt_packages}\n"
                "if ! command -v uv >/dev/null 2>&1; then\n"
                "  curl -LsSf https://astral.sh/uv/install.sh | sh\n"
                '  export PATH="$HOME/.local/bin:$PATH"\n'
                "fi\n"
                + sync_body
            )
            await asyncio.to_thread(
                _exec_check, ssh, "bash -lc " + _quote(script), timeout=3600
            )

        with _prebuilt_phase("verify_worker_python", expected=contract.python_version):
            await asyncio.to_thread(
                _exec_check,
                ssh,
                "bash -lc "
                + _quote(
                    _worker_python_version_probe_shell(
                        f"{_BUILDER_VENV_PATH}/bin/python", contract.python_version
                    )
                ),
                timeout=120,
            )

        with _prebuilt_phase("install_vibecomfy", workdir=_BUILDER_VIBECOMFY_DIR):
            install_body = _vibecomfy_install_builder_shell(
                _BUILDER_VIBECOMFY_DIR,
                python_path=f"{_BUILDER_VIBECOMFY_DIR}/.venv/bin/python",
                attention_profile=contract.attention_profile,
            )
            script = (
                "set -euo pipefail\n"
                f"export VIBECOMFY_ATTENTION_PROFILE={_quote(contract.attention_profile)}\n"
                + install_body
            )
            await asyncio.to_thread(
                _exec_check, ssh, "bash -lc " + _quote(script), timeout=3600
            )

        venv_bundle = f"{contract.cache_root}/{_VENV_BUNDLE_NAME}"
        vibecomfy_bundle = f"{contract.cache_root}/{_VIBECOMFY_BUNDLE_NAME}"

        with _prebuilt_phase("bundle_artifacts"):
            venv_parent, venv_name = _BUILDER_VENV_PATH.rsplit("/", 1)
            vc_parent, vc_name = _BUILDER_VIBECOMFY_DIR.rsplit("/", 1)
            mkdir_script = f"mkdir -p {_quote(contract.cache_root)}"
            await asyncio.to_thread(
                _exec_check, ssh, "bash -lc " + _quote(mkdir_script), timeout=60
            )
            venv_sha = await asyncio.to_thread(
                _run_bundle_capture_sha,
                ssh,
                _bundle_directory_shell(
                    source_parent=venv_parent, source_name=venv_name, bundle_path=venv_bundle
                ),
            )
            vibecomfy_sha = await asyncio.to_thread(
                _run_bundle_capture_sha,
                ssh,
                _bundle_directory_shell(
                    source_parent=vc_parent, source_name=vc_name, bundle_path=vibecomfy_bundle
                ),
            )

        with _prebuilt_phase("seed_models_dir", models_path=contract.models_path):
            seed_script = (
                "set -euo pipefail\n"
                f"mkdir -p {_quote(contract.models_path)}\n"
                f"if [ ! -f {_quote(contract.models_path)}/INDEX.json ]; then\n"
                f"  echo '{{}}' > {_quote(contract.models_path)}/INDEX.json\n"
                "fi\n"
            )
            await asyncio.to_thread(
                _exec_check, ssh, "bash -lc " + _quote(seed_script), timeout=60
            )

        with _prebuilt_phase("read_hashes"):
            pyproject_hash, custom_nodes_hash, worker_sha, vibecomfy_sha_git = (
                await asyncio.to_thread(_read_builder_hashes, ssh)
            )
            uv_version_stdout, _ = await asyncio.to_thread(
                _exec_check,
                ssh,
                "bash -lc 'export PATH=\"$HOME/.local/bin:$PATH\" && uv --version'",
                timeout=30,
            )
            venv_size_stdout, _ = await asyncio.to_thread(
                _exec_check,
                ssh,
                f"du -sb {_quote(_BUILDER_VENV_PATH)}/lib | awk '{{print $1}}'",
                timeout=120,
            )

        manifest = PrebuiltManifest(
            schema_version=_MANIFEST_SCHEMA_VERSION,
            bundle_format_version=_MANIFEST_BUNDLE_FORMAT_VERSION,
            built_at_utc=datetime.now(timezone.utc).isoformat(),
            built_by=pod_obj.id,
            pyproject_hash=pyproject_hash,
            custom_nodes_lock_hash=custom_nodes_hash,
            comfyui_pin=contract.comfyui_pin,
            attention_profile=contract.attention_profile,
            python_version=contract.python_version,
            cuda_extra="cuda124",
            vibecomfy_commit=vibecomfy_sha_git,
            reigh_worker_commit=worker_sha,
            uv_version=uv_version_stdout.strip(),
            venv_bundle_sha256=venv_sha,
            vibecomfy_bundle_sha256=vibecomfy_sha,
            models_index_sha256="",
            venv_size_bytes=int(venv_size_stdout.strip().splitlines()[-1] or 0),
            notes=args.notes or "",
        )

        with _prebuilt_phase("write_manifest"):
            await asyncio.to_thread(write_manifest, ssh, contract, manifest)

        print(
            json.dumps(
                {
                    "status": "ok",
                    "pod_id": pod_obj.id,
                    "volume_name": contract.volume_name,
                    "venv_bundle": venv_bundle,
                    "vibecomfy_bundle": vibecomfy_bundle,
                    "manifest_path": manifest_path(contract),
                },
                indent=2,
            )
        )
        return 0
    finally:
        if lock_release is not None:
            try:
                lock_release()
            except Exception as exc:
                print(f"warning: failed to release build lock: {exc}", file=sys.stderr)
        if ssh is not None:
            try:
                ssh.disconnect()
            except Exception:
                pass
        if pod_obj is not None:
            try:
                with _prebuilt_phase("terminate_builder_pod", pod_id=pod_obj.id):
                    await pod_obj.terminate()
            except Exception as exc:
                print(f"warning: failed to terminate builder pod {pod_obj.id}: {exc}", file=sys.stderr)


def _run_bundle_capture_sha(ssh: SSHClient, script_body: str) -> str:
    stdout, _ = _exec_check(ssh, "bash -lc " + _quote(script_body), timeout=7200)
    digest = ""
    for line in stdout.splitlines():
        candidate = line.strip()
        if len(candidate) == 64 and all(c in "0123456789abcdef" for c in candidate.lower()):
            digest = candidate.lower()
    if not digest:
        raise RuntimeError(f"failed to capture sha256 in bundle output: {stdout!r}")
    return digest


def _read_builder_hashes(ssh: SSHClient) -> tuple[str, str, str, str]:
    pyproject_stdout, _ = _exec_check(
        ssh, f"cat {_quote(_BUILDER_REIGH_WORKER_DIR)}/pyproject.toml", timeout=60
    )
    lock_stdout, _ = _exec_check(
        ssh, f"cat {_quote(_BUILDER_VIBECOMFY_DIR)}/custom_nodes.lock", timeout=60
    )
    worker_sha_stdout, _ = _exec_check(
        ssh, f"git -C {_quote(_BUILDER_REIGH_WORKER_DIR)} rev-parse HEAD", timeout=30
    )
    vibecomfy_sha_stdout, _ = _exec_check(
        ssh, f"git -C {_quote(_BUILDER_VIBECOMFY_DIR)} rev-parse HEAD", timeout=30
    )
    return (
        compute_pyproject_hash(pyproject_stdout),
        compute_lockfile_hash(lock_stdout),
        worker_sha_stdout.strip(),
        vibecomfy_sha_stdout.strip(),
    )


async def _cmd_prebuilt_inspect(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    contract = _builder_contract(args)
    pod_name = f"{_BUILDER_POD_PREFIX}{_builder_timestamp_label()}"
    pod_obj: Pod | None = None
    ssh: SSHClient | None = None
    try:
        with _prebuilt_phase("provision_probe_pod", name=pod_name):
            volumes = await asyncio.to_thread(get_network_volumes, api_key)
            volume_id = None
            for entry in volumes or []:
                if str(entry.get("name") or "") == contract.volume_name:
                    volume_id = str(entry.get("id") or "")
                    break
            if not volume_id:
                raise RuntimeError(
                    f"Network volume {contract.volume_name!r} not found."
                )
            config = RunPodConfig(
                api_key=api_key,
                gpu_type=args.gpu_type,
                worker_image=_RUNPOD_BASE_IMAGE,
                container_disk_gb=max(100, args.container_disk_gb),
                name_prefix=_BUILDER_POD_PREFIX,
                disk_size_gb=args.volume_disk_gb,
                storage_name=contract.volume_name,
            )
            pod_obj = await _launch(config, name=pod_name)
            await pod_obj.wait_ready(timeout=900)
        with _prebuilt_phase("read_manifest"):
            ssh = await _connect_builder_ssh(pod_obj)
            manifest = await asyncio.to_thread(read_manifest, ssh, contract)
            if manifest is None:
                print(f"no manifest present at {manifest_path(contract)}", file=sys.stderr)
                return 1
            print(
                json.dumps(
                    {
                        k: getattr(manifest, k)
                        for k in (
                            "schema_version",
                            "bundle_format_version",
                            "built_at_utc",
                            "built_by",
                            "python_version",
                            "cuda_extra",
                            "attention_profile",
                            "reigh_worker_commit",
                            "vibecomfy_commit",
                            "venv_size_bytes",
                            "venv_bundle_sha256",
                            "vibecomfy_bundle_sha256",
                        )
                    },
                    indent=2,
                )
            )
        return 0
    finally:
        if ssh is not None:
            try:
                ssh.disconnect()
            except Exception:
                pass
        if pod_obj is not None:
            try:
                await pod_obj.terminate()
            except Exception as exc:
                print(f"warning: failed to terminate probe pod: {exc}", file=sys.stderr)


async def _cmd_prebuilt_invalidate(args: argparse.Namespace) -> int:
    """Remove the manifest and both bundles from the volume, preserving models/ and build.lock."""
    contract = _builder_contract(args)
    pod_name = f"{_BUILDER_POD_PREFIX}{_builder_timestamp_label()}"
    if args.dry_run:
        print(
            json.dumps(
                {
                    "action": "prebuilt invalidate",
                    "dry_run": True,
                    "volume_name": contract.volume_name,
                    "removes": [
                        f"{contract.cache_root}/{_VENV_BUNDLE_NAME}",
                        f"{contract.cache_root}/{_VIBECOMFY_BUNDLE_NAME}",
                        manifest_path(contract),
                    ],
                    "preserves": [contract.models_path, f"{contract.cache_root}/build.lock"],
                },
                indent=2,
            )
        )
        return 0

    api_key = _resolve_api_key(args)
    pod_obj: Pod | None = None
    ssh: SSHClient | None = None
    try:
        with _prebuilt_phase("provision_invalidate_pod", name=pod_name):
            volumes = await asyncio.to_thread(get_network_volumes, api_key)
            volume_id = None
            for entry in volumes or []:
                if str(entry.get("name") or "") == contract.volume_name:
                    volume_id = str(entry.get("id") or "")
                    break
            if not volume_id:
                raise RuntimeError(
                    f"Network volume {contract.volume_name!r} not found."
                )
            config = RunPodConfig(
                api_key=api_key,
                gpu_type=args.gpu_type,
                worker_image=_RUNPOD_BASE_IMAGE,
                container_disk_gb=max(100, args.container_disk_gb),
                name_prefix=_BUILDER_POD_PREFIX,
                disk_size_gb=args.volume_disk_gb,
                storage_name=contract.volume_name,
            )
            pod_obj = await _launch(config, name=pod_name)
            await pod_obj.wait_ready(timeout=900)
            ssh = await _connect_builder_ssh(pod_obj)
        with _prebuilt_phase("invalidate"):
            # rm -rf the two bundle files and the manifest only; never touch
            # models/ or build.lock — they are explicitly preserved.
            script = (
                "set -euo pipefail\n"
                f"rm -f {_quote(contract.cache_root)}/{_VENV_BUNDLE_NAME}\n"
                f"rm -f {_quote(contract.cache_root)}/{_VIBECOMFY_BUNDLE_NAME}\n"
                f"rm -f {_quote(manifest_path(contract))}\n"
                f"ls -lA {_quote(contract.cache_root)} || true\n"
            )
            stdout, _ = await asyncio.to_thread(
                _exec_check, ssh, "bash -lc " + _quote(script), timeout=120
            )
            print(stdout, end="")
        return 0
    finally:
        if ssh is not None:
            try:
                ssh.disconnect()
            except Exception:
                pass
        if pod_obj is not None:
            try:
                await pod_obj.terminate()
            except Exception as exc:
                print(f"warning: failed to terminate invalidate pod: {exc}", file=sys.stderr)


async def _cmd_prebuilt_list(args: argparse.Namespace) -> int:
    api_key = _resolve_api_key(args)
    volumes = await asyncio.to_thread(get_network_volumes, api_key)
    matches = [
        v
        for v in (volumes or [])
        if str(v.get("name") or "").startswith(PREBUILT_VOLUME_NAME_PREFIX)
    ]
    if args.json:
        print(json.dumps(matches, default=str, indent=2))
        return 0
    if not matches:
        print("(no prebuilt volumes)")
        return 0
    headers = ["NAME", "DATACENTER", "SIZE_GB"]
    rows: list[list[str]] = []
    for v in matches:
        rows.append(
            [
                str(v.get("name") or "-"),
                str(v.get("dataCenterId") or "-"),
                str(v.get("size") or "-"),
            ]
        )
    _print_table(rows, headers)
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="runpod-lifecycle",
        description="RunPod pod lifecycle CLI.",
    )
    parser.add_argument("--api-key", help="Override RUNPOD_API_KEY env var.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- legacy verbs (unchanged from v0.1) ---------------------------------

    p_list = sub.add_parser("list", help="List all pods on the account.")
    p_list.add_argument("--name-prefix", help="Filter to pods whose name starts with PREFIX.")
    p_list.add_argument("--json", action="store_true")

    p_status = sub.add_parser("status", help="Show normalized status for a pod.")
    p_status.add_argument("pod_id")

    p_term = sub.add_parser("terminate", help="Terminate a pod.")
    p_term.add_argument("pod_id")
    p_term.add_argument("--yes", "-y", action="store_true", help="Skip confirmation.")

    p_orph = sub.add_parser(
        "find-orphans",
        help="Find pods on the account not in the supplied known-ids list.",
    )
    p_orph.add_argument("--known-ids-file", help="File with one pod id per line. Empty if omitted.")
    p_orph.add_argument(
        "--older-than",
        type=_parse_duration,
        default=None,
        help="Only orphans with uptime >= this duration (e.g. 1h, 30m, 90s).",
    )
    p_orph.add_argument("--name-prefix", help="Filter pods to those whose name starts with PREFIX.")
    p_orph.add_argument("--terminate", action="store_true", help="After listing, terminate each orphan.")
    p_orph.add_argument("--yes", "-y", action="store_true", help="Skip terminate confirmation.")

    p_gpu = sub.add_parser("gpu-types", help="List available GPU types from RunPod.")
    p_gpu.add_argument("--json", action="store_true")

    # --- Sprint 4 verbs -----------------------------------------------------

    p_launch = sub.add_parser("launch", help="Launch a new RunPod pod.")
    p_launch.add_argument("--detach", action="store_true", help="Launch and exit; keep pod running.")
    p_launch.add_argument("--name", help="Pod name (default: auto-generated).")
    p_launch.add_argument("--gpu-type", help="GPU type (default: RTX 4090).")
    p_launch.add_argument("--image", help="Docker image (default: pytorch devel).")
    p_launch.add_argument("--container-disk-gb", type=int, default=200, help="Container disk size GB.")
    p_launch.add_argument("--disk-size-gb", type=int, default=200, help="Pod disk size GB.")
    p_launch.add_argument("--min-memory-gb", type=int, help="Minimum host RAM GB for launch fallback.")
    p_launch.add_argument(
        "--ram-tiers",
        help="Comma-separated host RAM tiers to try, e.g. 32,24,16. Defaults to RUNPOD_RAM_TIERS.",
    )
    p_launch.add_argument("--name-prefix", help="Prefix for auto-generated pod name.")
    p_launch.add_argument("--storage-name", help="Network volume name to attach.")
    p_launch.add_argument(
        "--storage-volumes",
        help=(
            "Comma-separated network volume names to try after --storage-name. "
            "Defaults to RUNPOD_STORAGE_VOLUMES."
        ),
    )
    p_launch.add_argument("--timeout", type=int, default=600, help="Seconds to wait for pod readiness.")
    p_launch.add_argument("--datacenter-id", help="Datacenter ID (e.g. US-TX-1).")
    p_launch.add_argument(
        "--probe-only",
        action="store_true",
        help="Claim the first launchable GPU/storage/RAM candidate, print it, and immediately terminate it.",
    )
    p_launch.add_argument(
        "--wait-capacity",
        type=int,
        default=0,
        help="Seconds to keep retrying the requested GPU/storage matrix before failing.",
    )
    p_launch.add_argument(
        "--retry-interval",
        type=int,
        default=30,
        help="Seconds between --wait-capacity launch attempts.",
    )

    p_exec = sub.add_parser("exec", help="Execute a command on an existing pod via SSH.")
    p_exec.add_argument("pod_id")
    p_exec.add_argument("exec_cmd", nargs=argparse.REMAINDER, help="Command to execute.")
    p_exec.add_argument("--timeout", type=int, default=600, help="Command timeout in seconds.")
    p_exec.add_argument("--gpu-type", help="GPU type (for config; usually optional for exec).")

    p_ship = sub.add_parser("ship", help="Upload a local directory to a pod.")
    p_ship.add_argument("pod_id")
    p_ship.add_argument("--local", required=True, help="Local directory to upload.")
    p_ship.add_argument("--remote", required=True, help="Remote destination path on pod.")
    p_ship.add_argument("--exclude", help="Comma-separated list of patterns to exclude.")
    p_ship.add_argument("--upload-mode", choices=["sftp_walk", "tarball"], default="sftp_walk")

    p_fetch = sub.add_parser("fetch", help="Download artifact directories from a pod.")
    p_fetch.add_argument("pod_id")
    p_fetch.add_argument("--remote", required=True, help="Remote root path on pod (e.g. /workspace).")
    p_fetch.add_argument("--local", required=True, help="Local destination directory.")

    p_run = sub.add_parser("run", help="Ship a script file and run it on a pod (sync composite).")
    p_run.add_argument("pod_id")
    p_run.add_argument("--script", required=True, help="Path to the shell script to run.")
    p_run.add_argument("--remote-root", default="/workspace", help="Remote working directory.")
    p_run.add_argument("--upload-mode", choices=["sftp_walk", "tarball"], default="sftp_walk")
    p_run.add_argument("--timeout", type=int, default=600, help="Command timeout in seconds.")
    p_run.add_argument("--name-prefix", help="Name prefix for the pod.")
    p_run.add_argument("--keep-pod", action="store_true", help="Leave pod alive after script completes.")
    p_run.add_argument("--gpu-type", help="GPU type override.")
    p_run.add_argument("--image", help="Docker image override.")

    p_vols_ls = sub.add_parser("volumes", help="RunPod network volume operations.")
    vol_sub = p_vols_ls.add_subparsers(dest="volumes_cmd", required=True)

    p_vol_ls = vol_sub.add_parser("ls", help="List all network volumes.")
    p_vol_ls.add_argument("--json", action="store_true")

    p_vol_create = vol_sub.add_parser("create", help="Create a network volume.")
    p_vol_create.add_argument("name")
    p_vol_create.add_argument("size_gb", type=int)
    p_vol_create.add_argument("--datacenter", required=True, help="Datacenter ID (e.g. US-TX-1).")

    p_probe = sub.add_parser(
        "probe",
        help="Query RunPod for currently-launchable GPU configs (no pod created).",
    )
    p_probe.add_argument(
        "--gpu-types",
        dest="gpu_types",
        help="Comma-separated allow-list of GPU type ids (case-sensitive). "
        "Default: consider every type RunPod returns.",
    )
    p_probe.add_argument(
        "--min-memory",
        type=int,
        default=24,
        help="Minimum GPU VRAM in GB (default: 24).",
    )
    p_probe.add_argument(
        "--max-price",
        type=float,
        default=None,
        help="Cap hourly uninterruptable price (USD).",
    )
    p_probe.add_argument(
        "--allow-community-cloud",
        action="store_true",
        help="Include Community Cloud pricing (default: Secure Cloud only).",
    )
    p_probe.add_argument(
        "--exclude-blackwell",
        action="store_true",
        help="Drop Blackwell variants (hivemind reports training-quality regression).",
    )
    p_probe.add_argument(
        "--container-disk-gb",
        type=int,
        default=100,
        help="Container disk size used for forward-compatible availability checks.",
    )
    p_probe.add_argument(
        "--datacenter-ids",
        dest="datacenter_ids",
        help="Comma-separated datacenter id allow-list (forward-compatible; "
        "currently informational only).",
    )
    p_probe.add_argument(
        "--format",
        choices=["json", "table"],
        default="json",
        help="Output format (default: json).",
    )

    # --- prebuilt validation environment verbs ------------------------------
    p_prebuilt = sub.add_parser(
        "prebuilt",
        help="Manage the Reigh live-test prebuilt validation environment on a RunPod volume.",
    )
    prebuilt_sub = p_prebuilt.add_subparsers(dest="prebuilt_cmd", required=True)

    p_pb_build = prebuilt_sub.add_parser(
        "build",
        help="Provision a builder pod and bake the prebuilt env onto the named volume.",
    )
    p_pb_build.add_argument(
        "--volume-name",
        help="Prebuilt network volume name. Defaults to the canonical name for --attention-profile and --data-center.",
    )
    p_pb_build.add_argument("--data-center", required=True)
    p_pb_build.add_argument(
        "--attention-profile", choices=["portable", "sage"], default="portable"
    )
    p_pb_build.add_argument("--worker-ref", default="main")
    p_pb_build.add_argument("--vibecomfy-ref", default="main")
    p_pb_build.add_argument(
        "--gpu-type", default="NVIDIA GeForce RTX 4090", help="GPU type to provision the builder pod with."
    )
    p_pb_build.add_argument(
        "--storage-volumes",
        help="Comma-separated network volume names to try for the builder pod.",
    )
    p_pb_build.add_argument("--min-memory-gb", type=int, default=32)
    p_pb_build.add_argument(
        "--ram-tiers",
        help="Comma-separated host RAM tiers to try for the builder pod, e.g. 32,24,16.",
    )
    p_pb_build.add_argument(
        "--capacity-wait-sec",
        type=int,
        default=0,
        help="Seconds to keep retrying the requested builder GPU/storage matrix before failing.",
    )
    p_pb_build.add_argument(
        "--capacity-retry-interval-sec",
        type=int,
        default=30,
        help="Seconds between builder capacity retries.",
    )
    p_pb_build.add_argument(
        "--container-disk-gb",
        type=int,
        default=200,
        help="Builder pod container disk size in GB; floor 100.",
    )
    p_pb_build.add_argument(
        "--volume-disk-gb",
        type=int,
        default=500,
        help="Network volume disk size in GB for new volume provisioning.",
    )
    p_pb_build.add_argument("--python-version", default="3.10")
    p_pb_build.add_argument(
        "--comfyui-pin",
        default="fix/latentupscale-model-mmap-residency",
        help="ComfyUI pin recorded in the manifest.",
    )
    p_pb_build.add_argument(
        "--notes", default="", help="Free-form notes embedded in the manifest."
    )
    p_pb_build.add_argument("--dry-run", action="store_true")
    p_pb_build.add_argument(
        "--force",
        action="store_true",
        help="Continue past lock-busy errors (use only when previous build crashed).",
    )

    p_pb_check = prebuilt_sub.add_parser(
        "check",
        help="Check a prebuilt validation environment before expensive workflow runs.",
    )
    p_pb_check.add_argument(
        "--volume-name",
        help="Prebuilt network volume name. Defaults to the canonical name for --attention-profile and --data-center.",
    )
    p_pb_check.add_argument("--data-center", required=True)
    p_pb_check.add_argument("--attention-profile", choices=["portable", "sage"], default="portable")
    p_pb_check.add_argument("--gpu-type", default="NVIDIA GeForce RTX 4090")
    p_pb_check.add_argument("--container-disk-gb", type=int, default=100)
    p_pb_check.add_argument(
        "--min-memory-gb",
        type=int,
        default=16,
        help="Minimum host RAM for the short-lived probe pod; default keeps health checks launchable.",
    )
    p_pb_check.add_argument("--python-version", default="3.10")
    p_pb_check.add_argument("--targets-json", type=Path, help="Selected Reigh target manifest JSON.")
    p_pb_check.add_argument("--asset-manifest", type=Path, help="Optional enriched asset manifest JSON.")
    p_pb_check.add_argument("--enriched-targets-json", type=Path, help="VibeComfy enriched target manifest JSON.")
    p_pb_check.add_argument("--local-vibecomfy-dir", type=Path, help="Local VibeComfy checkout for enrichment handoff.")
    p_pb_check.add_argument("--models-root", type=Path, help="Model root used for local enrichment expectations.")
    p_pb_check.add_argument("--dry-run", action="store_true")

    p_pb_status = prebuilt_sub.add_parser(
        "status",
        help="List validation pods launched by the prebuilt tooling allowlist.",
    )
    p_pb_status.add_argument(
        "--prefix",
        action="append",
        choices=list(_PREBUILT_CLEANUP_PREFIXES),
        help="Allowlisted validation pod prefix to include. Repeatable; defaults to builder and prebuilt prefixes.",
    )
    p_pb_status.add_argument("--json", action="store_true")
    p_pb_status.add_argument("--dry-run", action="store_true")

    p_pb_cleanup = prebuilt_sub.add_parser(
        "cleanup",
        help="Terminate only allowlisted validation pods launched by prebuilt tooling.",
    )
    p_pb_cleanup.add_argument(
        "--prefix",
        action="append",
        choices=list(_PREBUILT_CLEANUP_PREFIXES),
        help="Allowlisted validation pod prefix to clean. Repeatable; defaults to builder and prebuilt prefixes.",
    )
    p_pb_cleanup.add_argument(
        "--older-than",
        type=_parse_duration,
        default=None,
        help="Only terminate validation pods with uptime >= this duration (e.g. 1h, 30m, 90s).",
    )
    p_pb_cleanup.add_argument("--yes", "-y", action="store_true", help="Required outside --dry-run.")
    p_pb_cleanup.add_argument("--json", action="store_true")
    p_pb_cleanup.add_argument("--dry-run", action="store_true")

    p_pb_inspect = prebuilt_sub.add_parser(
        "inspect", help="Provision a probe pod, attach the volume, print its manifest."
    )
    p_pb_inspect.add_argument(
        "--volume-name",
        help="Prebuilt network volume name. Defaults to the canonical name for --attention-profile and --data-center.",
    )
    p_pb_inspect.add_argument("--data-center", required=True)
    p_pb_inspect.add_argument("--attention-profile", choices=["portable", "sage"], default="portable")
    p_pb_inspect.add_argument("--gpu-type", default="NVIDIA GeForce RTX 4090")
    p_pb_inspect.add_argument("--container-disk-gb", type=int, default=100)
    p_pb_inspect.add_argument("--volume-disk-gb", type=int, default=500)
    p_pb_inspect.add_argument("--python-version", default="3.10")

    p_pb_invalidate = prebuilt_sub.add_parser(
        "invalidate",
        help="Remove the manifest and both bundles from the volume (preserves models/ and build.lock).",
    )
    p_pb_invalidate.add_argument(
        "--volume-name",
        help="Prebuilt network volume name. Defaults to the canonical name for --attention-profile and --data-center.",
    )
    p_pb_invalidate.add_argument("--data-center", required=True)
    p_pb_invalidate.add_argument("--attention-profile", choices=["portable", "sage"], default="portable")
    p_pb_invalidate.add_argument("--gpu-type", default="NVIDIA GeForce RTX 4090")
    p_pb_invalidate.add_argument("--container-disk-gb", type=int, default=100)
    p_pb_invalidate.add_argument("--volume-disk-gb", type=int, default=500)
    p_pb_invalidate.add_argument("--python-version", default="3.10")
    p_pb_invalidate.add_argument("--dry-run", action="store_true")

    p_pb_list = prebuilt_sub.add_parser(
        "list", help="Enumerate RunPod network volumes matching the prebuilt prefix."
    )
    p_pb_list.add_argument("--json", action="store_true")

    p_pb_reconcile = prebuilt_sub.add_parser(
        "reconcile",
        help="Plan or reconcile selected model assets for a prebuilt validation environment.",
    )
    p_pb_reconcile.add_argument(
        "--volume-name",
        help="Prebuilt network volume name. Defaults to the canonical name for --attention-profile and --data-center.",
    )
    p_pb_reconcile.add_argument("--data-center", required=True)
    p_pb_reconcile.add_argument("--attention-profile", choices=["portable", "sage"], default="portable")
    p_pb_reconcile.add_argument(
        "--gpu-type",
        default=_PREBUILT_RECONCILE_GPU_FALLBACKS,
        help=(
            "Comma-separated GPU candidate list for the short-lived asset hydration pod. "
            "Defaults to cheaper/smaller GPUs before RTX 4090 because reconciliation only downloads files."
        ),
    )
    p_pb_reconcile.add_argument("--container-disk-gb", type=int, default=100)
    p_pb_reconcile.add_argument(
        "--min-memory-gb",
        type=int,
        default=16,
        help="Minimum host RAM for the short-lived probe pod; default keeps asset reconciliation launchable.",
    )
    p_pb_reconcile.add_argument("--python-version", default="3.10")
    p_pb_reconcile.add_argument("--pod-id", help="Reuse an existing pod attached to the prebuilt volume; do not terminate it.")
    p_pb_reconcile.add_argument("--targets-json", type=Path, help="Selected Reigh target manifest JSON.")
    p_pb_reconcile.add_argument("--asset-manifest", type=Path, help="Asset manifest JSON with explicit assets.")
    p_pb_reconcile.add_argument("--enriched-targets-json", type=Path, help="VibeComfy enriched target manifest JSON.")
    p_pb_reconcile.add_argument("--local-vibecomfy-dir", type=Path, help="Local VibeComfy checkout for enrichment handoff.")
    p_pb_reconcile.add_argument("--models-root", type=Path, help="Model root used for local enrichment expectations.")
    p_pb_reconcile.add_argument("--dry-run", action="store_true")
    p_pb_reconcile.add_argument("--json", action="store_true", help="Reserved for symmetry; output is JSON.")

    return parser


_HANDLERS: dict[str, Any] = {
    "list": _cmd_list,
    "status": _cmd_status,
    "terminate": _cmd_terminate,
    "find-orphans": _cmd_find_orphans,
    "gpu-types": _cmd_gpu_types,
    # Sprint 4
    "launch": _cmd_launch,
    "exec": _cmd_exec,
    "ship": _cmd_ship,
    "fetch": _cmd_fetch,
    "run": _cmd_run,
    "probe": _cmd_probe,
    "volumes": None,  # dispatched via volumes_cmd below
    "prebuilt": _cmd_prebuilt,
}

_VOLUMES_HANDLERS: dict[str, Any] = {
    "ls": _cmd_volumes_ls,
    "create": _cmd_volume_create,
}


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)

    if args.cmd == "volumes":
        handler = _VOLUMES_HANDLERS[args.volumes_cmd]
    else:
        handler = _HANDLERS[args.cmd]

    try:
        return asyncio.run(handler(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
