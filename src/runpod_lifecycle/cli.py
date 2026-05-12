"""Command-line interface for runpod-lifecycle: launch, exec, ship, fetch, run, volumes, and legacy list/status/terminate/find-orphans/gpu-types."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from . import api, config as cfg, discovery
from .config import RunPodConfig
from .guard import PodGuard, install_signal_handlers
from .lifecycle import launch as _launch
from .pod import Pod
from .probe import probe as _probe


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


def _resolve_config(args: argparse.Namespace) -> RunPodConfig:
    api_key = _resolve_api_key(args)
    arg_storage = _coalesce_blank(getattr(args, "storage_name", None))
    env_storage = _coalesce_blank(os.getenv("RUNPOD_STORAGE_NAME"))
    return RunPodConfig(
        api_key=api_key,
        gpu_type=getattr(args, "gpu_type", None)
        or os.getenv("RUNPOD_GPU_TYPE", cfg.DEFAULT_GPU_TYPE),
        worker_image=getattr(args, "image", None)
        or os.getenv("RUNPOD_WORKER_IMAGE", cfg.DEFAULT_WORKER_IMAGE),
        container_disk_gb=getattr(args, "container_disk_gb", None)
        or int(os.getenv("RUNPOD_CONTAINER_DISK_GB", "200")),
        name_prefix=getattr(args, "name_prefix", None)
        or os.getenv("RUNPOD_NAME_PREFIX", "pod"),
        disk_size_gb=getattr(args, "disk_size_gb", None)
        or int(os.getenv("RUNPOD_DISK_SIZE_GB", "200")),
        storage_name=arg_storage or env_storage,
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
    pod = await _launch(config, name=name)
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
    p_launch.add_argument("--name-prefix", help="Prefix for auto-generated pod name.")
    p_launch.add_argument("--storage-name", help="Network volume name to attach.")
    p_launch.add_argument("--timeout", type=int, default=600, help="Seconds to wait for pod readiness.")
    p_launch.add_argument("--datacenter-id", help="Datacenter ID (e.g. US-TX-1).")

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