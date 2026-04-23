"""Command-line interface for runpod-lifecycle: list, status, terminate, find-orphans, gpu-types."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from typing import Any

from dotenv import load_dotenv

from . import api, discovery


def _resolve_api_key(args: argparse.Namespace) -> str:
    key = getattr(args, "api_key", None) or os.getenv("RUNPOD_API_KEY")
    if not key:
        print("error: RUNPOD_API_KEY not set (use --api-key or .env)", file=sys.stderr)
        sys.exit(2)
    return key


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="runpod-lifecycle", description="RunPod pod lifecycle CLI.")
    parser.add_argument("--api-key", help="Override RUNPOD_API_KEY env var.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List all pods on the account.")
    p_list.add_argument("--name-prefix", help="Filter to pods whose name starts with PREFIX.")
    p_list.add_argument("--json", action="store_true")

    p_status = sub.add_parser("status", help="Show normalized status for a pod.")
    p_status.add_argument("pod_id")

    p_term = sub.add_parser("terminate", help="Terminate a pod.")
    p_term.add_argument("pod_id")
    p_term.add_argument("--yes", "-y", action="store_true", help="Skip confirmation.")

    p_orph = sub.add_parser("find-orphans", help="Find pods on the account not in the supplied known-ids list.")
    p_orph.add_argument("--known-ids-file", help="File with one pod id per line. Empty if omitted.")
    p_orph.add_argument("--older-than", type=_parse_duration, default=None,
                        help="Only orphans with uptime >= this duration (e.g. 1h, 30m, 90s).")
    p_orph.add_argument("--name-prefix", help="Filter pods to those whose name starts with PREFIX.")
    p_orph.add_argument("--terminate", action="store_true", help="After listing, terminate each orphan.")
    p_orph.add_argument("--yes", "-y", action="store_true", help="Skip terminate confirmation.")

    p_gpu = sub.add_parser("gpu-types", help="List available GPU types from RunPod.")
    p_gpu.add_argument("--json", action="store_true")

    return parser


_HANDLERS = {
    "list": _cmd_list,
    "status": _cmd_status,
    "terminate": _cmd_terminate,
    "find-orphans": _cmd_find_orphans,
    "gpu-types": _cmd_gpu_types,
}


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
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
