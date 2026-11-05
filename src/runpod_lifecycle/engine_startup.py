"""Trusted startup selection for the one original HTTPS agent's engine slot.

The selection is a small local JSON file (``--engine-startup FILE`` or
``RUNPOD_ENGINE_STARTUP_FILE``) read by the claiming process. It carries only
the existing ``pod_agent.EngineProxy`` constructor fields plus the private
on-volume agent state directory. It never carries a bearer, provider key or
controller key, and it is not an HTTP-configurable surface: the normalized
mapping is encoded into the original pod-create environment and consumed by
the agent at process start.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import stat
from pathlib import Path
from typing import Any

from . import pod_agent

SCHEMA = "astrid.engine-startup.v1"
ENGINE = "vibecomfy-h3"
ENGINE_PORT = 8188
FILE_LIMIT = 16384
STATE_DEPTH_LIMIT = 16
STATE_LENGTH_LIMIT = 512
_SAFE_STATE = re.compile(r"/[A-Za-z0-9._/-]+\Z")
_PROXY_KEYS = {"endpoint_id", "port", "input_files", "output_roots"}


_PLAIN_TEXT = re.compile(r"[A-Za-z0-9+/=_.:@-]*")
_SAFE_KEY = re.compile(r"[A-Za-z0-9+/=_.:@ -]*")


def is_plain_text(value: Any) -> bool:
    """The one plain-text rule for anything the SDK interpolates into GraphQL."""
    return isinstance(value, str) and _PLAIN_TEXT.fullmatch(value) is not None


def require_plain_text(field: str, value: Any) -> None:
    """None means "not sent"; anything else must be plain text."""
    if value is not None and not is_plain_text(value):
        raise ValueError(f"engine startup requires plain text in {field}")


def require_safe_public_key(value: Any) -> None:
    """An SSH key is never sent on this path; refuse a hostile one outright."""
    if value is not None and (not isinstance(value, str) or _SAFE_KEY.fullmatch(value) is None):
        raise ValueError("engine startup requires plain text in public_key_string")


def claim_state_dir(state_dir: str) -> dict[str, str]:
    """A fresh per-claim private state directory beneath the selected one."""
    claim_id = secrets.token_hex(8)
    return {"claim_id": claim_id, "state_dir": f"{state_dir}/{claim_id}"}


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _state_path(value: Any, volume_mount_path: str | None) -> str:
    error = "engine startup state_dir is unsafe"
    try:
        pod_agent.observation_path(value)
    except ValueError:
        raise ValueError(error) from None
    if (len(value) > STATE_LENGTH_LIMIT or len(value.split("/")) - 1 > STATE_DEPTH_LIMIT
            or not _SAFE_STATE.match(value)):
        raise ValueError(error)
    if volume_mount_path is not None:
        try:
            mount = pod_agent.observation_path(volume_mount_path)
        except ValueError:
            raise ValueError("engine startup requires a canonical volume mount path") from None
        # The private state must live strictly beneath the persistent volume.
        if mount == "/" or value == mount or not _under(value, mount):
            raise ValueError("engine startup state_dir must be beneath the volume mount path")
    elif value == "/":
        raise ValueError(error)
    return value


def normalize_engine_startup(value: Any, *, volume_mount_path: str | None = None) -> dict[str, Any]:
    """Return the canonical, deep-copied selection or raise ValueError.

    Idempotent: a normalized mapping normalizes to an equal mapping.
    """
    error = "invalid engine startup configuration"
    if (not isinstance(value, dict) or set(value) != {"schema", "engine_proxy", "state_dir"}
            or value["schema"] != SCHEMA or not isinstance(value["engine_proxy"], dict)):
        raise ValueError(error)
    proxy = value["engine_proxy"]
    if not {"endpoint_id", "port", "input_files", "output_roots"} <= set(proxy) or not set(proxy) <= _PROXY_KEYS:
        raise ValueError(error)
    if proxy["endpoint_id"] != ENGINE or type(proxy["port"]) is not int or proxy["port"] != ENGINE_PORT:
        raise ValueError("engine startup must select the vibecomfy-h3 slot on loopback port 8188")
    # The complete grants are fixed before the first enable: empty is refused
    # rather than expanded later.
    if not isinstance(proxy["input_files"], (list, tuple)) or not isinstance(proxy["output_roots"], (list, tuple)):
        raise ValueError(error)
    if not proxy["input_files"] or not proxy["output_roots"]:
        raise ValueError("engine startup requires non-empty input_files and output_roots")
    # Reuse the one parser for slot fields and path syntax; also refuse a
    # config the agent itself would later reject.
    slot = pod_agent.EngineProxy(
        proxy["endpoint_id"], proxy["port"],
        input_files=list(proxy["input_files"]), output_roots=list(proxy["output_roots"]))
    state = _state_path(value["state_dir"], volume_mount_path)
    # Output reads and input writes must never reach the private state (the
    # original bearer and readiness profile live there), in either direction.
    for path in (*slot.input_files, *slot.output_roots):
        if _under(path, state) or _under(state, path):
            raise ValueError("engine startup state_dir overlaps an input or output path")
    return {"schema": SCHEMA, "engine_proxy": slot.configuration(), "state_dir": state}


def _no_duplicates(pairs):
    result = {}
    for key, item in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = item
    return result


def _reject_constant(_name):
    raise ValueError("non-finite number")


def load_engine_startup(path: str | os.PathLike, *, volume_mount_path: str | None = None) -> dict[str, Any]:
    """Read a bounded local selection file without following symlinks."""
    raw = os.fspath(path)
    selected = Path(raw)
    if (not isinstance(raw, str) or not selected.is_absolute() or str(selected) != raw
            or any(p in ("", ".", "..") for p in raw.split("/")[1:])):
        raise ValueError("engine startup file path is unsafe")
    try:
        directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            for part in selected.parts[1:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                                dir_fd=directory)
                os.close(directory)
                directory = child
            fd = os.open(selected.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                         dir_fd=directory)
        finally:
            os.close(directory)
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_mode & 0o022 or info.st_nlink != 1 or info.st_size > FILE_LIMIT):
                raise ValueError("engine startup file must be an owned, unshared, bounded regular file")
            data = os.read(fd, FILE_LIMIT + 1)
        finally:
            os.close(fd)
    except OSError:
        raise ValueError("engine startup file is unavailable or unsafe") from None
    if len(data) > FILE_LIMIT:
        raise ValueError("engine startup file is too large")
    try:
        parsed = json.loads(data.decode("utf-8"), object_pairs_hook=_no_duplicates,
                            parse_constant=_reject_constant)
    except (UnicodeDecodeError, ValueError):
        raise ValueError("engine startup file is not strict JSON") from None
    return normalize_engine_startup(parsed, volume_mount_path=volume_mount_path)
