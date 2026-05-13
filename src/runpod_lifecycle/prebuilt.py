"""Prebuilt validation-environment contract and manifest for Reigh live tests.

This module describes the long-lived prebuilt environment stored on a RunPod
network volume that lets live tests skip the cold ``uv sync`` + VibeComfy
install cycle. The artifacts are two zstd-compressed tarballs (the worker
virtualenv with ComfyUI baked into ``site-packages``; the VibeComfy install
tree) plus a manifest describing the build inputs and an HF model cache
directory shared by all consumer runs.

Invalidation precedence
-----------------------

When a consumer pod boots against a prebuilt volume the manifest is read and
compared against the consumer's resolved inputs. Fields fall into three
buckets:

* **HARD-FAIL** — ``schema_version``, ``bundle_format_version``,
  ``python_version``, ``cuda_extra``. Drift in any of these means the bundle
  cannot be reused safely; the consumer must abort and ask the operator to
  rebuild via ``rl prebuilt build``.
* **Delta-sync** — ``pyproject_hash``, ``custom_nodes_lock_hash``,
  ``comfyui_pin``, ``vibecomfy_commit``, ``reigh_worker_commit``. Drift in
  these is recoverable with an incremental ``uv sync`` / ``pip install -e``
  / ``vibecomfy.cli nodes restore`` on top of the extracted bundle.
* **No-op** — all other fields, or when every hash matches the bundle is
  used as-is.

The contract is intentionally bundle-format-only: ``schema_version`` covers
manifest schema changes, ``bundle_format_version`` covers the tar/zstd layout
the bundles inside the volume are written with. Bumping either forces a
rebuild.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shlex
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional


_DEFAULT_MOUNT_PATH = "/workspace"
_DEFAULT_CACHE_DIRNAME = "reigh-livetest-prebuilt"
_DEFAULT_RUNTIME_VENV_PATH = "/opt/reigh-worker-live-test-venv"
_DEFAULT_RUNTIME_PARENT = "/opt/reigh-livetest-prebuilt"


@dataclass(frozen=True)
class PrebuiltEnvContract:
    """Filesystem + version contract describing where the prebuilt env lives."""

    volume_name: str
    data_center_id: str
    attention_profile: str
    comfyui_pin: str
    python_version: str
    bundle_format_version: int
    mount_path: str = _DEFAULT_MOUNT_PATH
    cache_root: str = field(init=False)
    runtime_venv_path: str = _DEFAULT_RUNTIME_VENV_PATH
    runtime_worker_path: str = field(init=False)
    runtime_vibecomfy_path: str = field(init=False)
    models_path: str = field(init=False)

    def __post_init__(self) -> None:
        cache_root = f"{self.mount_path.rstrip('/')}/{_DEFAULT_CACHE_DIRNAME}"
        runtime_worker = f"{_DEFAULT_RUNTIME_PARENT}/worker"
        runtime_vibecomfy = f"{_DEFAULT_RUNTIME_PARENT}/vibecomfy"
        models = f"{cache_root}/models"
        object.__setattr__(self, "cache_root", cache_root)
        object.__setattr__(self, "runtime_worker_path", runtime_worker)
        object.__setattr__(self, "runtime_vibecomfy_path", runtime_vibecomfy)
        object.__setattr__(self, "models_path", models)


@dataclass(frozen=True)
class PrebuiltManifest:
    """Manifest describing the contents of a built prebuilt cache."""

    schema_version: int
    bundle_format_version: int
    built_at_utc: str
    built_by: str
    pyproject_hash: str
    custom_nodes_lock_hash: str
    comfyui_pin: str
    attention_profile: str
    python_version: str
    vibecomfy_commit: str
    reigh_worker_commit: str
    uv_version: str
    venv_bundle_sha256: str
    vibecomfy_bundle_sha256: str
    models_index_sha256: str
    venv_size_bytes: int
    cuda_extra: str = "cuda124"
    notes: str = ""


def _canonical_bytes(content: str) -> bytes:
    """Newline-normalize and strip a trailing newline before hashing."""
    return content.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def compute_pyproject_hash(pyproject_text: str) -> str:
    """Return the canonicalized SHA256 of a ``pyproject.toml`` text."""
    return hashlib.sha256(_canonical_bytes(pyproject_text)).hexdigest()


def compute_lockfile_hash(lockfile_text: str) -> str:
    """Return the canonicalized SHA256 of a lockfile (uv.lock or custom_nodes.lock)."""
    return hashlib.sha256(_canonical_bytes(lockfile_text)).hexdigest()


def manifest_path(contract: PrebuiltEnvContract) -> str:
    """Absolute path of the manifest file inside the prebuilt cache."""
    return f"{contract.cache_root}/env.manifest.json"


def lock_path(contract: PrebuiltEnvContract) -> str:
    """Absolute path of the builder lock file inside the prebuilt cache."""
    return f"{contract.cache_root}/build.lock"


def staging_path(contract: PrebuiltEnvContract) -> str:
    """Absolute path of the per-builder staging dir inside the prebuilt cache."""
    return f"{contract.cache_root}/staging"


# --------------------------------------------------------------------------- #
# SSH-side helpers
# --------------------------------------------------------------------------- #


def _quote(value: str) -> str:
    return shlex.quote(str(value))


def _ssh_execute(ssh, command: str, *, timeout: int = 600, check: bool = True) -> tuple[int, str, str]:
    """Run *command* over *ssh*, returning (exit_code, stdout, stderr)."""
    exit_code, stdout, stderr = ssh.execute_command(command, timeout=timeout)
    if check and exit_code != 0:
        raise RuntimeError(
            f"Remote command failed with exit {exit_code}: {command}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return exit_code, stdout, stderr


def _stderr_excerpt(stderr: str) -> str:
    lines = (stderr or "").splitlines()
    if len(lines) <= 100:
        return "\n".join(lines)
    return "\n".join(lines[:50] + ["..."] + lines[-50:])


def read_manifest(ssh, contract: PrebuiltEnvContract) -> Optional[PrebuiltManifest]:
    """Return the manifest at ``manifest_path(contract)`` or ``None`` if absent / unreadable."""
    path = manifest_path(contract)
    exit_code, stdout, _stderr = _ssh_execute(
        ssh, f"cat {_quote(path)}", timeout=60, check=False
    )
    if exit_code != 0:
        return None
    body = (stdout or "").strip()
    if not body:
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    field_names = {f.name for f in dataclasses.fields(PrebuiltManifest)}
    filtered = {k: v for k, v in payload.items() if k in field_names}
    try:
        return PrebuiltManifest(**filtered)
    except TypeError:
        return None


def write_manifest(ssh, contract: PrebuiltEnvContract, manifest: PrebuiltManifest) -> None:
    """Persist *manifest* atomically via heredoc + ``mv`` rename."""
    final_path = manifest_path(contract)
    staging = f"{final_path}.staging"
    payload = json.dumps(dataclasses.asdict(manifest), indent=2, sort_keys=True)
    parent = contract.cache_root
    script = (
        "set -euo pipefail\n"
        f"mkdir -p {_quote(parent)}\n"
        f"cat > {_quote(staging)} <<'PREBUILT_MANIFEST_EOF'\n"
        f"{payload}\n"
        "PREBUILT_MANIFEST_EOF\n"
        f"mv {_quote(staging)} {_quote(final_path)}\n"
    )
    _ssh_execute(ssh, "bash -lc " + _quote(script), timeout=120)


def acquire_build_lock(
    ssh,
    contract: PrebuiltEnvContract,
    *,
    holder_id: str,
    ttl_sec: int = 7200,
) -> Callable[[], None]:
    """Acquire an O_EXCL build lock with TTL takeover semantics.

    Returns a zero-arg callback that releases the lock. Raises ``RuntimeError``
    if the lock is held by another live holder.
    """
    lock_file = lock_path(contract)
    parent = contract.cache_root
    now_iso = datetime.now(timezone.utc).isoformat()
    holder_payload = json.dumps({"holder_id": holder_id, "acquired_at": now_iso, "ttl_sec": ttl_sec})
    script = (
        "set -euo pipefail\n"
        f"mkdir -p {_quote(parent)}\n"
        # Check for an existing lock. If present, decide whether it has expired.
        f"if [ -e {_quote(lock_file)} ]; then\n"
        f"  existing=$(cat {_quote(lock_file)} 2>/dev/null || echo '')\n"
        f"  mtime=$(stat -c %Y {_quote(lock_file)} 2>/dev/null || stat -f %m {_quote(lock_file)})\n"
        "  now=$(date -u +%s)\n"
        "  age=$(( now - mtime ))\n"
        f"  if [ \"$age\" -lt {int(ttl_sec)} ]; then\n"
        "    echo \"LOCK_BUSY $existing\" >&2\n"
        "    exit 17\n"
        "  fi\n"
        f"  echo \"TAKEOVER $existing (age=${{age}}s)\" >&2\n"
        f"  rm -f {_quote(lock_file)}\n"
        "fi\n"
        # O_EXCL create via `set -C` noclobber.
        "(\n"
        "  set -C\n"
        f"  printf '%s' {_quote(holder_payload)} > {_quote(lock_file)}\n"
        ")\n"
    )
    exit_code, _stdout, stderr = _ssh_execute(
        ssh, "bash -lc " + _quote(script), timeout=60, check=False
    )
    if exit_code != 0:
        excerpt = _stderr_excerpt(stderr)
        raise RuntimeError(
            f"acquire_build_lock failed for {lock_file} (exit={exit_code}); stderr:\n{excerpt}"
        )

    def _release() -> None:
        _ssh_execute(
            ssh,
            "bash -lc " + _quote(f"rm -f {_quote(lock_file)}"),
            timeout=30,
            check=False,
        )

    return _release


def _torch_cuda_expected_for_extra(cuda_extra: str) -> str | None:
    if cuda_extra == "cuda124":
        return "12.4"
    if cuda_extra == "cuda128":
        return "12.8"
    return None


def _probe_torch_cuda(ssh, contract: PrebuiltEnvContract, manifest: PrebuiltManifest) -> str | None:
    cmd = (
        f"cd {_quote(contract.runtime_worker_path)} && "
        f"{_quote(contract.runtime_venv_path + '/bin/python')} "
        "-c 'import torch; print(torch.version.cuda)'"
    )
    exit_code, stdout, stderr = _ssh_execute(ssh, "bash -lc " + _quote(cmd), timeout=180, check=False)
    expected = _torch_cuda_expected_for_extra(manifest.cuda_extra)
    if exit_code != 0:
        return (
            f"torch CUDA probe failed (exit={exit_code}); expected {expected} for cuda_extra={manifest.cuda_extra}. "
            f"Re-run `rl prebuilt invalidate --volume-name {contract.volume_name}` then `rl prebuilt build`. stderr:\n"
            f"{_stderr_excerpt(stderr)}"
        )
    observed = (stdout or "").strip().splitlines()[-1].strip() if stdout.strip() else ""
    if expected and observed != expected:
        return (
            f"torch CUDA version {observed!r} != expected {expected!r} for cuda_extra={manifest.cuda_extra}. "
            f"Re-run `rl prebuilt invalidate --volume-name {contract.volume_name}` then `rl prebuilt build`."
        )
    return None


def _probe_vibecomfy_assets(ssh, contract: PrebuiltEnvContract) -> list[str]:
    issues: list[str] = []
    for relative in ("template_index.json", "workflow_corpus/manifests/coverage.json"):
        path = f"{contract.runtime_vibecomfy_path}/{relative}"
        exit_code, _stdout, stderr = _ssh_execute(
            ssh, f"test -f {_quote(path)}", timeout=30, check=False
        )
        if exit_code != 0:
            issues.append(
                f"missing required VibeComfy asset {path}; rerun `rl prebuilt invalidate` then `rl prebuilt build`. "
                f"stderr:\n{_stderr_excerpt(stderr)}"
            )
    return issues


def _probe_venv_size(ssh, contract: PrebuiltEnvContract, manifest: PrebuiltManifest) -> str | None:
    expected = int(manifest.venv_size_bytes or 0)
    if expected <= 0:
        return None
    target = f"{contract.runtime_venv_path}/lib"
    exit_code, stdout, stderr = _ssh_execute(
        ssh, f"du -sb {_quote(target)} | awk '{{print $1}}'", timeout=120, check=False
    )
    if exit_code != 0:
        return (
            f"venv size probe failed for {target} (exit={exit_code}). "
            f"stderr:\n{_stderr_excerpt(stderr)}"
        )
    text = (stdout or "").strip().splitlines()[-1].strip() if stdout.strip() else ""
    try:
        observed = int(text)
    except ValueError:
        return f"venv size probe returned non-numeric output {text!r} for {target}."
    threshold = int(expected * 0.8)
    if observed < threshold:
        return (
            f"venv at {target} is {observed} bytes (>20% smaller than manifest {expected}); "
            f"the bundle extracted incomplete. Run `rl prebuilt invalidate --volume-name "
            f"{contract.volume_name}` then `rl prebuilt build`."
        )
    return None


def _probe_node_schema_verify(ssh, contract: PrebuiltEnvContract) -> str | None:
    cmd = (
        f"cd {_quote(contract.runtime_vibecomfy_path)} && "
        f"{_quote(contract.runtime_vibecomfy_path + '/.venv/bin/python')} "
        "-m vibecomfy.cli nodes list --limit 1 --json"
    )
    exit_code, _stdout, stderr = _ssh_execute(ssh, "bash -lc " + _quote(cmd), timeout=300, check=False)
    if exit_code != 0:
        return (
            "node-schema verify failed for VibeComfy at "
            f"{contract.runtime_vibecomfy_path}. Run `rl prebuilt invalidate --volume-name "
            f"{contract.volume_name}` then `rl prebuilt build`. stderr:\n"
            f"{_stderr_excerpt(stderr)}"
        )
    return None


def verify_extracted_env(
    ssh,
    contract: PrebuiltEnvContract,
    manifest: PrebuiltManifest,
) -> list[str]:
    """Run preflight probes against an extracted prebuilt env.

    Every probe uses ``_ssh_execute(check=False)`` so non-zero exit codes never
    raise from inside the probe — instead a diagnostic string is appended to
    the returned list. Returns an empty list when all probes pass. The consumer
    is expected to join all returned issues into a single error message.
    """
    issues: list[str] = []
    torch_issue = _probe_torch_cuda(ssh, contract, manifest)
    if torch_issue:
        issues.append(torch_issue)
    issues.extend(_probe_vibecomfy_assets(ssh, contract))
    size_issue = _probe_venv_size(ssh, contract, manifest)
    if size_issue:
        issues.append(size_issue)
    # Always run the node-schema verify probe regardless of any earlier drift.
    node_issue = _probe_node_schema_verify(ssh, contract)
    if node_issue:
        issues.append(node_issue)
    return issues


__all__ = [
    "PrebuiltEnvContract",
    "PrebuiltManifest",
    "acquire_build_lock",
    "compute_lockfile_hash",
    "compute_pyproject_hash",
    "lock_path",
    "manifest_path",
    "read_manifest",
    "staging_path",
    "verify_extracted_env",
    "write_manifest",
]
