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
from typing import Any, Callable, Optional


_DEFAULT_MOUNT_PATH = "/workspace"
_DEFAULT_CACHE_DIRNAME = "reigh-livetest-prebuilt"
_DEFAULT_RUNTIME_VENV_PATH = "/opt/reigh-worker-live-test-venv"
_DEFAULT_RUNTIME_PARENT = "/opt/reigh-livetest-prebuilt"
PREBUILT_HEALTH_SCHEMA_VERSION = 1
PREBUILT_VOLUME_NAME_PREFIX = "reigh-livetest-prebuilt-"


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


@dataclass(frozen=True)
class PrebuiltPythonEnvReport:
    """Observed state for one Python environment inside a prebuilt pod."""

    label: str
    python_path: str
    cwd: str
    python_version: str | None = None
    torch_version: str | None = None
    torch_cuda: str | None = None
    import_ok: bool = False
    import_error: str | None = None


@dataclass(frozen=True)
class PrebuiltHealthIssue:
    """Structured prebuilt-health issue grouped by validation surface."""

    group: str
    code: str
    message: str
    severity: str = "error"
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PrebuiltHealthReport:
    """Versioned health report written as ``env.health.json``."""

    schema_version: int
    generated_at_utc: str
    volume_name: str
    data_center_id: str
    attention_profile: str
    worker_env: PrebuiltPythonEnvReport
    vibecomfy_env: PrebuiltPythonEnvReport
    issues: list[PrebuiltHealthIssue] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)
    targets_path: str | None = None
    enriched_path: str | None = None

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


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


def health_path(contract: PrebuiltEnvContract) -> str:
    """Absolute path of the latest prebuilt-health report inside the cache."""
    return f"{contract.cache_root}/env.health.json"


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


def normalize_data_center_id(data_center_id: str) -> str:
    """Normalize a RunPod ``dataCenterId`` for canonical prebuilt names."""
    normalized = str(data_center_id or "").strip().lower().replace("_", "-")
    if not normalized:
        raise ValueError("data_center_id is required")
    return normalized


def prebuilt_volume_name_for_profile(profile: str, data_center_id: str) -> str:
    """Return ``reigh-livetest-prebuilt-{profile}-{normalized-data-center}``."""
    profile_slug = str(profile or "").strip().lower().replace("_", "-")
    if not profile_slug:
        raise ValueError("profile is required")
    return f"{PREBUILT_VOLUME_NAME_PREFIX}{profile_slug}-{normalize_data_center_id(data_center_id)}"


def select_prebuilt_volume(
    volumes: list[dict[str, Any]],
    *,
    profile: str,
    data_center_id: str | None = None,
    volume_name: str | None = None,
) -> dict[str, Any] | None:
    """Select a prebuilt volume deterministically from RunPod volume records.

    Selection is based on the actual ``dataCenterId`` field when a data center
    is supplied. Without a data-center or explicit name, multiple matching
    canonical profile volumes are ambiguous and must be resolved by the caller.
    """
    requested_name = str(volume_name or "").strip()
    normalized_dc = normalize_data_center_id(data_center_id) if data_center_id else None
    profile_prefix = f"{PREBUILT_VOLUME_NAME_PREFIX}{str(profile or '').strip().lower().replace('_', '-')}-"
    matches: list[dict[str, Any]] = []
    for volume in volumes or []:
        name = str(volume.get("name") or "")
        dc = str(volume.get("dataCenterId") or volume.get("data_center_id") or "")
        if requested_name:
            if name != requested_name:
                continue
        elif not name.startswith(profile_prefix):
            continue
        if normalized_dc and normalize_data_center_id(dc) != normalized_dc:
            continue
        expected = prebuilt_volume_name_for_profile(profile, dc) if dc else None
        if not requested_name and expected and name != expected:
            continue
        matches.append(volume)
    matches.sort(
        key=lambda item: (
            normalize_data_center_id(str(item.get("dataCenterId") or item.get("data_center_id") or "zz")),
            str(item.get("name") or ""),
            str(item.get("id") or ""),
        )
    )
    if len(matches) > 1 and not normalized_dc and not requested_name:
        candidates = [
            f"{item.get('name') or '<unnamed>'} "
            f"({item.get('dataCenterId') or item.get('data_center_id') or 'unknown-dc'}, "
            f"id={item.get('id') or 'unknown-id'})"
            for item in matches
        ]
        raise ValueError(
            "Multiple prebuilt volumes match profile "
            f"{profile!r}; pass --data-center or --volume-name. Candidates: "
            + ", ".join(candidates)
        )
    return matches[0] if matches else None


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


def _dataclass_from_dict(cls, payload: dict[str, Any]):
    field_names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in payload.items() if k in field_names})


def read_health_report(ssh, contract: PrebuiltEnvContract) -> Optional[PrebuiltHealthReport]:
    """Return ``env.health.json`` or ``None`` if absent / invalid."""
    path = health_path(contract)
    exit_code, stdout, _stderr = _ssh_execute(
        ssh, f"cat {_quote(path)}", timeout=60, check=False
    )
    if exit_code != 0:
        return None
    try:
        payload = json.loads((stdout or "").strip())
    except json.JSONDecodeError:
        return None
    try:
        worker_env = _dataclass_from_dict(PrebuiltPythonEnvReport, payload.get("worker_env") or {})
        vibecomfy_env = _dataclass_from_dict(PrebuiltPythonEnvReport, payload.get("vibecomfy_env") or {})
        issues = [
            _dataclass_from_dict(PrebuiltHealthIssue, issue)
            for issue in payload.get("issues", [])
            if isinstance(issue, dict)
        ]
        base = {
            k: v
            for k, v in payload.items()
            if k not in {"worker_env", "vibecomfy_env", "issues"}
        }
        return PrebuiltHealthReport(
            **{
                **{
                    k: v
                    for k, v in base.items()
                    if k in {f.name for f in dataclasses.fields(PrebuiltHealthReport)}
                },
                "worker_env": worker_env,
                "vibecomfy_env": vibecomfy_env,
                "issues": issues,
            }
        )
    except TypeError:
        return None


def write_health_report(ssh, contract: PrebuiltEnvContract, report: PrebuiltHealthReport) -> None:
    """Persist ``env.health.json`` atomically via heredoc + rename."""
    final_path = health_path(contract)
    staging = f"{final_path}.staging"
    payload = json.dumps(dataclasses.asdict(report), indent=2, sort_keys=True)
    script = (
        "set -euo pipefail\n"
        f"mkdir -p {_quote(contract.cache_root)}\n"
        f"cat > {_quote(staging)} <<'PREBUILT_HEALTH_EOF'\n"
        f"{payload}\n"
        "PREBUILT_HEALTH_EOF\n"
        f"mv {_quote(staging)} {_quote(final_path)}\n"
    )
    _ssh_execute(ssh, "bash -lc " + _quote(script), timeout=120)


def _health_issue(
    group: str,
    code: str,
    message: str,
    *,
    severity: str = "error",
    detail: dict[str, Any] | None = None,
) -> PrebuiltHealthIssue:
    return PrebuiltHealthIssue(
        group=group,
        code=code,
        message=message,
        severity=severity,
        detail=detail or {},
    )


def _json_probe(ssh, script: str, *, timeout: int = 300) -> tuple[dict[str, Any] | None, str | None]:
    exit_code, stdout, stderr = _ssh_execute(
        ssh,
        "bash -lc " + _quote(script),
        timeout=timeout,
        check=False,
    )
    if exit_code != 0:
        return None, f"exit={exit_code}; stderr:\n{_stderr_excerpt(stderr)}"
    try:
        return json.loads((stdout or "").strip().splitlines()[-1]), None
    except (IndexError, json.JSONDecodeError) as exc:
        return None, f"invalid JSON probe output: {exc}; stdout:\n{_stderr_excerpt(stdout)}"


def _probe_python_env_report(
    ssh,
    *,
    label: str,
    python_path: str,
    cwd: str,
    import_name: str | None,
) -> PrebuiltPythonEnvReport:
    import_stmt = f"import {import_name}" if import_name else "pass"
    script = (
        "set -euo pipefail\n"
        f"cd {_quote(cwd)}\n"
        f"{_quote(python_path)} - <<'PY'\n"
        "import json, sys\n"
        "payload = {'python_version': f'{sys.version_info.major}.{sys.version_info.minor}'}\n"
        "try:\n"
        "    import torch\n"
        "    payload['torch_version'] = getattr(torch, '__version__', None)\n"
        "    payload['torch_cuda'] = getattr(torch.version, 'cuda', None)\n"
        "except Exception as exc:\n"
        "    payload['torch_error'] = f'{type(exc).__name__}: {exc}'\n"
        "try:\n"
        f"    {import_stmt}\n"
        "    payload['import_ok'] = True\n"
        "except Exception as exc:\n"
        "    payload['import_ok'] = False\n"
        "    payload['import_error'] = f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps(payload, sort_keys=True))\n"
        "PY\n"
    )
    payload, error = _json_probe(ssh, script, timeout=300)
    if payload is None:
        return PrebuiltPythonEnvReport(
            label=label,
            python_path=python_path,
            cwd=cwd,
            import_ok=False,
            import_error=error,
        )
    return PrebuiltPythonEnvReport(
        label=label,
        python_path=python_path,
        cwd=cwd,
        python_version=payload.get("python_version"),
        torch_version=payload.get("torch_version"),
        torch_cuda=payload.get("torch_cuda"),
        import_ok=bool(payload.get("import_ok")),
        import_error=payload.get("import_error") or payload.get("torch_error"),
    )


def _probe_system_packages(ssh) -> list[PrebuiltHealthIssue]:
    packages = ("ffmpeg", "git", "curl", "zstd")
    missing: list[str] = []
    for package in packages:
        exit_code, _stdout, _stderr = _ssh_execute(
            ssh,
            f"command -v {_quote(package)} >/dev/null 2>&1",
            timeout=30,
            check=False,
        )
        if exit_code != 0:
            missing.append(package)
    if not missing:
        return []
    return [
        _health_issue(
            "environment",
            "missing_system_packages",
            "Required system packages are missing: " + ", ".join(missing),
            detail={"packages": missing},
        )
    ]


def _env_report_issues(
    report: PrebuiltPythonEnvReport,
    *,
    group: str,
    expected_python: str | None,
    expected_cuda: str | None,
) -> list[PrebuiltHealthIssue]:
    issues: list[PrebuiltHealthIssue] = []
    if not report.import_ok:
        issues.append(
            _health_issue(
                group,
                "python_import_failed",
                f"{report.label} import probe failed at {report.python_path}",
                detail={"error": report.import_error, "cwd": report.cwd},
            )
        )
    if expected_python and report.python_version and report.python_version != expected_python:
        issues.append(
            _health_issue(
                group,
                "python_version_mismatch",
                f"{report.label} Python {report.python_version} != expected {expected_python}",
                detail={"observed": report.python_version, "expected": expected_python},
            )
        )
    if expected_cuda and report.torch_cuda and report.torch_cuda != expected_cuda:
        issues.append(
            _health_issue(
                group,
                "torch_cuda_mismatch",
                f"{report.label} torch CUDA {report.torch_cuda} != expected {expected_cuda}",
                detail={"observed": report.torch_cuda, "expected": expected_cuda},
            )
        )
    if expected_cuda and not report.torch_cuda:
        issues.append(
            _health_issue(
                group,
                "torch_cuda_unavailable",
                f"{report.label} torch CUDA could not be read",
                detail={"error": report.import_error},
            )
        )
    return issues


def _probe_sageattention(ssh, contract: PrebuiltEnvContract) -> list[PrebuiltHealthIssue]:
    if contract.attention_profile != "sage":
        return []
    python_path = f"{contract.runtime_vibecomfy_path}/.venv/bin/python"
    script = (
        "set -euo pipefail\n"
        f"cd {_quote(contract.runtime_vibecomfy_path)}\n"
        f"{_quote(python_path)} - <<'PY'\n"
        "import sageattention\n"
        "if not callable(getattr(sageattention, 'sageattn', None)):\n"
        "    raise SystemExit('sageattn callable missing')\n"
        "PY\n"
    )
    exit_code, _stdout, stderr = _ssh_execute(
        ssh,
        "bash -lc " + _quote(script),
        timeout=120,
        check=False,
    )
    if exit_code == 0:
        return []
    return [
        _health_issue(
            "environment",
            "sageattention_unavailable",
            "SageAttention profile requested but sageattention could not be imported/validated.",
            detail={"stderr": _stderr_excerpt(stderr)},
        )
    ]


def _probe_custom_nodes(ssh, contract: PrebuiltEnvContract) -> list[PrebuiltHealthIssue]:
    checks = {
        "custom_nodes_dir": f"{contract.runtime_vibecomfy_path}/custom_nodes",
        "custom_nodes_lock": f"{contract.runtime_vibecomfy_path}/custom_nodes.lock",
    }
    issues: list[PrebuiltHealthIssue] = []
    for code, path in checks.items():
        exit_code, _stdout, stderr = _ssh_execute(
            ssh,
            f"test -e {_quote(path)}",
            timeout=30,
            check=False,
        )
        if exit_code != 0:
            issues.append(
                _health_issue(
                    "custom_nodes",
                    code + "_missing",
                    f"Missing VibeComfy custom-node artifact: {path}",
                    detail={"path": path, "stderr": _stderr_excerpt(stderr)},
                )
            )
    return issues


def _probe_extra_model_paths(ssh, contract: PrebuiltEnvContract) -> list[PrebuiltHealthIssue]:
    path = f"{contract.runtime_vibecomfy_path}/extra_model_paths.yaml"
    exit_code, stdout, stderr = _ssh_execute(
        ssh,
        f"test -r {_quote(path)} && cat {_quote(path)}",
        timeout=30,
        check=False,
    )
    if exit_code != 0:
        return [
            _health_issue(
                "environment",
                "extra_model_paths_missing",
                f"extra_model_paths.yaml is missing or unreadable at {path}",
                detail={"path": path, "stderr": _stderr_excerpt(stderr)},
            )
        ]
    if contract.models_path not in (stdout or ""):
        return [
            _health_issue(
                "environment",
                "extra_model_paths_missing_models_root",
                f"extra_model_paths.yaml does not mention expected models root {contract.models_path}",
                detail={"path": path, "expected_models_path": contract.models_path},
            )
        ]
    check_script = f"""
from pathlib import Path
from comfy.cli_args import default_configuration
from comfy.cmd.folder_paths import init_default_paths
from comfy.component_model.folder_path_types import FolderNames

config = default_configuration()
config.extra_model_paths_config = [str(Path({path!r}).resolve())]
folders = FolderNames(is_root=True)
init_default_paths(folders, config, replace_existing=True)
expected = {contract.models_path!r} + "/vae"
paths = list(folders["vae"].paths)
if expected not in paths:
    raise SystemExit(f"expected {{expected}} in vae paths, got {{paths!r}}")
print("extra_model_paths_ok")
"""
    check_command = (
        f"cd {_quote(contract.runtime_vibecomfy_path)} && "
        f"{_quote(contract.runtime_vibecomfy_path + '/.venv/bin/python')} - <<'PY'\n"
        f"{check_script}"
        "PY"
    )
    exit_code, stdout, stderr = _ssh_execute(
        ssh,
        "bash -lc " + _quote(check_command),
        timeout=120,
        check=False,
    )
    if exit_code != 0:
        return [
            _health_issue(
                "environment",
                "extra_model_paths_not_loaded_by_embedded_comfy",
                "Embedded Comfy did not load extra_model_paths.yaml into model picker paths.",
                detail={
                    "path": path,
                    "expected_models_path": contract.models_path,
                    "stdout": _stderr_excerpt(stdout),
                    "stderr": _stderr_excerpt(stderr),
                },
            )
        ]
    return []


def _selected_assets_from_enriched(enriched_manifest: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(enriched_manifest, dict):
        return []
    assets: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for target in enriched_manifest.get("targets", []):
        if not isinstance(target, dict):
            continue
        template_id = target.get("template_id")
        for asset in target.get("assets", []):
            if isinstance(asset, dict):
                key = (
                    str(template_id or ""),
                    str(asset.get("expected_path") or ""),
                    str(asset.get("name") or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                assets.append({"template_id": template_id, **asset})
    return assets


def health_issues_from_enriched_manifest(
    enriched_manifest: dict[str, Any] | None,
) -> list[PrebuiltHealthIssue]:
    """Convert selected VibeComfy enrichment diagnostics into grouped health issues."""
    if not isinstance(enriched_manifest, dict):
        return [
            _health_issue(
                "runtime_deferred",
                "enriched_manifest_missing",
                "Selected workflow source/schema/assets were not enriched before this health probe.",
                severity="warning",
                detail={
                    "remediation": "Run `vibecomfy workflows enrich-targets --targets-json ... --output enriched.json` first."
                },
            )
        ]
    issues: list[PrebuiltHealthIssue] = []
    for target in enriched_manifest.get("targets", []):
        if not isinstance(target, dict):
            continue
        template_id = target.get("template_id")
        source = target.get("source") if isinstance(target.get("source"), dict) else {}
        if source and source.get("runtime_source_of_truth") is False:
            issues.append(
                _health_issue(
                    "workflow_source",
                    "non_python_runtime_source",
                    f"Template {template_id} is not a pure-Python runtime source.",
                    detail={"template_id": template_id, "source": source},
                )
            )
        schema = target.get("schema") if isinstance(target.get("schema"), dict) else {}
        if schema.get("compile_error"):
            issues.append(
                _health_issue(
                    "schema",
                    "api_compile_failed",
                    f"Template {template_id} failed API compilation during enrichment.",
                    detail={"template_id": template_id, "compile_error": schema.get("compile_error")},
                )
            )
        for item in target.get("issues", []):
            if not isinstance(item, dict):
                continue
            group = str(item.get("group") or "runtime_deferred")
            code = str(item.get("code") or "enrichment_issue")
            if group == "assets" and code == "missing_model_asset":
                # Asset presence must be checked against the actual attached
                # RunPod volume, not the local machine that produced the
                # enrichment manifest.
                continue
            issues.append(
                _health_issue(
                    group,
                    code,
                    str(item.get("message") or "VibeComfy enrichment reported an issue."),
                    severity=str(item.get("severity") or "error"),
                    detail={
                        "template_id": template_id,
                        **(item.get("detail") if isinstance(item.get("detail"), dict) else {}),
                    },
                )
            )
    if not issues:
        issues.append(
            _health_issue(
                "runtime_deferred",
                "runtime_workflow_execution_not_run",
                "Health probe validated selected metadata but did not execute the workflow.",
                severity="info",
            )
        )
    return issues


def _probe_selected_assets(
    ssh,
    *,
    enriched_manifest: dict[str, Any] | None,
) -> list[PrebuiltHealthIssue]:
    issues: list[PrebuiltHealthIssue] = []
    for asset in _selected_assets_from_enriched(enriched_manifest):
        name = asset.get("name")
        expected_path = asset.get("expected_path")
        paths_checked = asset.get("paths_checked") if isinstance(asset.get("paths_checked"), list) else []
        if not expected_path:
            issues.append(
                _health_issue(
                    "assets",
                    "missing_expected_path",
                    f"Selected asset {name!r} has no expected_path metadata.",
                    detail={"asset": asset},
                )
            )
            continue
        exit_code, _stdout, _stderr = _ssh_execute(
            ssh,
            f"test -f {_quote(str(expected_path))}",
            timeout=30,
            check=False,
        )
        if exit_code != 0:
            checked = [str(path) for path in paths_checked] or [str(expected_path)]
            issues.append(
                _health_issue(
                    "assets",
                    "missing_model_asset",
                    f"Missing selected model asset {name} at {expected_path}",
                    detail={
                        "template_id": asset.get("template_id"),
                        "name": name,
                        "category": asset.get("category") or asset.get("subdir"),
                        "expected_path": str(expected_path),
                        "paths_checked": checked,
                        "url": asset.get("url"),
                        "remediation": asset.get("remediation"),
                    },
                )
            )
    return issues


def build_error_health_report(
    contract: PrebuiltEnvContract,
    *,
    group: str,
    code: str,
    reason: str,
    targets_path: str | None = None,
    enriched_path: str | None = None,
    detail: dict[str, Any] | None = None,
) -> PrebuiltHealthReport:
    empty_worker = PrebuiltPythonEnvReport(
        label="worker",
        python_path=f"{contract.runtime_venv_path}/bin/python",
        cwd=contract.runtime_worker_path,
    )
    empty_vibecomfy = PrebuiltPythonEnvReport(
        label="vibecomfy",
        python_path=f"{contract.runtime_vibecomfy_path}/.venv/bin/python",
        cwd=contract.runtime_vibecomfy_path,
    )
    return PrebuiltHealthReport(
        schema_version=PREBUILT_HEALTH_SCHEMA_VERSION,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        volume_name=contract.volume_name,
        data_center_id=contract.data_center_id,
        attention_profile=contract.attention_profile,
        worker_env=empty_worker,
        vibecomfy_env=empty_vibecomfy,
        issues=[
            _health_issue(
                group,
                code,
                reason,
                detail=detail or {},
            )
        ],
        targets_path=targets_path,
        enriched_path=enriched_path,
    )


def build_missing_manifest_health_report(
    contract: PrebuiltEnvContract,
    *,
    reason: str,
    targets_path: str | None = None,
    enriched_path: str | None = None,
) -> PrebuiltHealthReport:
    return build_error_health_report(
        contract,
        group="environment",
        code="manifest_missing",
        reason=reason,
        targets_path=targets_path,
        enriched_path=enriched_path,
        detail={"manifest_path": manifest_path(contract)},
    )


def run_prebuilt_health_probes(
    ssh,
    contract: PrebuiltEnvContract,
    manifest: PrebuiltManifest,
    *,
    targets_path: str | None = None,
    enriched_path: str | None = None,
    enriched_manifest: dict[str, Any] | None = None,
) -> PrebuiltHealthReport:
    """Run grouped health probes against an attached/extracted prebuilt pod."""
    expected_cuda = _torch_cuda_expected_for_extra(manifest.cuda_extra)
    worker_env = _probe_python_env_report(
        ssh,
        label="worker",
        python_path=f"{contract.runtime_venv_path}/bin/python",
        cwd=contract.runtime_worker_path,
        import_name=None,
    )
    vibecomfy_env = _probe_python_env_report(
        ssh,
        label="vibecomfy",
        python_path=f"{contract.runtime_vibecomfy_path}/.venv/bin/python",
        cwd=contract.runtime_vibecomfy_path,
        import_name="vibecomfy",
    )
    issues: list[PrebuiltHealthIssue] = []
    issues.extend(_probe_system_packages(ssh))
    issues.extend(
        _env_report_issues(
            worker_env,
            group="environment",
            expected_python=manifest.python_version,
            expected_cuda=expected_cuda,
        )
    )
    issues.extend(
        _env_report_issues(
            vibecomfy_env,
            group="environment",
            expected_python=None,
            expected_cuda=expected_cuda,
        )
    )
    issues.extend(_probe_sageattention(ssh, contract))
    issues.extend(_probe_custom_nodes(ssh, contract))
    issues.extend(_probe_extra_model_paths(ssh, contract))
    issues.extend(health_issues_from_enriched_manifest(enriched_manifest))
    issues.extend(_probe_selected_assets(ssh, enriched_manifest=enriched_manifest))
    return PrebuiltHealthReport(
        schema_version=PREBUILT_HEALTH_SCHEMA_VERSION,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        volume_name=contract.volume_name,
        data_center_id=contract.data_center_id,
        attention_profile=contract.attention_profile,
        worker_env=worker_env,
        vibecomfy_env=vibecomfy_env,
        issues=issues,
        manifest=dataclasses.asdict(manifest),
        targets_path=targets_path,
        enriched_path=enriched_path,
    )


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


def _probe_python_torch_cuda(
    ssh,
    *,
    python_path: str,
    cwd: str,
    label: str,
    contract: PrebuiltEnvContract,
    manifest: PrebuiltManifest,
) -> str | None:
    cmd = (
        f"cd {_quote(cwd)} && "
        f"{_quote(python_path)} "
        "-c 'import torch; print(torch.version.cuda)'"
    )
    exit_code, stdout, stderr = _ssh_execute(ssh, "bash -lc " + _quote(cmd), timeout=180, check=False)
    expected = _torch_cuda_expected_for_extra(manifest.cuda_extra)
    if exit_code != 0:
        return (
            f"{label} torch CUDA probe failed (exit={exit_code}); expected {expected} for cuda_extra={manifest.cuda_extra}. "
            f"Re-run `rl prebuilt invalidate --volume-name {contract.volume_name}` then `rl prebuilt build`. stderr:\n"
            f"{_stderr_excerpt(stderr)}"
        )
    observed = (stdout or "").strip().splitlines()[-1].strip() if stdout.strip() else ""
    if expected and observed != expected:
        return (
            f"{label} torch CUDA version {observed!r} != expected {expected!r} for cuda_extra={manifest.cuda_extra}. "
            f"Re-run `rl prebuilt invalidate --volume-name {contract.volume_name}` then `rl prebuilt build`."
        )
    return None


def _probe_worker_torch_cuda(ssh, contract: PrebuiltEnvContract, manifest: PrebuiltManifest) -> str | None:
    return _probe_python_torch_cuda(
        ssh,
        python_path=contract.runtime_venv_path + "/bin/python",
        cwd=contract.runtime_worker_path,
        label="worker",
        contract=contract,
        manifest=manifest,
    )


def _probe_vibecomfy_torch_cuda(ssh, contract: PrebuiltEnvContract, manifest: PrebuiltManifest) -> str | None:
    return _probe_python_torch_cuda(
        ssh,
        python_path=contract.runtime_vibecomfy_path + "/.venv/bin/python",
        cwd=contract.runtime_vibecomfy_path,
        label="VibeComfy",
        contract=contract,
        manifest=manifest,
    )


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


def _probe_vibecomfy_runtime_import(ssh, contract: PrebuiltEnvContract) -> str | None:
    cmd = (
        f"cd {_quote(contract.runtime_vibecomfy_path)} && "
        f"{_quote(contract.runtime_vibecomfy_path + '/.venv/bin/python')} "
        "-c 'import vibecomfy; import comfy; print(\"vibecomfy-runtime-ok\")'"
    )
    exit_code, stdout, stderr = _ssh_execute(ssh, "bash -lc " + _quote(cmd), timeout=300, check=False)
    if exit_code != 0:
        return (
            "VibeComfy runtime import failed at "
            f"{contract.runtime_vibecomfy_path}. Run `rl prebuilt invalidate --volume-name "
            f"{contract.volume_name}` then `rl prebuilt build`. stderr:\n"
            f"{_stderr_excerpt(stderr)}\nstdout:\n{_stderr_excerpt(stdout)}"
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
    worker_torch_issue = _probe_worker_torch_cuda(ssh, contract, manifest)
    if worker_torch_issue:
        issues.append(worker_torch_issue)
    vibecomfy_torch_issue = _probe_vibecomfy_torch_cuda(ssh, contract, manifest)
    if vibecomfy_torch_issue:
        issues.append(vibecomfy_torch_issue)
    issues.extend(_probe_vibecomfy_assets(ssh, contract))
    size_issue = _probe_venv_size(ssh, contract, manifest)
    if size_issue:
        issues.append(size_issue)
    runtime_issue = _probe_vibecomfy_runtime_import(ssh, contract)
    if runtime_issue:
        issues.append(runtime_issue)
    return issues


__all__ = [
    "PREBUILT_HEALTH_SCHEMA_VERSION",
    "PREBUILT_VOLUME_NAME_PREFIX",
    "PrebuiltEnvContract",
    "PrebuiltHealthIssue",
    "PrebuiltHealthReport",
    "PrebuiltManifest",
    "PrebuiltPythonEnvReport",
    "acquire_build_lock",
    "build_error_health_report",
    "build_missing_manifest_health_report",
    "compute_lockfile_hash",
    "compute_pyproject_hash",
    "health_issues_from_enriched_manifest",
    "health_path",
    "lock_path",
    "manifest_path",
    "normalize_data_center_id",
    "prebuilt_volume_name_for_profile",
    "read_manifest",
    "read_health_report",
    "run_prebuilt_health_probes",
    "select_prebuilt_volume",
    "staging_path",
    "verify_extracted_env",
    "write_health_report",
    "write_manifest",
]
