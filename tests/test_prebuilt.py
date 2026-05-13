"""Unit tests for runpod_lifecycle.prebuilt (T13)."""

from __future__ import annotations

import dataclasses
import argparse
import asyncio
import json
import re
import time

import pytest

from runpod_lifecycle.prebuilt import (
    PrebuiltEnvContract,
    PrebuiltManifest,
    acquire_build_lock,
    compute_lockfile_hash,
    compute_pyproject_hash,
    lock_path,
    manifest_path,
    read_manifest,
    write_manifest,
)
from runpod_lifecycle import cli


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _make_contract(**overrides) -> PrebuiltEnvContract:
    defaults = dict(
        volume_name="reigh-livetest-prebuilt-portable-eu-no-1",
        data_center_id="eu-no-1",
        attention_profile="portable",
        comfyui_pin="fix/latentupscale-model-mmap-residency",
        python_version="3.11",
        bundle_format_version=1,
    )
    defaults.update(overrides)
    return PrebuiltEnvContract(**defaults)


def _make_manifest(**overrides) -> PrebuiltManifest:
    defaults = dict(
        schema_version=1,
        bundle_format_version=1,
        built_at_utc="2026-05-13T12:00:00+00:00",
        built_by="pod-abc",
        pyproject_hash="a" * 64,
        custom_nodes_lock_hash="b" * 64,
        comfyui_pin="fix/latentupscale-model-mmap-residency",
        attention_profile="portable",
        python_version="3.11",
        cuda_extra="cuda124",
        vibecomfy_commit="c" * 40,
        reigh_worker_commit="d" * 40,
        uv_version="uv 0.4.10",
        venv_bundle_sha256="e" * 64,
        vibecomfy_bundle_sha256="f" * 64,
        models_index_sha256="",
        venv_size_bytes=12_000_000_000,
        notes="",
    )
    defaults.update(overrides)
    return PrebuiltManifest(**defaults)


# --------------------------------------------------------------------------- #
# Hash determinism
# --------------------------------------------------------------------------- #


def test_compute_pyproject_hash_is_deterministic():
    a = compute_pyproject_hash("name = 'reigh-worker'\n[deps]\n")
    b = compute_pyproject_hash("name = 'reigh-worker'\n[deps]\n")
    assert a == b
    assert len(a) == 64


def test_compute_pyproject_hash_normalises_newlines():
    crlf = "line1\r\nline2\r\n"
    lf = "line1\nline2\n"
    cr = "line1\rline2\r"
    assert compute_pyproject_hash(crlf) == compute_pyproject_hash(lf)
    assert compute_pyproject_hash(cr) == compute_pyproject_hash(lf)


def test_compute_pyproject_hash_changes_with_content():
    assert compute_pyproject_hash("a") != compute_pyproject_hash("b")


def test_compute_lockfile_hash_matches_pyproject_normalisation():
    crlf = "node1\r\nnode2\r\n"
    lf = "node1\nnode2\n"
    assert compute_lockfile_hash(crlf) == compute_lockfile_hash(lf)


def test_prebuilt_build_dry_run_does_not_require_runpod_api_key(monkeypatch, capsys):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    args = argparse.Namespace(
        volume_name="reigh-livetest-prebuilt-portable-eu-no-1",
        data_center="eu-no-1",
        attention_profile="portable",
        worker_ref="main",
        vibecomfy_ref="main",
        gpu_type="NVIDIA GeForce RTX 4090",
        container_disk_gb=200,
        volume_disk_gb=200,
        python_version="3.11",
        comfyui_pin="fix/latentupscale-model-mmap-residency",
        notes="",
        dry_run=True,
        force=False,
    )

    assert asyncio.run(cli._cmd_prebuilt_build(args)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["volume_name"] == args.volume_name


def test_prebuilt_phase_accepts_name_field(capsys):
    with cli._prebuilt_phase("provision_builder_pod", name="pod-name"):
        pass

    out = capsys.readouterr().out
    assert "phase_start name=provision_builder_pod name=pod-name" in out
    assert "phase_done name=provision_builder_pod" in out


def test_vibecomfy_builder_install_uses_separate_python311_venv():
    body = cli._vibecomfy_install_builder_shell(
        "/opt/build/vibecomfy",
        python_path="/opt/build/vibecomfy/.venv/bin/python",
        attention_profile="portable",
    )

    assert "uv venv --seed --python 3.11 /opt/build/vibecomfy/.venv" in body
    assert "uv pip install --python /opt/build/vibecomfy/.venv/bin/python -e /opt/build/vibecomfy" in body


def test_prebuilt_build_installs_bundle_system_tools():
    import inspect

    source = inspect.getsource(cli._cmd_prebuilt_build)
    assert "zstd" in source
    assert "pv" in source


def test_prebuilt_manifest_uv_probe_uses_bootstrapped_path():
    import inspect

    source = inspect.getsource(cli._cmd_prebuilt_build)
    assert "uv --version" in source
    assert ".local/bin" in source


# --------------------------------------------------------------------------- #
# Manifest round-trip via simulated SSH
# --------------------------------------------------------------------------- #


class _FakeFileSystem:
    """Minimal in-memory FS that supports cat / cat-heredoc / mv / rm / test / stat / date."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.mtimes: dict[str, float] = {}
        self.now: float = 1_700_000_000.0  # arbitrary fixed clock

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeSSH:
    """Scripted SSH that interprets a tiny subset of bash for manifest/lock tests."""

    def __init__(self, fs: _FakeFileSystem | None = None) -> None:
        self.fs = fs or _FakeFileSystem()
        self.commands: list[tuple[str, int]] = []
        # crash_after_staging: when True, the next mv operation raises before
        # the rename happens (simulating builder crash).
        self.crash_after_staging = False
        self.staging_written: list[str] = []

    def execute_command(self, command: str, timeout: int = 600) -> tuple[int, str, str]:
        self.commands.append((command, timeout))
        # `cat path` returns file contents.
        cat_match = re.match(r"^cat (\S+)$", command)
        if cat_match:
            path = cat_match.group(1)
            content = self.fs.files.get(path)
            return (0, content, "") if content is not None else (1, "", "no such file")
        # `bash -lc 'script'` — parse the multiline body.
        bash_match = re.match(r"^bash -lc '(.*)'$", command, re.DOTALL)
        if bash_match:
            body = bash_match.group(1).replace("'\"'\"'", "'")
            return self._run_bash(body)
        return (1, "", f"unhandled fake ssh command: {command}")

    def _run_bash(self, body: str) -> tuple[int, str, str]:
        stdout_lines: list[str] = []
        # Strip set -euo pipefail.
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped == "set -euo pipefail":
                continue
            if stripped.startswith("mkdir -p "):
                continue
            if stripped.startswith("rm -f "):
                target = stripped[len("rm -f ") :].strip().strip("'")
                self.fs.files.pop(target, None)
                self.fs.mtimes.pop(target, None)
                continue
            # heredoc support: `cat > path <<'EOF'` block.
            if stripped.startswith("cat > "):
                # parse `cat > /path <<'EOF'`
                m = re.match(r"cat > (\S+) <<'(\w+)'", stripped)
                if not m:
                    return (1, "", f"unparseable heredoc start: {stripped}")
                target = m.group(1).strip("'")
                marker = m.group(2)
                # Pull subsequent lines until marker.
                idx = body.splitlines().index(line) + 1
                heredoc_lines = []
                for next_line in body.splitlines()[idx:]:
                    if next_line.strip() == marker:
                        break
                    heredoc_lines.append(next_line)
                contents = "\n".join(heredoc_lines)
                self.fs.files[target + ".staging" if target.endswith(".staging") is False else target] = contents
                # Actually we always treat the literal target here.
                self.fs.files[target] = contents
                self.fs.mtimes[target] = self.fs.now
                self.staging_written.append(target)
                if self.crash_after_staging:
                    raise RuntimeError("simulated builder crash before mv")
                continue
            if stripped.startswith("mv "):
                # mv src dst
                parts = stripped.split()
                src = parts[1].strip("'")
                dst = parts[2].strip("'")
                if src in self.fs.files:
                    self.fs.files[dst] = self.fs.files.pop(src)
                    self.fs.mtimes[dst] = self.fs.mtimes.pop(src, self.fs.now)
                continue
            if stripped.startswith("printf '%s' "):
                # `printf '%s' 'payload' > path`
                m = re.match(r"printf '%s' (.*) > (\S+)", stripped)
                if not m:
                    return (1, "", f"unparseable printf: {stripped}")
                payload = m.group(1).strip("'")
                target = m.group(2).strip("'")
                self.fs.files[target] = payload
                self.fs.mtimes[target] = self.fs.now
                continue
            if stripped.startswith("("):
                # `( set -C; printf ... > lockfile )` subshell — process inner.
                # We've already handled the printf body inside the block.
                continue
            if stripped.startswith(")"):
                continue
            if stripped.startswith("set -C"):
                continue
            if stripped.startswith("if [ -e "):
                # Multi-line if block handled via prefix scan; simplest path
                # is to evaluate the whole block manually.
                return self._eval_lock_acquire(body)
            if stripped == "fi":
                continue
            if stripped.startswith("echo "):
                continue
        return (0, "\n".join(stdout_lines), "")

    def _eval_lock_acquire(self, body: str) -> tuple[int, str, str]:
        # Extract lockfile path and ttl. Look for `[ -e {path} ]` and
        # `[ "$age" -lt N ]` and the printf target.
        lock_match = re.search(r"\[ -e (\S+) \]", body)
        ttl_match = re.search(r"\[ \"\$age\" -lt (\d+) \]", body)
        printf_match = re.search(r"printf '%s' (.*) > (\S+)", body)
        if not (lock_match and ttl_match and printf_match):
            return (1, "", "could not parse lock acquire body")
        lockfile = lock_match.group(1).strip("'")
        ttl = int(ttl_match.group(1))
        payload = printf_match.group(1).strip("'")
        target = printf_match.group(2).strip("'")
        # Decide: lock present?
        if lockfile in self.fs.files:
            age = self.fs.now - self.fs.mtimes.get(lockfile, 0)
            if age < ttl:
                existing = self.fs.files[lockfile]
                return (17, "", f"LOCK_BUSY {existing}")
            # Stale takeover: rm and create.
            del self.fs.files[lockfile]
            self.fs.mtimes.pop(lockfile, None)
        # Create.
        self.fs.files[target] = payload
        self.fs.mtimes[target] = self.fs.now
        return (0, "", "")


def test_write_manifest_then_read_manifest_round_trip():
    ssh = _FakeSSH()
    contract = _make_contract()
    manifest = _make_manifest()
    write_manifest(ssh, contract, manifest)
    loaded = read_manifest(ssh, contract)
    assert loaded is not None
    assert loaded == manifest


def test_read_manifest_returns_none_when_missing():
    ssh = _FakeSSH()
    contract = _make_contract()
    assert read_manifest(ssh, contract) is None


def test_read_manifest_returns_none_on_invalid_json():
    ssh = _FakeSSH()
    contract = _make_contract()
    ssh.fs.files[manifest_path(contract)] = "not valid json{"
    assert read_manifest(ssh, contract) is None


def test_read_manifest_filters_unknown_payload_keys():
    ssh = _FakeSSH()
    contract = _make_contract()
    payload = dataclasses.asdict(_make_manifest())
    payload["a_future_field_we_dont_know_about"] = "anything"
    ssh.fs.files[manifest_path(contract)] = json.dumps(payload)
    loaded = read_manifest(ssh, contract)
    assert loaded is not None
    assert loaded.python_version == "3.11"


# --------------------------------------------------------------------------- #
# Atomic-rename simulation: builder crash before final mv leaves manifest intact.
# --------------------------------------------------------------------------- #


def test_atomic_rename_crash_preserves_existing_manifest():
    ssh = _FakeSSH()
    contract = _make_contract()
    original = _make_manifest(built_by="pod-original")
    write_manifest(ssh, contract, original)
    # Now simulate a second builder writing a new manifest but crashing
    # between heredoc-write and mv.
    new_manifest = _make_manifest(built_by="pod-new")
    ssh.crash_after_staging = True
    with pytest.raises(RuntimeError, match="simulated builder crash"):
        write_manifest(ssh, contract, new_manifest)
    # The existing manifest must remain readable and equal to the original.
    survivor = read_manifest(ssh, contract)
    assert survivor is not None
    assert survivor.built_by == "pod-original"


# --------------------------------------------------------------------------- #
# Drift detection — schema_version/bundle_format_version/python_version/cuda_extra
# are HARD-FAIL; pyproject_hash/custom_nodes_lock_hash/comfyui_pin/vibecomfy_commit/
# reigh_worker_commit are DELTA-SYNC.
# --------------------------------------------------------------------------- #


_HARD_FAIL_FIELDS = (
    "schema_version",
    "bundle_format_version",
    "python_version",
    "cuda_extra",
)
_DELTA_SYNC_FIELDS = (
    "pyproject_hash",
    "custom_nodes_lock_hash",
    "comfyui_pin",
    "vibecomfy_commit",
    "reigh_worker_commit",
)


def _drift_value(field: str) -> object:
    sample = _make_manifest()
    original = getattr(sample, field)
    if isinstance(original, int):
        return original + 1
    if isinstance(original, str):
        return original + "-drifted"
    raise AssertionError(f"unhandled field type for {field}: {type(original)}")


@pytest.mark.parametrize("field", _HARD_FAIL_FIELDS)
def test_hard_fail_fields_diverge_when_drifted(field: str):
    """Each hard-fail field must produce a different value when drifted.

    The consumer (variant_prebuilt._check_hard_fail_drift) compares the
    manifest's hard-fail fields to the expected values; if any differ the
    consumer raises rather than delta-syncing. This test confirms the value
    actually changes so the comparison can detect it.
    """
    baseline = _make_manifest()
    drifted = dataclasses.replace(baseline, **{field: _drift_value(field)})
    assert getattr(drifted, field) != getattr(baseline, field)
    assert field in _HARD_FAIL_FIELDS
    # Confirm the field is NOT in the delta-sync bucket — these must be disjoint.
    assert field not in _DELTA_SYNC_FIELDS


@pytest.mark.parametrize("field", _DELTA_SYNC_FIELDS)
def test_delta_sync_fields_diverge_when_drifted(field: str):
    """Each delta-sync field must produce a different value when drifted, so the
    consumer can detect drift via hash comparison and run a sync, not a rebuild."""
    baseline = _make_manifest()
    drifted = dataclasses.replace(baseline, **{field: _drift_value(field)})
    assert getattr(drifted, field) != getattr(baseline, field)
    assert field in _DELTA_SYNC_FIELDS
    assert field not in _HARD_FAIL_FIELDS


def test_hard_fail_and_delta_sync_buckets_are_disjoint():
    assert set(_HARD_FAIL_FIELDS).isdisjoint(_DELTA_SYNC_FIELDS)


# --------------------------------------------------------------------------- #
# build.lock — O_EXCL acquire, release, TTL takeover, concurrent acquire failure.
# --------------------------------------------------------------------------- #


def test_acquire_build_lock_creates_lockfile_and_release_removes_it():
    ssh = _FakeSSH()
    contract = _make_contract()
    release = acquire_build_lock(ssh, contract, holder_id="pod-1", ttl_sec=7200)
    assert lock_path(contract) in ssh.fs.files
    release()
    assert lock_path(contract) not in ssh.fs.files


def test_acquire_build_lock_records_holder_id_and_acquired_at():
    ssh = _FakeSSH()
    contract = _make_contract()
    release = acquire_build_lock(ssh, contract, holder_id="pod-42", ttl_sec=7200)
    try:
        payload = ssh.fs.files[lock_path(contract)]
        parsed = json.loads(payload)
        assert parsed["holder_id"] == "pod-42"
        assert parsed["ttl_sec"] == 7200
        assert "acquired_at" in parsed
    finally:
        release()


def test_concurrent_acquire_within_ttl_fails():
    ssh = _FakeSSH()
    contract = _make_contract()
    first_release = acquire_build_lock(ssh, contract, holder_id="pod-1")
    try:
        with pytest.raises(RuntimeError) as info:
            acquire_build_lock(ssh, contract, holder_id="pod-2")
        # The diagnostic surfaces the existing holder payload.
        assert "pod-1" in str(info.value) or "LOCK_BUSY" in str(info.value)
    finally:
        first_release()


def test_acquire_build_lock_ttl_takeover_after_expiry():
    ssh = _FakeSSH()
    contract = _make_contract()
    first_release = acquire_build_lock(ssh, contract, holder_id="pod-1", ttl_sec=10)
    # Advance the fake clock past TTL.
    ssh.fs.advance(20)
    # A second acquire should succeed (takeover).
    second_release = acquire_build_lock(ssh, contract, holder_id="pod-2", ttl_sec=10)
    try:
        payload = ssh.fs.files[lock_path(contract)]
        parsed = json.loads(payload)
        assert parsed["holder_id"] == "pod-2"
    finally:
        second_release()
    # `first_release` is still safe to call (it just rm -f's the now-missing file).
    first_release()
