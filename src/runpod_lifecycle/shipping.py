"""Upload/download primitives for shipping payloads to/from RunPod pods."""

from __future__ import annotations

import asyncio
import os
import posixpath
import re
import shlex
import shutil
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

MiB = 1024 * 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_bytes(value: int) -> str:
    if value >= 1024 * MiB:
        return f"{value / (1024 * MiB):.1f}GiB"
    if value >= MiB:
        return f"{value / MiB:.1f}MiB"
    if value >= 1024:
        return f"{value / 1024:.1f}KiB"
    return f"{value}B"


def _log_phase(name: str, detail: str = "") -> None:
    suffix = f" {detail}" if detail else ""
    print(f"phase={name}{suffix}", flush=True)


def _scoped_token(value: str | None) -> str:
    """Return a short shell-safe token for one transport attempt."""
    raw = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "")).strip("-")
    nonce = uuid.uuid4().hex[:12]
    prefix = raw[: max(1, 64 - len(nonce) - 1)]
    return f"{prefix}-{nonce}" if prefix else nonce


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    """Extract only regular files/directories contained by *destination*.

    Validation is completed for every member before the first write. Symlinks,
    hardlinks, device nodes, absolute paths, parent traversal, and duplicate
    destinations are rejected because the archive is remote input.
    """
    root = destination.resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        validated: list[tuple[tarfile.TarInfo, Path]] = []
        seen: set[Path] = set()
        for member in members:
            name = member.name
            relative = Path(name)
            if (
                not name
                or relative.is_absolute()
                or ".." in relative.parts
                or member.issym()
                or member.islnk()
                or not (member.isdir() or member.isfile())
            ):
                raise RuntimeError(f"unsafe artifact archive member: {name!r}")
            target = (root / relative).resolve(strict=False)
            if target != root and not target.is_relative_to(root):
                raise RuntimeError(f"artifact archive member escapes custody: {name!r}")
            if target in seen:
                raise RuntimeError(f"duplicate artifact archive member: {name!r}")
            seen.add(target)
            validated.append((member, target))

        for member, target in validated:
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                raise RuntimeError(f"artifact archive member is unreadable: {member.name!r}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


# ---------------------------------------------------------------------------
# UploadHeartbeat
# ---------------------------------------------------------------------------

DEFAULT_UPLOAD_PROGRESS_SECONDS = 10.0
DEFAULT_UPLOAD_PROGRESS_FILES = 250


class UploadHeartbeat:
    """Progress reporter for long-running upload operations."""

    def __init__(self, *, label: str) -> None:
        self.label = label
        self.files = 0
        self.bytes = 0
        self.start = time.monotonic()
        self.last_log = self.start
        self.every_seconds = float(
            os.getenv(
                "RUNPOD_LIFECYCLE_UPLOAD_PROGRESS_SECONDS",
                str(DEFAULT_UPLOAD_PROGRESS_SECONDS),
            )
        )
        self.every_files = max(
            1,
            int(
                os.getenv(
                    "RUNPOD_LIFECYCLE_UPLOAD_PROGRESS_FILES",
                    str(DEFAULT_UPLOAD_PROGRESS_FILES),
                )
            ),
        )

    def tick(self, *, files: int = 0, bytes_added: int = 0, force: bool = False) -> None:
        self.files += files
        self.bytes += bytes_added
        now = time.monotonic()
        if force or self.files % self.every_files == 0 or now - self.last_log >= self.every_seconds:
            elapsed = max(now - self.start, 0.001)
            print(
                f"{self.label}_progress files={self.files} bytes={self.bytes} "
                f"size={_format_bytes(self.bytes)} elapsed_seconds={elapsed:.1f}",
                flush=True,
            )
            self.last_log = now


# ---------------------------------------------------------------------------
# Exclude / skip logic
# ---------------------------------------------------------------------------

def should_skip(path: Path, root: Path, exclude_set: set[str]) -> bool:
    """Return ``True`` if *path* under *root* matches an exclusion rule."""
    rel = path.relative_to(root).as_posix()
    parts = Path(rel).parts
    return (
        any(rel == item or rel.startswith(f"{item}/") or item in parts for item in exclude_set)
        or path.suffix in {".pyc", ".pyo"}
    )


# ---------------------------------------------------------------------------
# SFTP-walk upload
# ---------------------------------------------------------------------------

def upload_dir(
    sftp,
    local: Path,
    remote: str,
    exclude_set: set[str],
    *,
    progress: UploadHeartbeat | None = None,
    local_root: Path,
) -> None:
    """Recursively upload *local* directory tree to *remote* via SFTP."""
    try:
        sftp.mkdir(remote)
    except OSError:
        pass
    for child in local.iterdir():
        if should_skip(child, local_root, exclude_set):
            continue
        remote_child = posixpath.join(remote, child.name)
        if child.is_dir():
            upload_dir(sftp, child, remote_child, exclude_set, progress=progress, local_root=local_root)
        else:
            size = child.stat().st_size
            if progress is not None:
                sent = 0

                def _progress(current: int, _total: int) -> None:
                    nonlocal sent
                    delta = max(0, current - sent)
                    sent = current
                    progress.tick(bytes_added=delta)

                sftp.put(str(child), remote_child, callback=_progress)
                progress.tick(files=1, bytes_added=max(0, size - sent))
            else:
                sftp.put(str(child), remote_child)


# ---------------------------------------------------------------------------
# Tarball upload helpers
# ---------------------------------------------------------------------------

def _upload_tmpdir() -> Path:
    configured = os.getenv("RUNPOD_LIFECYCLE_UPLOAD_TMPDIR")
    temp_dir = Path(configured).expanduser() if configured else Path(tempfile.gettempdir())
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def _iter_upload_files(root: Path, exclude: set[str]) -> Iterable[Path]:
    for current, dir_names, file_names in os.walk(root):
        current_path = Path(current)
        dir_names[:] = [
            name for name in dir_names if not should_skip(current_path / name, root, exclude)
        ]
        for file_name in file_names:
            path = current_path / file_name
            if should_skip(path, root, exclude):
                continue
            yield path


def _estimate_upload_payload(root: Path, exclude: set[str]) -> tuple[list[Path], int]:
    files: list[Path] = []
    total_bytes = 0
    heartbeat = UploadHeartbeat(label="upload_scan")
    for path in _iter_upload_files(root, exclude):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        files.append(path)
        total_bytes += size
        heartbeat.tick(files=1, bytes_added=size)
    heartbeat.tick(force=True)
    return files, total_bytes


def _preflight_upload_disk(temp_dir: Path, estimated_bytes: int) -> None:
    usage = shutil.disk_usage(temp_dir)
    min_free_bytes = int(
        os.getenv("RUNPOD_LIFECYCLE_UPLOAD_MIN_FREE_BYTES", str(512 * MiB))
    )
    required_free = max(min_free_bytes, int(estimated_bytes * 0.25))
    if usage.free >= required_free:
        return
    raise RuntimeError(
        "insufficient local disk for RunPod upload tarball: "
        f"tmpdir={temp_dir} free={_format_bytes(usage.free)} "
        f"estimated_payload={_format_bytes(estimated_bytes)} required_free={_format_bytes(required_free)}. "
        "Free disk space, set RUNPOD_LIFECYCLE_UPLOAD_TMPDIR to a larger volume, "
        "or add excludes for bulky local paths."
    )


def _build_upload_tarball(exclude: set[str], *, root: Path) -> Path:
    _log_phase("building_upload", f"mode=tarball root={root}")
    temp_dir = _upload_tmpdir()
    files, estimated_bytes = _estimate_upload_payload(root, exclude)
    _log_phase(
        "building_upload",
        f"files={len(files)} estimated_payload={_format_bytes(estimated_bytes)} tmpdir={temp_dir}",
    )
    _preflight_upload_disk(temp_dir, estimated_bytes)

    handle = tempfile.NamedTemporaryFile(
        prefix="runpod-lifecycle-upload-", suffix=".tar.gz", dir=temp_dir, delete=False
    )
    tar_path = Path(handle.name)
    handle.close()
    heartbeat = UploadHeartbeat(label="tarball_build")
    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            for path in files:
                tar.add(path, arcname=path.relative_to(root))
                try:
                    size = path.stat().st_size
                except OSError:
                    size = 0
                heartbeat.tick(files=1, bytes_added=size)
        heartbeat.tick(force=True)
        _log_phase(
            "building_upload",
            f"archive={tar_path} size={_format_bytes(tar_path.stat().st_size)}",
        )
    except Exception:
        tar_path.unlink(missing_ok=True)
        raise
    return tar_path


# ---------------------------------------------------------------------------
# Tarball upload (full round-trip)
# ---------------------------------------------------------------------------

async def _upload_tarball(
    pod,
    exclude: set[str],
    *,
    local_root: Path,
    remote_root: str,
) -> dict[str, Any]:
    tar_path = _build_upload_tarball(exclude, root=local_root)
    remote_archive = "/tmp/runpod-lifecycle-upload.tar.gz"
    upload_info = {
        "mode": "tarball",
        "local_archive_path": tar_path.as_posix(),
        "remote_archive_path": remote_archive,
        "archive_bytes": tar_path.stat().st_size,
        "excludes": sorted(exclude),
    }
    try:
        _log_phase("uploading", f"mode=tarball size={_format_bytes(tar_path.stat().st_size)}")
        client = pod.open_ssh_client()
        try:
            sftp = client.open_sftp()
            try:
                progress = UploadHeartbeat(label="tarball_upload")

                def _progress(sent: int, total: int) -> None:
                    progress.bytes = sent
                    progress.files = 1
                    now = time.monotonic()
                    if sent == total or now - progress.last_log >= progress.every_seconds:
                        print(
                            f"tarball_upload_progress bytes={sent} total_bytes={total} "
                            f"size={_format_bytes(sent)} total_size={_format_bytes(total)}",
                            flush=True,
                        )
                        progress.last_log = now

                sftp.put(
                    str(tar_path), remote_archive, callback=_progress, confirm=False
                )
            finally:
                sftp.close()
        finally:
            client.close()
        _log_phase("extracting", f"remote_root={remote_root}")
        quoted_remote_root = shlex.quote(remote_root)
        code, stdout, stderr = await pod.exec_ssh(
            f"rm -rf {quoted_remote_root} && mkdir -p {quoted_remote_root} && "
            f"tar --no-same-owner -xzf {remote_archive} -C {quoted_remote_root}",
            timeout=300,
        )
        if code != 0:
            print(stdout, flush=True)
            print(stderr, flush=True)
            raise RuntimeError(f"remote tarball extraction failed with exit code {code}")
        return upload_info
    finally:
        tar_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Remote script upload
# ---------------------------------------------------------------------------

async def _upload_remote_script(pod, remote_script: str) -> None:
    handle = tempfile.NamedTemporaryFile(
        prefix="runpod-lifecycle-remote-run-", suffix=".sh", mode="w", delete=False
    )
    script_path = Path(handle.name)
    remote_script_path = "/tmp/runpod-lifecycle-remote-run.sh"
    try:
        handle.write(remote_script)
        handle.write("\n")
        handle.close()
        client = pod.open_ssh_client()
        try:
            sftp = client.open_sftp()
            try:
                sftp.put(str(script_path), remote_script_path, confirm=False)
            finally:
                sftp.close()
        finally:
            client.close()
        code, stdout, stderr = await pod.exec_ssh(
            f"chmod +x {remote_script_path}", timeout=30
        )
        if code != 0:
            print(stdout, flush=True)
            print(stderr, flush=True)
            raise RuntimeError(f"remote script chmod failed with exit code {code}")
    finally:
        script_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Artifact download
# ---------------------------------------------------------------------------

async def download_artifact_archive(
    pod,
    *,
    remote_root: str,
    artifact_paths: list[str],
    local_artifact_root: Path,
    remote_archive_path: str | None = None,
    scope: str | None = None,
    exit_code: int | None = None,
    remote_command: str | None = None,
    upload: dict[str, Any] | None = None,
) -> Path | None:
    """Download artifact directories from *pod*, archives them remotely,
    pulls the tarball via SFTP, and extracts into *local_artifact_root*.

    Returns *local_artifact_root* on success, ``None`` on failure.
    """
    local_artifact_root.mkdir(parents=True, exist_ok=True)
    token = _scoped_token(scope or f"{local_artifact_root.name}-{uuid.uuid4().hex[:12]}")
    remote_archive_path = remote_archive_path or f"/tmp/runpod-lifecycle-artifacts-{token}.tar.gz"
    remote_error_path = f"/tmp/runpod-lifecycle-artifacts-{token}.err"
    _log_phase(
        "downloading_artifacts",
        f"local={local_artifact_root} remote_root={remote_root} paths={artifact_paths}",
    )

    # Build remote archive command
    paths_str = " ".join(shlex.quote(path) for path in artifact_paths) or "''"
    quoted_root = shlex.quote(remote_root)
    quoted_archive = shlex.quote(remote_archive_path)
    quoted_error = shlex.quote(remote_error_path)
    empty_root = shlex.quote(f"/tmp/runpod-lifecycle-empty-artifacts-{token}")
    cmd = (
        f"cd {quoted_root} || exit $?; "
        f"set --; "
        f"for path in {paths_str}; do if [ -e \"$path\" ]; then set -- \"$@\" \"$path\"; fi; done; "
        f"if [ \"$#\" -eq 0 ]; then mkdir -p {empty_root} && "
        f"tar -czf {quoted_archive} -C {empty_root} .; "
        f"elif ! tar -czf {quoted_archive} -- \"$@\" 2>{quoted_error}; then "
        f"cat {quoted_error}; exit 1; fi"
    )

    code, stdout, stderr = await pod.exec_ssh(cmd, timeout=300)
    if code != 0:
        print(stdout, flush=True)
        if stderr.strip():
            print(stderr, flush=True)
        print("artifact_download_failed=archive", flush=True)
        return None

    client = pod.open_ssh_client()
    try:
        sftp = client.open_sftp()
        try:
            archive = local_artifact_root / "artifacts.tar.gz"
            sftp.get(remote_archive_path, str(archive))
        finally:
            sftp.close()
    except Exception as exc:
        print(f"artifact_download_failed={exc}", flush=True)
        return None
    finally:
        client.close()

    try:
        _safe_extract_tar(local_artifact_root / "artifacts.tar.gz", local_artifact_root)
    except (OSError, RuntimeError, tarfile.TarError) as exc:
        print(f"artifact_download_failed={exc}", flush=True)
        return None

    print(f"artifact_downloaded={local_artifact_root}", flush=True)
    return local_artifact_root
