"""Unit tests for runpod_lifecycle.shipping — upload/download primitives.

Coverage:
- should_skip: directory prefix match, part match, .pyc/.pyo exclusion, explicit name match
- _build_upload_tarball: creates tarball, respects excludes, produces valid tar.gz
- _preflight_upload_disk: fails with RuntimeError when disk too small, passes when enough
"""

from __future__ import annotations

import os
import shutil
import tarfile
import tempfile
from pathlib import Path

import pytest

from runpod_lifecycle.shipping import (
    _build_upload_tarball,
    _preflight_upload_disk,
    should_skip,
)


# ---------------------------------------------------------------------------
# should_skip
# ---------------------------------------------------------------------------

class TestShouldSkip:
    """Edge cases for the should_skip function."""

    def test_exact_name_match(self, tmp_path: Path) -> None:
        """An exact name match in the exclude set causes a skip."""
        root = tmp_path
        (root / "__pycache__").mkdir()
        assert should_skip(root / "__pycache__", root, {"__pycache__"}) is True

    def test_directory_prefix_match(self, tmp_path: Path) -> None:
        """A file inside an excluded directory is skipped."""
        root = tmp_path
        (root / ".git").mkdir()
        (root / ".git" / "config").write_text("data")
        assert should_skip(root / ".git" / "config", root, {".git"}) is True

    def test_part_match_in_parts(self, tmp_path: Path) -> None:
        """If any part of the path matches an exclude, skip."""
        root = tmp_path
        (root / "node_modules").mkdir()
        (root / "node_modules" / "pkg").mkdir()
        (root / "node_modules" / "pkg" / "index.js").write_text("//")
        assert should_skip(root / "node_modules" / "pkg" / "index.js", root, {"node_modules"}) is True

    def test_no_match_returns_false(self, tmp_path: Path) -> None:
        """A file that doesn't match any exclude rule is not skipped."""
        root = tmp_path
        (root / "src").mkdir()
        (root / "src" / "main.py").write_text("print('ok')")
        assert should_skip(root / "src" / "main.py", root, {".git", "__pycache__"}) is False

    def test_pyc_suffix_auto_skipped(self, tmp_path: Path) -> None:
        """.pyc and .pyo files are always skipped regardless of exclude set."""
        root = tmp_path
        (root / "module.pyc").write_text("")
        (root / "module.pyo").write_text("")
        assert should_skip(root / "module.pyc", root, set()) is True
        assert should_skip(root / "module.pyo", root, set()) is True

    def test_non_pyc_py_file_not_skipped(self, tmp_path: Path) -> None:
        """Regular .py files are not auto-skipped."""
        root = tmp_path
        (root / "script.py").write_text("x=1")
        assert should_skip(root / "script.py", root, set()) is False

    def test_subpath_containing_excluded_name_not_skipped(self, tmp_path: Path) -> None:
        """A file whose path contains an exclude string as substring but not as a part is NOT skipped."""
        root = tmp_path
        (root / "my_git_backup").mkdir()
        # "my_git_backup" contains "git" as substring but not as a path part named ".git"
        (root / "my_git_backup" / "data.txt").write_text("")
        assert should_skip(root / "my_git_backup" / "data.txt", root, {".git"}) is False


# ---------------------------------------------------------------------------
# _build_upload_tarball
# ---------------------------------------------------------------------------

class TestBuildUploadTarball:

    def test_respects_excludes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Files matching exclude patterns are not included in the tarball."""
        root = tmp_path / "payload"
        root.mkdir()
        (root / "keep.txt").write_text("keep me")
        (root / "skip.pyc").write_text("skip")
        (root / ".hidden").mkdir()
        (root / ".hidden" / "secret.txt").write_text("secret")

        # Use a temp dir for the archive to avoid disk preflight issues
        tmpdir = tmp_path / "tmp"
        tmpdir.mkdir()
        monkeypatch.setenv("RUNPOD_LIFECYCLE_UPLOAD_TMPDIR", str(tmpdir))
        monkeypatch.setenv("RUNPOD_LIFECYCLE_UPLOAD_MIN_FREE_BYTES", "1")

        tar_path = _build_upload_tarball({".hidden", ".pyc"}, root=root)

        try:
            assert tar_path.exists()
            with tarfile.open(tar_path, "r:gz") as tar:
                names = tar.getnames()
                assert "keep.txt" in names
                assert "skip.pyc" not in names
                assert ".hidden" not in names
                assert ".hidden/secret.txt" not in names
        finally:
            tar_path.unlink(missing_ok=True)

    def test_creates_valid_tarball(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """_build_upload_tarball produces a valid gzipped tar file."""
        root = tmp_path / "payload2"
        root.mkdir()
        (root / "a.txt").write_text("aaa")
        (root / "sub").mkdir()
        (root / "sub" / "b.txt").write_text("bbb")

        tmpdir = tmp_path / "tmp2"
        tmpdir.mkdir()
        monkeypatch.setenv("RUNPOD_LIFECYCLE_UPLOAD_TMPDIR", str(tmpdir))
        monkeypatch.setenv("RUNPOD_LIFECYCLE_UPLOAD_MIN_FREE_BYTES", "1")

        tar_path = _build_upload_tarball(set(), root=root)
        try:
            assert tar_path.suffixes == [".tar", ".gz"]
            with tarfile.open(tar_path, "r:gz") as tar:
                names = sorted(tar.getnames())
                assert "a.txt" in names
                assert "sub/b.txt" in names
        finally:
            tar_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# _preflight_upload_disk
# ---------------------------------------------------------------------------

class TestPreflightUploadDisk:

    def test_raises_when_disk_too_small(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """_preflight_upload_disk raises RuntimeError when free space is insufficient."""
        temp_dir = tmp_path / "small_disk"
        temp_dir.mkdir()

        # Force a very high min_free_bytes so the check fails
        monkeypatch.setenv("RUNPOD_LIFECYCLE_UPLOAD_MIN_FREE_BYTES", str(10 * 1024**4))  # 10 TiB

        with pytest.raises(RuntimeError, match="insufficient local disk"):
            _preflight_upload_disk(temp_dir, 1024 * 1024 * 1024)  # 1 GiB estimated

    def test_passes_when_disk_has_space(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """_preflight_upload_disk does not raise when free space is sufficient."""
        temp_dir = tmp_path / "big_disk"
        temp_dir.mkdir()

        # Set min_free_bytes very low so the check passes
        monkeypatch.setenv("RUNPOD_LIFECYCLE_UPLOAD_MIN_FREE_BYTES", "1")

        # Should not raise
        _preflight_upload_disk(temp_dir, 1024)  # tiny estimate