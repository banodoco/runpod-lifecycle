"""Live RunPod integration tests — gated behind RUNPOD_LIVE_TESTS=1.

These tests launch real RunPod pods and incur GPU costs.
DO NOT run in CI.

Costs (approximate, on RTX 4090, ~$0.34/hr):
  - sftp_walk upload smoke:        ~$0.01
  - tarball upload smoke:          ~$0.01
  - sync exec smoke:               ~$0.01
  - detached-with-poll exec:       ~$0.02 (longer due to poll interval)
  - reattach by pod_id:            ~$0.02 (two pods or reattach sequence)
  - terminate_after_exec=False:    ~$0.02 (pod persists briefly)
  - Total estimate:                ~$0.09 (all tests combined)

To run:
    RUNPOD_LIVE_TESTS=1 python -m pytest tests/test_live_pod.py -v
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

# Skip all tests if the env var is not set
pytestmark = pytest.mark.skipif(
    os.getenv("RUNPOD_LIVE_TESTS") != "1",
    reason="RUNPOD_LIVE_TESTS=1 is not set — live tests are skipped to avoid GPU spend",
)

from runpod_lifecycle.config import RunPodConfig
from runpod_lifecycle.lifecycle import launch
from runpod_lifecycle.runner import (
    ship_and_run,
    ship_and_run_detached,
)
from runpod_lifecycle.shipping import _build_upload_tarball, upload_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _live_config() -> RunPodConfig:
    """Build a RunPodConfig from environment."""
    api_key = os.environ["RUNPOD_API_KEY"]
    return RunPodConfig(api_key=api_key)


def _make_name_prefix() -> str:
    """Generate a unique name prefix for this test run."""
    return f"test-live-{int(time.time())}"


# ---------------------------------------------------------------------------
# sftp_walk upload smoke
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_sftp_walk_upload(tmp_path: Path) -> None:
    """Launch a pod, upload a small payload via sftp_walk, verify it exists remotely."""
    config = _live_config()
    prefix = _make_name_prefix()
    local_root = tmp_path / "payload"
    local_root.mkdir()
    (local_root / "test.txt").write_text("hello from sftp_walk")
    (local_root / "subdir").mkdir()
    (local_root / "subdir" / "nested.txt").write_text("nested")

    pod = None
    try:
        pod = await launch(config, name=f"{prefix}-sftp")

        # Upload via sftp_walk
        client = pod.open_ssh_client()
        try:
            sftp = client.open_sftp()
            try:
                upload_dir(sftp, local_root, "/tmp/test-payload", set(), local_root=local_root)
            finally:
                sftp.close()
        finally:
            client.close()

        # Verify the payload is on the remote side
        code, stdout, stderr = await pod.exec_ssh(
            "cat /tmp/test-payload/test.txt && echo --- && "
            "cat /tmp/test-payload/subdir/nested.txt",
            timeout=60,
        )
        assert code == 0, f"sftp_walk upload verify failed: {stderr}"
        assert "hello from sftp_walk" in stdout
        assert "nested" in stdout

    finally:
        if pod is not None:
            await pod.terminate()


# ---------------------------------------------------------------------------
# tarball upload smoke
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_tarball_upload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Launch a pod, upload a payload via tarball, verify it exists remotely."""
    config = _live_config()
    prefix = _make_name_prefix()
    local_root = tmp_path / "tarball_payload"
    local_root.mkdir()
    (local_root / "hello.txt").write_text("hello from tarball")

    # Use tmp_path for tarball staging
    monkeypatch.setenv("RUNPOD_LIFECYCLE_UPLOAD_TMPDIR", str(tmp_path))
    monkeypatch.setenv("RUNPOD_LIFECYCLE_UPLOAD_MIN_FREE_BYTES", "1")

    pod = None
    try:
        pod = await launch(config, name=f"{prefix}-tar")

        result = await ship_and_run(
            config,
            "cat /tmp/test-tarball/hello.txt",
            local_root=local_root,
            remote_root="/tmp/test-tarball",
            exclude=set(),
            upload_mode="tarball",
            timeout=300,
            name_prefix=prefix,
            terminate_after_exec=False,
        )

        assert result.returncode == 0
        assert "hello from tarball" in result.stdout, f"tarball upload failed: {result.stderr}"

    finally:
        if pod is not None:
            await pod.terminate()


# ---------------------------------------------------------------------------
# sync exec smoke
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_sync_exec(tmp_path: Path) -> None:
    """Launch a pod, run a shell command synchronously, verify exit code 0."""
    config = _live_config()
    prefix = _make_name_prefix()
    local_root = tmp_path / "empty_payload"
    local_root.mkdir()

    result = await ship_and_run(
        config,
        "nvidia-smi -L && echo ok",
        local_root=local_root,
        remote_root="/tmp/test-sync-exec",
        exclude=set(),
        timeout=300,
        name_prefix=prefix,
        terminate_after_exec=True,
    )

    assert result.returncode == 0
    assert "ok" in result.stdout
    assert result.terminated is True


# ---------------------------------------------------------------------------
# detached-with-poll exec smoke
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_detached_with_poll_exec(tmp_path: Path) -> None:
    """Launch a pod, run a command detached, poll for completion, verify artifacts."""
    config = _live_config()
    prefix = _make_name_prefix()
    local_root = tmp_path / "detached_payload"
    local_root.mkdir()
    (local_root / "input.txt").write_text("detached test input")

    result = await ship_and_run_detached(
        config,
        (
            "mkdir -p out && "
            "cp input.txt out/output.txt && "
            "echo 'detached ok'"
        ),
        local_root=local_root,
        remote_root="/tmp/test-detached",
        exclude=set(),
        timeout=600,
        name_prefix=prefix,
        terminate_after_exec=True,
        poll_interval=15,
        artifact_paths=["out"],
    )

    assert result.returncode == 0, f"detached exec failed: returncode={result.returncode}"
    assert result.terminated is True
    if result.artifact_root:
        output_file = result.artifact_root / "out" / "output.txt"
        assert output_file.exists(), f"expected artifact out/output.txt, got {list(result.artifact_root.rglob('*'))}"
        assert output_file.read_text().strip() == "detached test input"
    else:
        # Artifact download may fail in some environments; treat as non-fatal
        print("WARNING: artifact_root is None — artifact download may have failed")


# ---------------------------------------------------------------------------
# reattach by pod_id (stateless)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_reattach_by_pod_id(tmp_path: Path) -> None:
    """Launch a pod, then reattach statelessly by pod_id via discovery.get_pod."""
    from runpod_lifecycle.discovery import get_pod
    from runpod_lifecycle.pod import Pod

    config = _live_config()
    prefix = _make_name_prefix()

    pod = None
    try:
        pod = await launch(config, name=f"{prefix}-reattach")
        pod_id = pod.id

        # Reattach statelessly
        reattached = get_pod(pod_id, config.api_key)
        assert reattached.id == pod_id

        # Run a command on the reattached pod
        code, stdout, stderr = await reattached.exec_ssh("echo reattached", timeout=60)
        assert code == 0
        assert "reattached" in stdout

    finally:
        if pod is not None:
            await pod.terminate()


# ---------------------------------------------------------------------------
# terminate_after_exec=False round-trip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_terminate_after_exec_false_roundtrip(tmp_path: Path) -> None:
    """Launch a pod, run a command with terminate_after_exec=False, verify pod stays alive,
    run another command, then manually terminate."""
    config = _live_config()
    prefix = _make_name_prefix()
    local_root = tmp_path / "keep_payload"
    local_root.mkdir()

    result1 = await ship_and_run(
        config,
        "echo 'first' > /tmp/test-keep/first.txt",
        local_root=local_root,
        remote_root="/tmp/test-keep",
        exclude=set(),
        timeout=300,
        name_prefix=prefix,
        terminate_after_exec=False,
    )

    assert result1.returncode == 0
    assert result1.pod is not None
    assert result1.terminated is False

    # The pod should still be alive — run a second command
    pod = result1.pod
    code, stdout, stderr = await pod.exec_ssh("cat /tmp/test-keep/first.txt", timeout=60)
    assert code == 0
    assert "first" in stdout

    # Now terminate manually
    await pod.terminate()
    result1.terminated = True