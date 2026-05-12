"""Unit tests for runpod_lifecycle.runner — ship_and_run, ship_and_run_detached.

Coverage:
- ship_and_run with terminate_after_exec=True → pod is terminated
- ship_and_run with terminate_after_exec=False → pod returned, NOT terminated
- guard_factory mock → custom guard used
- CancelledError path → result.returncode == 130
- ship_and_run_detached with parameterized poll targets
- ship_and_run_detached timeout path → returncode 124
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from runpod_lifecycle.config import RunPodConfig
from runpod_lifecycle.guard import PodGuard
from runpod_lifecycle.runner import (
    ShipAndRunResult,
    ship_and_run,
    ship_and_run_detached,
    _parse_detached_exit,
)


# ---------------------------------------------------------------------------
# _parse_detached_exit
# ---------------------------------------------------------------------------

class TestParseDetachedExit:
    def test_positive_integer(self) -> None:
        assert _parse_detached_exit("0") == 0
        assert _parse_detached_exit("42") == 42

    def test_negative_integer(self) -> None:
        assert _parse_detached_exit("-1") == -1
        assert _parse_detached_exit("-15") == -15

    def test_with_whitespace(self) -> None:
        assert _parse_detached_exit("  42  \n") == 42

    def test_non_integer_returns_none(self) -> None:
        assert _parse_detached_exit("") is None
        assert _parse_detached_exit("not a number") is None
        assert _parse_detached_exit("0.5") is None

    def test_multiline_takes_first(self) -> None:
        assert _parse_detached_exit("42\n99\n") == 42


# ---------------------------------------------------------------------------
# Helper: build a mock pod that satisfies the ship_and_run flow
# ---------------------------------------------------------------------------

def _make_mock_pod(pod_id: str = "pod-test") -> MagicMock:
    """Create a mock Pod with all required methods."""
    mock_pod = MagicMock()
    mock_pod.id = pod_id
    mock_pod.wait_ready = AsyncMock()
    mock_pod._ensure_ssh_details = AsyncMock(return_value={
        "ip": "1.2.3.4", "port": 2201, "password": "***"
    })
    mock_pod.exec_ssh = AsyncMock(return_value=(0, "ok", ""))
    mock_pod.terminate = AsyncMock()

    # SFTP upload: open_ssh_client → open_sftp → upload_dir → close
    mock_sftp = MagicMock()
    mock_client = MagicMock()
    mock_client.open_sftp.return_value = mock_sftp
    mock_pod.open_ssh_client.return_value = mock_client

    return mock_pod


# ---------------------------------------------------------------------------
# ship_and_run — terminate_after_exec=True
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ship_and_run_terminates_pod_by_default(tmp_path: Path) -> None:
    """When terminate_after_exec=True (default), the pod is terminated after exec.

    Note: result.pod is set during execution (for guard.attach) but result.terminated
    is the authoritative signal that teardown happened.
    """
    config = RunPodConfig(api_key="***")
    mock_pod = _make_mock_pod("pod-sar-1")
    local_root = tmp_path / "local"
    local_root.mkdir()

    with patch("runpod_lifecycle.runner._launch_pod", new_callable=AsyncMock) as mock_launch:
        mock_launch.return_value = mock_pod

        result = await ship_and_run(
            config,
            "echo ok",
            local_root=local_root,
            remote_root="/tmp/remote",
            exclude=set(),
            timeout=30,
        )

    assert result.returncode == 0
    assert result.terminated is True
    mock_launch.assert_called_once()
    mock_pod.terminate.assert_called()


# ---------------------------------------------------------------------------
# ship_and_run — terminate_after_exec=False
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ship_and_run_keeps_pod_when_terminate_false(tmp_path: Path) -> None:
    """When terminate_after_exec=False, the pod is NOT terminated and is returned."""
    config = RunPodConfig(api_key="***")
    mock_pod = _make_mock_pod("pod-sar-2")
    local_root = tmp_path / "local2"
    local_root.mkdir()

    with patch("runpod_lifecycle.runner._launch_pod", new_callable=AsyncMock) as mock_launch:
        mock_launch.return_value = mock_pod

        result = await ship_and_run(
            config,
            "echo ok",
            local_root=local_root,
            remote_root="/tmp/remote",
            exclude=set(),
            timeout=30,
            terminate_after_exec=False,
        )

    assert result.returncode == 0
    assert result.terminated is False
    assert result.pod is mock_pod  # pod returned
    mock_pod.terminate.assert_not_called()


# ---------------------------------------------------------------------------
# guard_factory mock
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ship_and_run_guard_factory_mock(tmp_path: Path) -> None:
    """When guard_factory is provided, it is used instead of PodGuard."""
    config = RunPodConfig(api_key="***")
    mock_pod = _make_mock_pod("pod-sar-3")
    local_root = tmp_path / "local3"
    local_root.mkdir()

    mock_guard = MagicMock(spec=PodGuard)
    mock_guard.breach_log = []
    mock_guard.terminate = AsyncMock()

    def fake_factory(**kwargs) -> MagicMock:
        return mock_guard

    with patch("runpod_lifecycle.runner._launch_pod", new_callable=AsyncMock) as mock_launch:
        mock_launch.return_value = mock_pod

        result = await ship_and_run(
            config,
            "echo ok",
            local_root=local_root,
            remote_root="/tmp/remote",
            exclude=set(),
            timeout=30,
            guard_factory=fake_factory,
        )

    assert result.returncode == 0
    mock_guard.attach.assert_called_once_with(mock_pod)
    mock_guard.terminate.assert_called_once()


# ---------------------------------------------------------------------------
# CancelledError path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ship_and_run_cancelled_error_returns_130(tmp_path: Path) -> None:
    """When ship_and_run receives a CancelledError, it returns 130.

    We make the exec_ssh call actually sleep so it can be interrupted by cancellation.
    """
    config = RunPodConfig(api_key="***")
    mock_pod = _make_mock_pod("pod-sar-4")
    local_root = tmp_path / "local4"
    local_root.mkdir()

    # Make exec_ssh sleep so cancellation can interrupt
    cancelled = False

    async def slow_exec(cmd: str, timeout: int = 600) -> tuple[int, str, str]:
        nonlocal cancelled
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            cancelled = True
            raise
        return (0, "ok", "")

    mock_pod.exec_ssh = slow_exec

    with patch("runpod_lifecycle.runner._launch_pod", new_callable=AsyncMock) as mock_launch:
        mock_launch.return_value = mock_pod

        task = asyncio.create_task(ship_and_run(
            config,
            "echo ok",
            local_root=local_root,
            remote_root="/tmp/remote",
            exclude=set(),
            timeout=30,
        ))

        # Let it start, then cancel
        await asyncio.sleep(0.05)
        task.cancel()

        result = await task

    assert result.returncode == 130


# ---------------------------------------------------------------------------
# ship_and_run_detached — parameterized poll targets
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ship_and_run_detached_with_parameterized_poll(tmp_path: Path) -> None:
    """ship_and_run_detached uses the poll_command_template and poll_exit_marker."""
    config = RunPodConfig(api_key="***")
    mock_pod = _make_mock_pod("pod-det-1")
    local_root = tmp_path / "local5"
    local_root.mkdir()

    # First exec_ssh for nvidia-smi returns 0 (ok), subsequent for poll returns "42"
    exec_results = iter([
        (0, "GPU 0: ...", ""),   # nvidia-smi
        (0, "12345\n", ""),       # nohup launch pid
        (0, "42", ""),            # poll response
    ])

    async def fake_exec(cmd: str, timeout: int = 600) -> tuple[int, str, str]:
        return next(exec_results)

    mock_pod.exec_ssh = fake_exec

    custom_marker = "/tmp/my_custom_exit_code"

    with patch("runpod_lifecycle.runner._launch_pod", new_callable=AsyncMock) as mock_launch:
        mock_launch.return_value = mock_pod

        with patch("runpod_lifecycle.runner.download_artifact_archive", new_callable=AsyncMock) as mock_download:
            mock_download.return_value = tmp_path / "artifacts"

            # Patch _upload_remote_script to avoid filesystem ops
            with patch("runpod_lifecycle.runner._upload_remote_script", new_callable=AsyncMock):
                result = await ship_and_run_detached(
                    config,
                    "echo ok",
                    local_root=local_root,
                    remote_root="/tmp/remote",
                    exclude=set(),
                    timeout=300,
                    terminate_after_exec=True,
                    poll_interval=1,
                    poll_exit_marker=custom_marker,
                    artifact_paths=["out"],
                )

    assert result.returncode == 42  # parsed from polled "42"
    assert result.terminated is True


@pytest.mark.asyncio
async def test_ship_and_run_detached_timeout_returns_124(tmp_path: Path) -> None:
    """When the detached command exceeds timeout, returncode 124 is returned."""
    config = RunPodConfig(api_key="***")
    mock_pod = _make_mock_pod("pod-det-2")
    local_root = tmp_path / "local6"
    local_root.mkdir()

    # exec_ssh calls: nvidia-smi, nohup launch, then N poll calls (all empty)
    call_count = [0]

    async def fake_exec(cmd: str, timeout: int = 600) -> tuple[int, str, str]:
        call_count[0] += 1
        if call_count[0] == 1:
            return (0, "GPU 0: ...", "")   # nvidia-smi
        elif call_count[0] == 2:
            return (0, "12345\n", "")      # nohup launch
        else:
            return (0, "", "")              # poll: no exit code

    mock_pod.exec_ssh = fake_exec

    with patch("runpod_lifecycle.runner._launch_pod", new_callable=AsyncMock) as mock_launch:
        mock_launch.return_value = mock_pod

        # Use a callable side_effect that only returns the timeout value after
        # enough calls to get past the upload phase. The asyncio event loop also
        # calls time.monotonic, so we can't use a fixed list.
        call_count = [0]

        def fake_monotonic() -> float:
            call_count[0] += 1
            if call_count[0] > 100:
                return 1000.0  # trigger timeout
            return 1.0

        with patch("runpod_lifecycle.runner.time.monotonic", side_effect=fake_monotonic):
            with patch("runpod_lifecycle.runner._upload_remote_script", new_callable=AsyncMock):
                result = await ship_and_run_detached(
                    config,
                    "echo ok",
                    local_root=local_root,
                    remote_root="/tmp/remote",
                    exclude=set(),
                    timeout=30,
                    terminate_after_exec=True,
                    poll_interval=1,
                )

    assert result.returncode == 124  # timeout


# ---------------------------------------------------------------------------
# ShipAndRunResult dataclass
# ---------------------------------------------------------------------------

def test_ship_and_run_result_defaults() -> None:
    """ShipAndRunResult has sensible defaults."""
    result = ShipAndRunResult(returncode=0)
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert result.pod is None
    assert result.artifact_root is None
    assert result.breach_log == []
    assert result.terminated is False
    assert result.upload_info == {}