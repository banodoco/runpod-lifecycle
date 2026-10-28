"""Original startup bearer projection, independent of HTTP transport policy."""
import os
from pathlib import Path
import stat

import pytest

from runpod_lifecycle.https import read_token_file
from runpod_lifecycle.pod_agent import AgentServer, OBSERVATION_SCHEMA, _write_engine_bearer_file

TOKEN = "synthetic-original-startup-bearer-" + "x" * 32


@pytest.mark.parametrize("observation", [False, True])
@pytest.mark.parametrize("constructor", [False, True])
def test_original_bearer_exists_before_engine_enable_and_survives_attachment(tmp_path, observation, constructor):
    state = tmp_path / "state"
    options = {"observation_policy": {"schema": OBSERVATION_SCHEMA, "roots": [],
                                      "files": [str(tmp_path / "public.json")]}} if observation else {}
    if constructor:
        options["engine_proxy"] = {"endpoint_id": "vibecomfy-h3", "port": 8188}
    server = AgentServer(("127.0.0.1", 0), TOKEN, state, **options)
    try:
        if not constructor:
            assert server.engine_proxy is None
            server.enable_engine("vibecomfy-h3", 8188)
        path = state / "engine-bearer.token"
        assert read_token_file(path) == TOKEN
        before = path.stat()
        assert stat.S_IMODE(before.st_mode) == 0o600
        assert stat.S_IMODE(state.stat().st_mode) == 0o700
        server.enable_engine("vibecomfy-h3", 8188)
        assert path.stat().st_ino == before.st_ino
        if observation:
            assert server.observation_policy is not None and not server.jobs
    finally:
        server.server_close()
    assert path.exists()  # Closing a borrowed attachment never removes startup state.


@pytest.mark.parametrize("unsafe", ["different", "mode", "hardlink", "leaf-symlink", "parent-symlink", "fifo", "directory-mode"])
def test_unsafe_or_different_restart_refuses_without_replacing_original(tmp_path, unsafe):
    state = tmp_path / "private"
    original = _write_engine_bearer_file(state, TOKEN)
    chosen = state
    if unsafe == "different": original.write_text("y" * len(TOKEN))
    if unsafe == "mode": original.chmod(0o644)
    if unsafe == "hardlink": os.link(original, tmp_path / "alias")
    if unsafe == "leaf-symlink":
        original.unlink()
        other = tmp_path / "other"
        other.write_text(TOKEN)
        original.symlink_to(other)
    if unsafe == "parent-symlink":
        chosen = tmp_path / "linked"
        chosen.symlink_to(state, target_is_directory=True)
    if unsafe == "fifo":
        original.unlink()
        os.mkfifo(original, 0o600)
    if unsafe == "directory-mode": state.chmod(0o755)
    with pytest.raises(ValueError):
        _write_engine_bearer_file(chosen, TOKEN)


def test_http_bearer_actual_open_rejects_symlink_ancestor(tmp_path):
    path = _write_engine_bearer_file(tmp_path / "private", TOKEN)
    alias = tmp_path / "alias"
    alias.symlink_to(path.parent, target_is_directory=True)
    with pytest.raises((OSError, ValueError)):
        read_token_file(alias / path.name)
