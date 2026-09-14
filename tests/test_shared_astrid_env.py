from __future__ import annotations

import os
from pathlib import Path

import pytest

from runpod_lifecycle.config import RunPodConfig, astrid_env_file_path, load_runpod_env


def _clear_runpod_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "RUNPOD_API_KEY",
        "RUNPOD_GPU_TYPE",
        "RUNPOD_STORAGE_NAME",
        "RUNPOD_WORKER_IMAGE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_shared_path_matches_astrid_override_and_home(tmp_path: Path) -> None:
    assert astrid_env_file_path({"ASTRID_ENV_FILE": str(tmp_path / "custom.env")}) == tmp_path / "custom.env"
    assert astrid_env_file_path({"ASTRID_HOME": str(tmp_path / "home")}) == tmp_path / "home" / "astrid.env"
    assert astrid_env_file_path(
        {"ASTRID_HOME": str(tmp_path / "home"), "ASTRID_ENV_FILE": "shared/keys.env"}
    ) == tmp_path / "home" / "shared" / "keys.env"


def test_shared_credential_beats_stale_environment_and_project_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_runpod_settings(monkeypatch)
    shared_file = tmp_path / "astrid.env"
    shared_file.write_text("RUNPOD_API_KEY=canonical-key\n", encoding="utf-8")
    project_file = tmp_path / ".env"
    project_file.write_text(
        "RUNPOD_API_KEY=stale-project-key\nRUNPOD_GPU_TYPE=RTX 3090\nRUNPOD_STORAGE_NAME=keep-me\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ASTRID_ENV_FILE", str(shared_file))
    monkeypatch.setenv("RUNPOD_API_KEY", "stale-process-key")

    load_runpod_env(project_file)

    assert os.environ["RUNPOD_API_KEY"] == "canonical-key"
    assert os.environ["RUNPOD_GPU_TYPE"] == "RTX 3090"
    assert os.environ["RUNPOD_STORAGE_NAME"] == "keep-me"


def test_project_dotenv_cannot_supply_runpod_key_but_keeps_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_runpod_settings(monkeypatch)
    shared_file = tmp_path / "absent-astrid.env"
    project_file = tmp_path / ".env"
    project_file.write_text(
        "RUNPOD_API_KEY=stale-project-key\nRUNPOD_GPU_TYPE=RTX 3090\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ASTRID_ENV_FILE", str(shared_file))

    load_runpod_env(project_file)

    assert "RUNPOD_API_KEY" not in os.environ
    assert os.environ["RUNPOD_GPU_TYPE"] == "RTX 3090"


def test_process_injected_key_works_when_shared_file_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_runpod_settings(monkeypatch)
    monkeypatch.setenv("ASTRID_ENV_FILE", str(tmp_path / "absent-astrid.env"))
    monkeypatch.setenv("RUNPOD_API_KEY", "deployment-injected-key")
    project_file = tmp_path / ".env"
    project_file.write_text("RUNPOD_API_KEY=ignored-project-key\n", encoding="utf-8")

    load_runpod_env(project_file)

    assert os.environ["RUNPOD_API_KEY"] == "deployment-injected-key"


def test_empty_shared_key_falls_back_to_process_but_not_project_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_runpod_settings(monkeypatch)
    shared_file = tmp_path / "astrid.env"
    shared_file.write_text("RUNPOD_API_KEY=''\n", encoding="utf-8")
    project_file = tmp_path / ".env"
    project_file.write_text("RUNPOD_API_KEY=stale-project-key\n", encoding="utf-8")
    monkeypatch.setenv("ASTRID_ENV_FILE", str(shared_file))
    monkeypatch.setenv("RUNPOD_API_KEY", "deployment-injected-key")

    load_runpod_env(project_file)

    assert os.environ["RUNPOD_API_KEY"] == "deployment-injected-key"


def test_runpod_config_reads_key_from_shared_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_runpod_settings(monkeypatch)
    shared_file = tmp_path / "astrid.env"
    shared_file.write_text("RUNPOD_API_KEY=canonical-key\n", encoding="utf-8")
    monkeypatch.setenv("ASTRID_ENV_FILE", str(shared_file))
    monkeypatch.chdir(tmp_path)

    config = RunPodConfig.from_env()

    assert config.api_key == "canonical-key"
