from __future__ import annotations

import json
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from runpod_lifecycle import cli, discovery


def _summary(pod_id: str = "p1", cost: float = 0.5) -> discovery.PodSummary:
    return discovery.PodSummary(
        id=pod_id, name=f"name-{pod_id}", desired_status="RUNNING",
        actual_status="RUNNING", gpu_type="RTX 4090", image="img",
        created_at="2026-04-01", cost_per_hr=cost, uptime_seconds=100,
        ports=[], network_volume_id=None,
    )


def test_cli_help_runs(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "list" in out and "find-orphans" in out and "terminate" in out and "resume" in out


def test_resume_cli_accepts_capacity_wait_options() -> None:
    args = cli.build_parser().parse_args(
        [
            "resume",
            "pod-1",
            "--wait-capacity",
            "900",
            "--retry-interval",
            "60",
            "--wait-ready",
        ]
    )

    assert args.cmd == "resume"
    assert args.pod_id == "pod-1"
    assert args.wait_capacity == 900
    assert args.retry_interval == 60
    assert args.wait_ready is True


def test_prebuilt_help_lists_validation_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(["prebuilt", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for subcommand in ("check", "status", "cleanup", "reconcile"):
        assert subcommand in out


def test_prebuilt_dry_run_commands_do_not_require_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setattr("runpod_lifecycle.cli.load_runpod_env", lambda *a, **k: None)
    enriched = tmp_path / "enriched.json"
    enriched.write_text(json.dumps({"targets": []}), encoding="utf-8")

    commands = [
        ["prebuilt", "check", "--data-center", "EUR-NO-1", "--dry-run"],
        ["prebuilt", "status", "--dry-run"],
        ["prebuilt", "cleanup", "--dry-run"],
        [
            "prebuilt",
            "reconcile",
            "--data-center",
            "EUR-NO-1",
            "--dry-run",
            "--enriched-targets-json",
            str(enriched),
        ],
    ]
    for argv in commands:
        assert cli.main(argv) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["dry_run"] is True
        assert payload["no_credentials_required"] is True
        if argv[1] == "check":
            assert payload["min_memory_gb"] == 16


def test_prebuilt_reconcile_plain_targets_blocks_before_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setattr("runpod_lifecycle.cli.load_runpod_env", lambda *a, **k: None)
    targets = tmp_path / "targets.json"
    targets.write_text(json.dumps({"targets": [{"template_id": "image/z_image"}]}), encoding="utf-8")

    rc = cli.main([
        "prebuilt",
        "reconcile",
        "--data-center",
        "EUR-NO-1",
        "--targets-json",
        str(targets),
    ])

    assert rc == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["status"] == "blocked"
    assert "Plain --targets-json must be enriched first" in payload["message"]


def test_prebuilt_reconcile_dry_run_local_enrichment_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setattr("runpod_lifecycle.cli.load_runpod_env", lambda *a, **k: None)
    targets = tmp_path / "targets.json"
    targets.write_text(json.dumps({"targets": [{"template_id": "image/z_image"}]}), encoding="utf-8")

    rc = cli.main([
        "prebuilt",
        "reconcile",
        "--data-center",
        "US-TX-1",
        "--dry-run",
        "--targets-json",
        str(targets),
        "--local-vibecomfy-dir",
        "/opt/vibecomfy",
        "--models-root",
        "/workspace/reigh-livetest-prebuilt/models",
    ])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["requires_enrichment"] is True
    assert "vibecomfy workflows enrich-targets" in payload["remediation"]
    assert payload["local_enrichment_command"].startswith("cd /opt/vibecomfy")


def test_prebuilt_cleanup_rejects_unallowlisted_prefix() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args([
            "prebuilt",
            "cleanup",
            "--prefix",
            "user-owned-pod-",
            "--dry-run",
        ])
    assert exc.value.code == 2


def test_cli_missing_api_key_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    monkeypatch.setattr("runpod_lifecycle.cli.load_runpod_env", lambda *a, **k: None)
    with pytest.raises(SystemExit) as exc:
        cli.main(["list"])
    assert exc.value.code == 2


def test_cli_list_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr("runpod_lifecycle.cli.load_runpod_env", lambda *a, **k: None)

    async def fake_list(api_key, *, name_prefix=None):
        assert api_key == "k"
        return [_summary("a"), _summary("b")]

    monkeypatch.setattr("runpod_lifecycle.cli.discovery.list_pods", fake_list)
    rc = cli.main(["list", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert [p["id"] for p in payload] == ["a", "b"]


def test_cli_status_redacts_credentials_without_changing_status_fields(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "operator-api-key")
    monkeypatch.setattr("runpod_lifecycle.cli.load_runpod_env", lambda *a, **k: None)
    status = {
        "runpod_id": "pod-1",
        "desired_status": "RUNNING",
        "ssh_password": "ssh-password-sentinel",
        "runtime": {
            "apiKey": "api-key-sentinel",
            "RUNPOD_API_KEY": "runpod-api-key-sentinel",
            "HF_TOKEN": "hf-token-sentinel",
            "PASSWORD": "password-sentinel",
            "runpod_api_key": "composite-api-key-sentinel",
            "ssh_private_key": "private-key-sentinel",
            "private_key": "private-key-sentinel",
            "token": "token-sentinel",
            "ports": [{"privatePort": 22, "publicPort": 2201}],
        },
        "ip": "203.0.113.10",
    }
    original_status = deepcopy(status)
    monkeypatch.setattr("runpod_lifecycle.cli.api.get_pod_status", lambda *_: status)

    assert cli.main(["status", "pod-1"]) == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert "sentinel" not in captured.out
    assert payload["desired_status"] == "RUNNING"
    assert payload["ip"] == "203.0.113.10"
    assert payload["ssh_password"] == "[REDACTED]"
    assert payload["runtime"] == {
        "apiKey": "[REDACTED]",
        "RUNPOD_API_KEY": "[REDACTED]",
        "HF_TOKEN": "[REDACTED]",
        "PASSWORD": "[REDACTED]",
        "runpod_api_key": "[REDACTED]",
        "ssh_private_key": "[REDACTED]",
        "private_key": "[REDACTED]",
        "token": "[REDACTED]",
        "ports": [{"privatePort": 22, "publicPort": 2201}],
    }
    assert status == original_status


def test_cli_resume_redacts_credentials_from_operator_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "operator-api-key")
    monkeypatch.setattr("runpod_lifecycle.cli.load_runpod_env", lambda *a, **k: None)
    status = {
        "desired_status": "RUNNING",
        "ssh_password": "resume-password-sentinel",
        "access_token": "resume-token-sentinel",
        "nested": [
            {
                "RUNPOD_API_KEY": "resume-runpod-api-key-sentinel",
                "HF_TOKEN": "resume-hf-token-sentinel",
                "PASSWORD": "resume-password-sentinel",
                "runpod_api_key": "resume-composite-api-key-sentinel",
                "ssh_private_key": "resume-private-key-sentinel",
                "privatePort": 22,
                "publicPort": 2201,
            }
        ],
    }
    original_status = deepcopy(status)

    class FakePod:
        def __init__(self, pod_id, name, config):
            assert pod_id == name == "pod-1"
            assert config.api_key == "operator-api-key"

        async def status(self):
            return status

        async def wait_ready(self, timeout: int):
            assert timeout == 17
            return status

    async def fake_resume(pod, **kwargs):
        assert isinstance(pod, FakePod)
        assert kwargs == {"max_wait_sec": 900, "retry_interval_sec": 30}
        return pod

    monkeypatch.setattr("runpod_lifecycle.cli.Pod", FakePod)
    monkeypatch.setattr("runpod_lifecycle.cli._resume_when_available", fake_resume)

    assert cli.main(["resume", "pod-1", "--wait-ready", "--timeout", "17"]) == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert "sentinel" not in captured.out
    assert payload["pod_id"] == "pod-1"
    assert payload["status"]["desired_status"] == "RUNNING"
    assert payload["status"]["ssh_password"] == "[REDACTED]"
    assert payload["status"]["access_token"] == "[REDACTED]"
    assert payload["status"]["nested"] == [
        {
            "RUNPOD_API_KEY": "[REDACTED]",
            "HF_TOKEN": "[REDACTED]",
            "PASSWORD": "[REDACTED]",
            "runpod_api_key": "[REDACTED]",
            "ssh_private_key": "[REDACTED]",
            "privatePort": 22,
            "publicPort": 2201,
        }
    ]
    assert status == original_status


def test_cli_find_orphans_reads_known_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr("runpod_lifecycle.cli.load_runpod_env", lambda *a, **k: None)
    ids = tmp_path / "known.txt"
    ids.write_text("a\nb\n\n")

    seen: dict = {}

    async def fake_find(api_key, known, *, name_prefix=None, older_than_seconds=None):
        seen["known"] = list(known)
        seen["older"] = older_than_seconds
        return [_summary("c")]

    monkeypatch.setattr("runpod_lifecycle.cli.discovery.find_orphans", fake_find)
    rc = cli.main(["find-orphans", "--known-ids-file", str(ids), "--older-than", "1h"])
    assert rc == 0
    assert seen["known"] == ["a", "b"]
    assert seen["older"] == 3600
    out = capsys.readouterr().out
    assert "c" in out and "Total" in out


def test_cli_find_orphans_terminate_yes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr("runpod_lifecycle.cli.load_runpod_env", lambda *a, **k: None)

    async def fake_find(api_key, known, *, name_prefix=None, older_than_seconds=None):
        return [_summary("orph1"), _summary("orph2")]

    terminated: list[str] = []

    async def fake_term(pod_id, api_key, *, hooks=None):
        terminated.append(pod_id)

    monkeypatch.setattr("runpod_lifecycle.cli.discovery.find_orphans", fake_find)
    monkeypatch.setattr("runpod_lifecycle.cli.discovery.terminate", fake_term)
    rc = cli.main(["find-orphans", "--terminate", "--yes"])
    assert rc == 0
    assert terminated == ["orph1", "orph2"]


def test_cli_terminate_requires_confirmation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr("runpod_lifecycle.cli.load_runpod_env", lambda *a, **k: None)
    monkeypatch.setattr("builtins.input", lambda *_a, **_k: "n")

    called: list[str] = []

    async def fake_term(pod_id, api_key, *, hooks=None):
        called.append(pod_id)

    monkeypatch.setattr("runpod_lifecycle.cli.discovery.terminate", fake_term)
    rc = cli.main(["terminate", "abc"])
    assert rc == 1
    assert called == []


def test_cli_probe_prints_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr("runpod_lifecycle.cli.load_runpod_env", lambda *a, **k: None)

    captured: dict = {}

    async def fake_probe(*, api_key, **kwargs):
        captured["api_key"] = api_key
        captured["kwargs"] = kwargs
        return [
            {
                "gpu_type": "RTX 6000 Ada",
                "memory_gb": 48,
                "price_per_hour": 0.77,
                "secure_cloud": True,
                "is_blackwell": False,
                "datacenters_available": [],
            }
        ]

    monkeypatch.setattr("runpod_lifecycle.cli._probe", fake_probe)
    rc = cli.main([
        "probe",
        "--min-memory",
        "48",
        "--exclude-blackwell",
        "--max-price",
        "1.5",
    ])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["gpu_type"] == "RTX 6000 Ada"
    assert captured["api_key"] == "k"
    assert captured["kwargs"]["min_memory_gb"] == 48
    assert captured["kwargs"]["exclude_blackwell"] is True
    assert captured["kwargs"]["max_price_per_hour"] == 1.5
    # Secure-cloud is the default; --allow-community-cloud was not passed.
    assert captured["kwargs"]["require_secure_cloud"] is True


def test_cli_probe_table_format(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr("runpod_lifecycle.cli.load_runpod_env", lambda *a, **k: None)

    async def fake_probe(**_kwargs):
        return [
            {
                "gpu_type": "RTX 6000 Ada",
                "memory_gb": 48,
                "price_per_hour": 0.77,
                "secure_cloud": True,
                "is_blackwell": False,
                "datacenters_available": [],
            }
        ]

    monkeypatch.setattr("runpod_lifecycle.cli._probe", fake_probe)
    rc = cli.main(["probe", "--format", "table"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "GPU TYPE" in out
    assert "RTX 6000 Ada" in out
    assert "$0.770" in out


def test_resolve_config_accepts_storage_volume_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setenv("RUNPOD_STORAGE_NAME", "primary")
    monkeypatch.setenv("RUNPOD_STORAGE_VOLUMES", "env-a, env-b")

    args = cli.build_parser().parse_args([
        "launch",
        "--storage-volumes",
        "cli-a, cli-b",
        "--gpu-type",
        "NVIDIA L40S",
        "--min-memory-gb",
        "16",
        "--ram-tiers",
        "32, 24, 16",
    ])

    config = cli._resolve_config(args)
    assert config.storage_name == "primary"
    assert config.storage_volumes == ("cli-a", "cli-b")
    assert config.gpu_type == "NVIDIA L40S"
    assert config.min_memory_gb == 16
    assert config.ram_tiers == (32, 24, 16)


def test_resolve_config_keeps_complete_environment_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI resolver must not discard fields needed by launch/SSH."""
    monkeypatch.setattr("runpod_lifecycle.cli.load_runpod_env", lambda *a, **k: None)
    monkeypatch.setattr("runpod_lifecycle.config.load_dotenv", lambda *a, **k: None)
    values = {
        "RUNPOD_API_KEY": "api-secret",
        "RUNPOD_GPU_TYPE": "NVIDIA GeForce RTX 5090",
        "RUNPOD_WORKER_IMAGE": "h3-worker:latest",
        "RUNPOD_TEMPLATE_ID": "h3-template",
        "RUNPOD_VOLUME_MOUNT_PATH": "/mnt/peter",
        "RUNPOD_CONTAINER_DISK_GB": "321",
        "RUNPOD_DISK_SIZE_GB": "654",
        "RUNPOD_MIN_VCPU_COUNT": "14",
        "RUNPOD_MIN_MEMORY_GB": "40",
        "RUNPOD_STORAGE_NAME": "Peter",
        "RUNPOD_SSH_PUBLIC_KEY": "ssh-public-secret",
        "RUNPOD_SSH_PRIVATE_KEY": "ssh-private-secret",
        "RUNPOD_SSH_PUBLIC_KEY_PATH": "/keys/id.pub",
        "RUNPOD_SSH_PRIVATE_KEY_PATH": "/keys/id",
        "RUNPOD_ENV_VARS": '{"HF_TOKEN":"token-secret","MODE":"h3"}',
        "RUNPOD_PORTS": "8675/http,22/tcp",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    args = cli.build_parser().parse_args(["launch"])
    config = cli._resolve_config(args)

    assert config.api_key == "api-secret"
    assert config.gpu_type == "NVIDIA GeForce RTX 5090"
    assert config.worker_image == "h3-worker:latest"
    assert config.template_id == "h3-template"
    assert config.volume_mount_path == "/mnt/peter"
    assert config.container_disk_gb == 321
    assert config.disk_size_gb == 654
    assert config.min_vcpu_count == 14
    assert config.min_memory_gb == 40
    assert config.storage_name == "Peter"
    assert config.ssh_public_key is None
    assert config.ssh_private_key is None
    assert config.ssh_public_key_path == "/keys/id.pub"
    assert config.ssh_private_key_path == "/keys/id"
    assert config.env_vars == {"HF_TOKEN": "token-secret", "MODE": "h3"}
    assert config.ports == "8675/http,22/tcp"


def test_launch_probe_only_does_not_print_config_secrets(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Probe output is safe even when the resolved config has credentials."""
    monkeypatch.setattr("runpod_lifecycle.cli.load_runpod_env", lambda *a, **k: None)
    monkeypatch.setattr("runpod_lifecycle.config.load_dotenv", lambda *a, **k: None)
    secrets = {
        "RUNPOD_API_KEY": "api-secret-not-printable",
        "RUNPOD_SSH_PUBLIC_KEY": "public-secret-not-printable",
        "RUNPOD_SSH_PRIVATE_KEY": "private-secret-not-printable",
        "RUNPOD_ENV_VARS": '{"HF_TOKEN":"env-secret-not-printable"}',
    }
    for key, value in secrets.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("RUNPOD_SSH_PRIVATE_KEY_PATH", raising=False)
    monkeypatch.delenv("RUNPOD_SSH_PUBLIC_KEY_PATH", raising=False)

    class FakePod:
        id = "pod-probe-safe"
        name = "probe-safe"
        _gpu_type = "NVIDIA GeForce RTX 5090"
        _ram_tier = 32
        _storage_name = "Peter"
        _storage_volume = "vol-peter"

        async def terminate(self) -> None:
            return None

    async def fake_launch(config, *, name=None):
        assert config.api_key == secrets["RUNPOD_API_KEY"]
        assert config.ssh_private_key == secrets["RUNPOD_SSH_PRIVATE_KEY"]
        assert config.env_vars["HF_TOKEN"] == "env-secret-not-printable"
        return FakePod()

    monkeypatch.setattr("runpod_lifecycle.cli._launch", fake_launch)
    assert cli.main(["launch", "--probe-only"]) == 0
    output = capsys.readouterr().out
    for secret in secrets.values():
        assert secret not in output


def test_launch_probe_only_terminates_claimed_pod(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr("runpod_lifecycle.cli.load_runpod_env", lambda *a, **k: None)

    class FakePod:
        id = "pod-probe"
        name = "probe-name"
        _gpu_type = "NVIDIA L4"
        _ram_tier = 32
        _storage_name = "portable"
        _storage_volume = "vol-123"

        def __init__(self) -> None:
            self.terminated = False

        async def terminate(self) -> None:
            self.terminated = True

    launched: dict[str, object] = {}

    async def fake_launch(config, *, name=None):
        launched["config"] = config
        launched["name"] = name
        return FakePod()

    monkeypatch.setattr("runpod_lifecycle.cli._launch", fake_launch)

    rc = cli.main([
        "launch",
        "--probe-only",
        "--name",
        "claim-test",
        "--gpu-type",
        "NVIDIA L4,NVIDIA RTX A5000",
        "--storage-name",
        "primary",
        "--storage-volumes",
        "portable, fallback",
    ])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["pod_id"] == "pod-probe"
    assert payload["terminated"] is True
    assert payload["selected_gpu_type"] == "NVIDIA L4"
    assert payload["selected_storage_name"] == "portable"
    assert payload["gpu_type_candidates"] == ["NVIDIA L4", "NVIDIA RTX A5000"]
    assert payload["storage_candidates"] == ["primary", "portable", "fallback"]
    assert launched["name"] == "claim-test"


def test_run_reattaches_to_supplied_pod_and_prints_artifact_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI run must reuse POD_ID and expose detached artifact/teardown state."""
    monkeypatch.setenv("RUNPOD_API_KEY", "api-key")
    monkeypatch.setattr("runpod_lifecycle.cli.load_runpod_env", lambda *a, **k: None)
    monkeypatch.setattr("runpod_lifecycle.config.load_dotenv", lambda *a, **k: None)
    script = tmp_path / "job.sh"
    script.write_text("echo h3\n", encoding="utf-8")
    supplied_pod = object()
    seen: dict[str, object] = {}

    async def fake_get_pod(pod_id, config):
        seen["pod_id"] = pod_id
        return supplied_pod

    async def fake_run(config, remote_script, **kwargs):
        seen["pod"] = kwargs["pod"]
        seen["remote_script"] = remote_script
        seen["terminate_after_exec"] = kwargs["terminate_after_exec"]
        seen["local_root"] = kwargs["local_root"]
        return SimpleNamespace(
            returncode=0,
            artifact_root=tmp_path / "artifacts",
            terminated=True,
        )

    monkeypatch.setattr("runpod_lifecycle.cli.discovery.get_pod", fake_get_pod)
    monkeypatch.setattr("runpod_lifecycle.runner.ship_and_run_detached", fake_run)

    rc = cli.main(["run", "pod-existing", "--script", str(script)])

    assert rc == 0
    assert seen["pod_id"] == "pod-existing"
    assert seen["pod"] is supplied_pod
    assert seen["remote_script"] == "echo h3\n"
    assert seen["terminate_after_exec"] is True
    assert seen["local_root"] == tmp_path
    assert json.loads(capsys.readouterr().out) == {
        "pod_id": "pod-existing",
        "returncode": 0,
        "artifact_root": str(tmp_path / "artifacts"),
        "terminated": True,
    }


def test_run_keep_pod_passes_non_terminating_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "api-key")
    monkeypatch.setattr("runpod_lifecycle.cli.load_runpod_env", lambda *a, **k: None)
    monkeypatch.setattr("runpod_lifecycle.config.load_dotenv", lambda *a, **k: None)
    script = tmp_path / "job.sh"
    script.write_text("true\n", encoding="utf-8")
    seen: dict[str, object] = {}

    async def fake_get_pod(pod_id, config):
        return object()

    async def fake_run(config, remote_script, **kwargs):
        seen["terminate_after_exec"] = kwargs["terminate_after_exec"]
        return SimpleNamespace(returncode=3, artifact_root=None, terminated=False)

    monkeypatch.setattr("runpod_lifecycle.cli.discovery.get_pod", fake_get_pod)
    monkeypatch.setattr("runpod_lifecycle.runner.ship_and_run_detached", fake_run)

    assert cli.main(["run", "pod-kept", "--script", str(script), "--keep-pod"]) == 3
    assert seen["terminate_after_exec"] is False
    assert json.loads(capsys.readouterr().out)["terminated"] is False


def test_run_surfaces_remote_output_before_human_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "api-key")
    monkeypatch.setattr("runpod_lifecycle.cli.load_runpod_env", lambda *a, **k: None)
    monkeypatch.setattr("runpod_lifecycle.config.load_dotenv", lambda *a, **k: None)
    script = tmp_path / "job.sh"
    script.write_text("echo output\n", encoding="utf-8")

    async def fake_get_pod(pod_id, config):
        return object()

    async def fake_run(config, remote_script, **kwargs):
        return SimpleNamespace(
            returncode=7,
            stdout="remote stdout\n",
            stderr="remote stderr\n",
            artifact_root=None,
            terminated=True,
        )

    monkeypatch.setattr("runpod_lifecycle.cli.discovery.get_pod", fake_get_pod)
    monkeypatch.setattr("runpod_lifecycle.runner.ship_and_run_detached", fake_run)

    assert cli.main(["run", "pod-output", "--script", str(script)]) == 7
    captured = capsys.readouterr()
    assert captured.out.startswith("remote stdout\n{")
    summary = json.loads(captured.out[captured.out.index("{"):])
    assert summary["returncode"] == 7
    assert captured.err == "remote stderr\n"


def test_run_json_keeps_remote_output_machine_readable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "api-key")
    monkeypatch.setattr("runpod_lifecycle.cli.load_runpod_env", lambda *a, **k: None)
    monkeypatch.setattr("runpod_lifecycle.config.load_dotenv", lambda *a, **k: None)
    script = tmp_path / "job.sh"
    script.write_text("echo output\n", encoding="utf-8")

    async def fake_get_pod(pod_id, config):
        return object()

    async def fake_run(config, remote_script, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="remote stdout\n",
            stderr="remote stderr\n",
            artifact_root=tmp_path / "artifacts",
            terminated=False,
        )

    monkeypatch.setattr("runpod_lifecycle.cli.discovery.get_pod", fake_get_pod)
    monkeypatch.setattr("runpod_lifecycle.runner.ship_and_run_detached", fake_run)

    assert cli.main(["run", "pod-json", "--script", str(script), "--json"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["stdout"] == "remote stdout\n"
    assert payload["stderr"] == "remote stderr\n"
    assert captured.err == ""


def test_prebuilt_build_accepts_storage_volume_candidates() -> None:
    args = cli.build_parser().parse_args([
        "prebuilt",
        "build",
        "--data-center",
        "EU-RO-1",
        "--storage-volumes",
        "Peter, EUR-IS-1",
    ])

    assert cli._resolve_storage_volumes(args) == ("Peter", "EUR-IS-1")


def test_cli_terminate_yes_skips_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr("runpod_lifecycle.cli.load_runpod_env", lambda *a, **k: None)

    called: list[str] = []

    async def fake_term(pod_id, api_key, *, hooks=None):
        called.append(pod_id)

    monkeypatch.setattr("runpod_lifecycle.cli.discovery.terminate", fake_term)
    rc = cli.main(["terminate", "abc", "--yes"])
    assert rc == 0
    assert called == ["abc"]
