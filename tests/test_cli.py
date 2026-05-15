from __future__ import annotations

import json
import sys

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
    assert "list" in out and "find-orphans" in out and "terminate" in out


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
    monkeypatch.setattr("runpod_lifecycle.cli.load_dotenv", lambda *a, **k: None)
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
    monkeypatch.setattr("runpod_lifecycle.cli.load_dotenv", lambda *a, **k: None)
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
    monkeypatch.setattr("runpod_lifecycle.cli.load_dotenv", lambda *a, **k: None)
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
    monkeypatch.setattr("runpod_lifecycle.cli.load_dotenv", lambda *a, **k: None)
    with pytest.raises(SystemExit) as exc:
        cli.main(["list"])
    assert exc.value.code == 2


def test_cli_list_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr("runpod_lifecycle.cli.load_dotenv", lambda *a, **k: None)

    async def fake_list(api_key, *, name_prefix=None):
        assert api_key == "k"
        return [_summary("a"), _summary("b")]

    monkeypatch.setattr("runpod_lifecycle.cli.discovery.list_pods", fake_list)
    rc = cli.main(["list", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert [p["id"] for p in payload] == ["a", "b"]


def test_cli_find_orphans_reads_known_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr("runpod_lifecycle.cli.load_dotenv", lambda *a, **k: None)
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
    monkeypatch.setattr("runpod_lifecycle.cli.load_dotenv", lambda *a, **k: None)

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
    monkeypatch.setattr("runpod_lifecycle.cli.load_dotenv", lambda *a, **k: None)
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
    monkeypatch.setattr("runpod_lifecycle.cli.load_dotenv", lambda *a, **k: None)

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
    monkeypatch.setattr("runpod_lifecycle.cli.load_dotenv", lambda *a, **k: None)

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


def test_launch_probe_only_terminates_claimed_pod(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr("runpod_lifecycle.cli.load_dotenv", lambda *a, **k: None)

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


def test_cli_terminate_yes_skips_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr("runpod_lifecycle.cli.load_dotenv", lambda *a, **k: None)

    called: list[str] = []

    async def fake_term(pod_id, api_key, *, hooks=None):
        called.append(pod_id)

    monkeypatch.setattr("runpod_lifecycle.cli.discovery.terminate", fake_term)
    rc = cli.main(["terminate", "abc", "--yes"])
    assert rc == 0
    assert called == ["abc"]
