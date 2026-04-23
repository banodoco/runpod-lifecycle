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
