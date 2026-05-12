from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

import sys

# Resolve the probe submodule directly; ``runpod_lifecycle.__init__`` rebinds
# the name ``probe`` to the function, so ``from runpod_lifecycle import probe``
# does *not* give us the module.
probe_module = sys.modules.get("runpod_lifecycle.probe")
if probe_module is None:  # pragma: no cover - first-time import.
    import importlib

    probe_module = importlib.import_module("runpod_lifecycle.probe")
probe = probe_module.probe


def _gpu(
    gpu_id: str,
    display: str,
    mem: int,
    price: float | None,
) -> dict[str, Any]:
    return {
        "id": gpu_id,
        "displayName": display,
        "memoryInGb": mem,
        "secureCloud": True,
        "communityCloud": False,
        "lowestPrice": {"uninterruptablePrice": price} if price is not None else None,
    }


def _patch_response(
    monkeypatch: pytest.MonkeyPatch, payload: list[dict[str, Any]]
) -> dict[str, Any]:
    """Stub httpx.post used by probe._fetch_gpu_types; record the sent query."""
    captured: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, Any], headers: dict[str, str], timeout: int):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"data": {"gpuTypes": payload}}
        return response

    monkeypatch.setattr(probe_module.httpx, "post", fake_post)
    return captured


@pytest.mark.asyncio
async def test_probe_filters_by_min_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_response(
        monkeypatch,
        [
            _gpu("NVIDIA RTX A4000", "RTX A4000", 16, 0.30),
            _gpu("NVIDIA RTX 6000 Ada Generation", "RTX 6000 Ada", 48, 0.77),
            _gpu("NVIDIA A100 80GB PCIe", "A100 80GB", 80, 1.89),
        ],
    )

    results = await probe(api_key="k", min_memory_gb=24)
    gpu_names = [r["gpu_type"] for r in results]
    assert "RTX A4000" not in gpu_names
    assert "RTX 6000 Ada" in gpu_names
    assert "A100 80GB" in gpu_names


@pytest.mark.asyncio
async def test_probe_excludes_blackwell_when_flagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_response(
        monkeypatch,
        [
            _gpu("NVIDIA RTX 6000 Ada Generation", "RTX 6000 Ada", 48, 0.77),
            _gpu("NVIDIA B200 Blackwell", "B200 Blackwell", 180, 4.50),
            _gpu("NVIDIA GeForce RTX 5090", "RTX 5090 (Blackwell)", 32, 0.95),
        ],
    )

    with_blackwell = await probe(api_key="k", min_memory_gb=24, exclude_blackwell=False)
    assert len(with_blackwell) == 3
    assert any(r["is_blackwell"] for r in with_blackwell)

    without_blackwell = await probe(
        api_key="k", min_memory_gb=24, exclude_blackwell=True
    )
    assert len(without_blackwell) == 1
    assert without_blackwell[0]["gpu_type"] == "RTX 6000 Ada"
    assert without_blackwell[0]["is_blackwell"] is False


@pytest.mark.asyncio
async def test_probe_ranks_by_price_ascending(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_response(
        monkeypatch,
        [
            _gpu("NVIDIA A100 80GB PCIe", "A100 80GB", 80, 1.89),
            _gpu("NVIDIA RTX 6000 Ada Generation", "RTX 6000 Ada", 48, 0.77),
            _gpu("NVIDIA H100 PCIe", "H100 PCIe", 80, 2.69),
        ],
    )

    results = await probe(api_key="k", min_memory_gb=24)
    prices = [r["price_per_hour"] for r in results]
    assert prices == sorted(prices)
    assert results[0]["gpu_type"] == "RTX 6000 Ada"


@pytest.mark.asyncio
async def test_probe_filters_by_max_price(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_response(
        monkeypatch,
        [
            _gpu("NVIDIA A100 80GB PCIe", "A100 80GB", 80, 1.89),
            _gpu("NVIDIA RTX 6000 Ada Generation", "RTX 6000 Ada", 48, 0.77),
            _gpu("NVIDIA H100 PCIe", "H100 PCIe", 80, 2.69),
        ],
    )

    results = await probe(api_key="k", min_memory_gb=24, max_price_per_hour=1.00)
    assert len(results) == 1
    assert results[0]["gpu_type"] == "RTX 6000 Ada"


@pytest.mark.asyncio
async def test_probe_drops_entries_with_no_lowest_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_response(
        monkeypatch,
        [
            _gpu("NVIDIA RTX 6000 Ada Generation", "RTX 6000 Ada", 48, 0.77),
            _gpu("NVIDIA H100 NVL", "H100 NVL", 94, None),  # no availability
        ],
    )
    results = await probe(api_key="k", min_memory_gb=24)
    assert [r["gpu_type"] for r in results] == ["RTX 6000 Ada"]


@pytest.mark.asyncio
async def test_probe_gpu_types_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_response(
        monkeypatch,
        [
            _gpu("NVIDIA RTX 6000 Ada Generation", "RTX 6000 Ada", 48, 0.77),
            _gpu("NVIDIA A100 80GB PCIe", "A100 80GB", 80, 1.89),
        ],
    )
    results = await probe(
        api_key="k",
        min_memory_gb=24,
        gpu_types=["NVIDIA A100 80GB PCIe"],
    )
    assert [r["gpu_type"] for r in results] == ["A100 80GB"]


@pytest.mark.asyncio
async def test_probe_query_uses_secure_cloud_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _patch_response(monkeypatch, [])
    await probe(api_key="k", require_secure_cloud=True)
    assert "secureCloud: true" in captured["json"]["query"]

    captured2 = _patch_response(monkeypatch, [])
    await probe(api_key="k", require_secure_cloud=False)
    assert "secureCloud: false" in captured2["json"]["query"]


@pytest.mark.asyncio
async def test_probe_includes_secure_cloud_and_datacenter_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_response(
        monkeypatch,
        [_gpu("NVIDIA RTX 6000 Ada Generation", "RTX 6000 Ada", 48, 0.77)],
    )
    results = await probe(api_key="k", min_memory_gb=24, require_secure_cloud=True)
    assert results[0]["secure_cloud"] is True
    assert results[0]["datacenters_available"] == []
    assert results[0]["memory_gb"] == 48


@pytest.mark.asyncio
async def test_probe_raises_on_graphql_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(*args: Any, **kwargs: Any):
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {"errors": [{"message": "bad"}]}
        return response

    monkeypatch.setattr(probe_module.httpx, "post", fake_post)

    with pytest.raises(RuntimeError, match="errors"):
        await probe(api_key="k")


@pytest.mark.asyncio
async def test_probe_raises_on_http_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(*args: Any, **kwargs: Any):
        response = MagicMock()
        response.status_code = 500
        response.text = "internal error"
        return response

    monkeypatch.setattr(probe_module.httpx, "post", fake_post)

    with pytest.raises(RuntimeError, match="HTTP 500"):
        await probe(api_key="k")
