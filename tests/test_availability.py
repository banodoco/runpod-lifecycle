"""Offline discovery tests using sanitized live GraphQL stock/price responses."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import httpx
import pytest

from runpod_lifecycle import cli, gpu_availability, probe

availability = importlib.import_module("runpod_lifecycle.availability")
GPU_ID = "NVIDIA GeForce RTX 5090"


@pytest.fixture
def recorded():
    return json.loads(
        (Path(__file__).parent / "fixtures/rtx5090-stock.json").read_text()
    )


def mock_provider(monkeypatch, recorded):
    calls = []
    centers = recorded["datacenters"]["data"]["dataCenters"]
    original_prices = recorded["prices"]["data"]["gpuTypes"][0]
    by_dc = {dc["id"]: original_prices[f"dc{i}"] for i, dc in enumerate(centers)}

    def post(url, *, json, headers, timeout):
        assert url == availability.GRAPHQL_URL
        assert json["query"].lstrip().startswith("query ")
        assert "mutation" not in json["query"]
        calls.append(copy.deepcopy(json))
        if "GpuCatalogue" in json["query"]:
            body = recorded["catalogue"]
        elif "GpuTypesProbe" in json["query"]:
            gpu = {
                **recorded["catalogue"]["data"]["gpuTypes"][0],
                "secureCloud": True,
                "communityCloud": False,
                "lowestPrice": {"uninterruptablePrice": 1.19},
            }
            body = {"data": {"gpuTypes": [gpu]}}
        elif "DataCenterStock" in json["query"]:
            body = recorded["datacenters"]
        else:
            assert "DataCenterPrices" in json["query"]
            prices = {"id": GPU_ID}
            for key, constraints in json["variables"].items():
                if key.startswith("p"):
                    prices[f"dc{key[1:]}"] = by_dc[constraints["dataCenterId"]]
            body = {"data": {"gpuTypes": [prices]}}
        return httpx.Response(200, json=body)

    monkeypatch.setattr(availability.httpx, "post", post)
    return calls


async def test_live_shape_resolves_name_and_retains_false_and_null(
    monkeypatch, recorded
):
    calls = mock_provider(monkeypatch, recorded)
    result = await gpu_availability(api_key="sentinel", gpu="rtx 5090")
    assert result.gpu_type_id == GPU_ID
    assert result.status == "available"
    centers = {dc.datacenter_id: dc for dc in result.datacenters}
    assert centers["EU-CZ-1"].status == "unavailable"
    assert centers["EU-CZ-1"].stock_level is None
    assert centers["EU-CZ-1"].price_per_hour is None
    assert centers["EUR-NO-1"].status == "available"
    assert centers["EUR-NO-1"].stock_level == "Low"
    assert centers["EUR-NO-1"].price_per_hour == 1.19
    assert centers["EUR-NO-1"].network_volume_support is True
    assert all(
        dc.driver_version is None and dc.cuda_version is None
        for dc in result.datacenters
    )
    assert len(calls) == 3
    assert "sentinel" not in json.dumps(result.to_dict())
    assert result.observed_at.endswith("+00:00")


async def test_cuda13_unavailable_is_not_unknown(monkeypatch):
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/rtx5090-cuda13.json").read_text()
    )
    calls = mock_provider(monkeypatch, fixture)
    result = await gpu_availability(
        api_key="k", gpu=GPU_ID, allowed_cuda_versions=["13.0"]
    )
    assert result.status == "unavailable"
    assert len(result.datacenters) == 5
    assert all(dc.status == "unavailable" for dc in result.datacenters)
    assert calls[1]["variables"]["availability"]["allowedCudaVersions"] == ["13.0"]
    assert all(
        value["allowedCudaVersions"] == ["13.0"]
        for key, value in calls[2]["variables"].items()
        if key.startswith("p")
    )


async def test_filter_applies_to_dc_rows_and_price_query(monkeypatch, recorded):
    calls = mock_provider(monkeypatch, recorded)
    result = await gpu_availability(
        api_key="k", gpu=GPU_ID, datacenter_ids=["EUR-NO-1"]
    )
    assert [dc.datacenter_id for dc in result.datacenters] == ["EUR-NO-1"]
    assert calls[-1]["variables"]["p0"]["dataCenterId"] == "EUR-NO-1"
    assert "p1" not in calls[-1]["variables"]
    assert (
        await gpu_availability(api_key="k", gpu=GPU_ID, datacenter_ids=[])
    ).status == "unavailable"
    result = await gpu_availability(
        api_key="k", gpu=GPU_ID, datacenter_ids=["invented"]
    )
    assert result.status == "unknown"
    assert result.datacenters[0].name is None


@pytest.mark.parametrize(
    "support,expected", [(False, "unavailable"), (None, "unknown")]
)
async def test_network_volume_requirement_is_not_guessed(
    monkeypatch, recorded, support, expected
):
    recorded["datacenters"]["data"]["dataCenters"][-1]["storageSupport"] = support
    mock_provider(monkeypatch, recorded)
    result = await gpu_availability(
        api_key="k", gpu=GPU_ID, datacenter_ids=["EUR-NO-1"]
    )
    assert result.status == expected
    assert result.datacenters[0].network_volume_support is support
    result = await gpu_availability(
        api_key="k",
        gpu=GPU_ID,
        datacenter_ids=["EUR-NO-1"],
        require_network_volume=False,
    )
    assert result.status == "available"


@pytest.mark.parametrize("stage", ["catalogue", "datacenters", "prices"])
@pytest.mark.parametrize(
    "body", [{}, {"errors": [{"message": "sentinel"}]}, {"data": None}]
)
async def test_partial_or_error_response_is_typed_unknown(
    monkeypatch, recorded, stage, body
):
    mock_provider(monkeypatch, recorded)
    # Price fixture injection is handled at the transport boundary.
    original = availability.httpx.post

    def post(url, **kwargs):
        if {
            "prices": "DataCenterPrices",
            "catalogue": "GpuCatalogue",
            "datacenters": "DataCenterStock",
        }[stage] in kwargs["json"]["query"]:
            return httpx.Response(200, json=body)
        return original(url, **kwargs)

    monkeypatch.setattr(availability.httpx, "post", post)
    result = await gpu_availability(
        api_key="sentinel", gpu=GPU_ID, datacenter_ids=["EUR-NO-1"]
    )
    assert result.status == "unknown"
    assert "sentinel" not in json.dumps(result.to_dict())


@pytest.mark.parametrize("failure", ["http", "timeout", "json"])
async def test_transport_errors_never_leak_response_or_credentials(
    monkeypatch, failure
):
    def post(*args, **kwargs):
        if failure == "timeout":
            raise httpx.ReadTimeout("sentinel")
        return httpx.Response(403 if failure == "http" else 200, text="sentinel")

    monkeypatch.setattr(availability.httpx, "post", post)
    result = await gpu_availability(api_key="sentinel", gpu="RTX 5090")
    assert result.status == "unknown"
    assert "sentinel" not in json.dumps(result.to_dict())


async def test_gpu_not_in_catalogue_and_ambiguous_name(monkeypatch, recorded):
    mock_provider(monkeypatch, recorded)
    result = await gpu_availability(api_key="k", gpu="not a gpu")
    assert result.status == "unavailable" and result.gpu_type_id is None
    recorded["catalogue"]["data"]["gpuTypes"].append(
        {"id": "different", "displayName": "RTX 5090"}
    )
    result = await gpu_availability(api_key="k", gpu="RTX 5090")
    assert result.status == "unknown" and result.gpu_type_id is None


async def test_missing_stock_does_not_mean_unavailable(monkeypatch, recorded):
    recorded["datacenters"]["data"]["dataCenters"][-1]["gpuAvailability"] = None
    mock_provider(monkeypatch, recorded)
    result = await gpu_availability(
        api_key="k", gpu=GPU_ID, datacenter_ids=["EUR-NO-1"]
    )
    assert result.status == "unknown"
    recorded["datacenters"]["data"]["dataCenters"][-1]["gpuAvailability"] = []
    result = await gpu_availability(
        api_key="k", gpu=GPU_ID, datacenter_ids=["EUR-NO-1"]
    )
    assert result.status == "unavailable"


@pytest.mark.parametrize("price", [None, -1, float("nan"), "1.19", True])
async def test_available_flag_without_valid_dc_price_is_unknown(
    monkeypatch, recorded, price
):
    recorded["prices"]["data"]["gpuTypes"][0]["dc4"]["uninterruptablePrice"] = price
    mock_provider(monkeypatch, recorded)
    result = await gpu_availability(
        api_key="k", gpu=GPU_ID, datacenter_ids=["EUR-NO-1"]
    )
    assert result.status == "unknown"
    assert result.datacenters[0].price_per_hour is None


async def test_disagreeing_snapshots_are_unknown(monkeypatch, recorded):
    recorded["prices"]["data"]["gpuTypes"][0]["dc4"]["stockStatus"] = "High"
    mock_provider(monkeypatch, recorded)
    result = await gpu_availability(
        api_key="k", gpu=GPU_ID, datacenter_ids=["EUR-NO-1"]
    )
    assert result.status == "unknown"


async def test_probe_filter_and_dc_specific_price_cap(monkeypatch, recorded):
    mock_provider(monkeypatch, recorded)
    result = await probe(api_key="k", gpu_types=[GPU_ID], datacenter_ids=["EUR-NO-1"])
    assert result[0]["datacenters_available"] == ["EUR-NO-1"]
    assert result[0]["datacenter_availability"][0]["price_per_hour"] == 1.19
    assert (
        await probe(api_key="k", gpu_types=[GPU_ID], datacenter_ids=["EU-CZ-1"]) == []
    )
    recorded["prices"]["data"]["gpuTypes"][0]["dc4"]["uninterruptablePrice"] = 2.0
    mock_provider(monkeypatch, recorded)
    assert (
        await probe(
            api_key="k",
            gpu_types=[GPU_ID],
            datacenter_ids=["EUR-NO-1"],
            max_price_per_hour=1.5,
        )
        == []
    )


@pytest.mark.parametrize(
    "status,exit_code", [("available", 0), ("unavailable", 2), ("unknown", 3)]
)
def test_cli_json_and_exit_status(monkeypatch, capsys, status, exit_code):
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.setattr(cli, "load_runpod_env", lambda: None)

    async def fake(**kwargs):
        assert kwargs["gpu"] == "RTX 5090"
        assert kwargs["datacenter_ids"] == ["EUR-NO-1"]
        assert kwargs["allowed_cuda_versions"] == ("13.0",)
        return availability.GpuAvailability(
            "RTX 5090", GPU_ID, "RTX 5090", status, "now", {}
        )

    monkeypatch.setattr(cli, "_gpu_availability", fake)
    assert (
        cli.main(
            [
                "availability",
                "--gpu",
                "RTX 5090",
                "--datacenter-ids",
                "EUR-NO-1",
                "--allowed-cuda-versions",
                "13.0",
            ]
        )
        == exit_code
    )
    assert json.loads(capsys.readouterr().out)["status"] == status


async def test_minimum_cuda_uses_provider_filter_on_stock_and_prices(monkeypatch):
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/rtx5090-min-cuda13.json").read_text()
    )
    calls = mock_provider(monkeypatch, fixture)
    result = await gpu_availability(
        api_key="k", gpu="RTX 5090", min_cuda_version="13.0"
    )
    assert result.status == "available"
    assert calls[1]["variables"]["availability"]["minCudaVersion"] == "13.0"
    assert all(
        value["minCudaVersion"] == "13.0"
        for key, value in calls[2]["variables"].items()
        if key.startswith("p")
    )
    assert "allowedCudaVersions" not in result.constraints
    assert result.constraints["minCudaVersion"] == "13.0"
    assert all(dc.driver_version is None for dc in result.datacenters)


def test_cli_minimum_cuda_with_recorded_provider(monkeypatch, capsys):
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/rtx5090-min-cuda13.json").read_text()
    )
    mock_provider(monkeypatch, fixture)
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    monkeypatch.delenv("RUNPOD_ALLOWED_CUDA_VERSIONS", raising=False)
    monkeypatch.setattr(cli, "load_runpod_env", lambda: None)
    assert (
        cli.main(["availability", "--gpu", "RTX 5090", "--min-cuda-version", "13.0"])
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["constraints"]["minCudaVersion"] == "13.0"
    assert result["status"] == "available"
