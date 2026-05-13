"""Storage volume helpers and pure storage-health parsing utilities."""

from __future__ import annotations

import logging
from typing import Any

import requests

from .api import get_network_volumes

logger = logging.getLogger("runpod_lifecycle.storage")

STORAGE_CHECK_COMMAND = """
            echo "=== WORKSPACE STORAGE ==="
            df -h /workspace 2>/dev/null | tail -1
            echo ""
            echo "=== WORKSPACE USAGE DETAILS ==="
            df -BG /workspace 2>/dev/null | tail -1
            echo ""
            echo "=== LARGEST DIRECTORIES ==="
            du -sh /workspace/*/ 2>/dev/null | sort -rh | head -10
            """


def _expand_network_volume(api_key: str, volume_id: str, size_gb: int) -> bool:
    """Expand a network volume to the requested size in GiB."""
    try:
        response = requests.patch(
            f"https://rest.runpod.io/v1/networkvolumes/{volume_id}",
            json={"size": size_gb},
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        return response.status_code == 200
    except Exception as exc:
        logger.warning("Failed to expand network volume %s: %s", volume_id, exc)
        return False


def check_and_expand_storage(
    api_key: str,
    volume_id: str,
    min_free_gb: int = 50,
    storage_name: str | None = None,
    target_size_gb: int | None = None,
) -> dict[str, Any]:
    """Check current network volume size and expand when below the requested threshold."""
    try:
        volumes = get_network_volumes(api_key)
        volume_info = next((volume for volume in volumes if volume.get("id") == volume_id), None)
        label = storage_name or volume_id

        if not volume_info:
            return {
                "ok": True,
                "expanded": False,
                "current_size_gb": None,
                "target_size_gb": None,
                "message": f"Could not find volume info for {label}",
            }

        current_size_gb = volume_info.get("size", 0)
        requested_target_gb = int(target_size_gb or 0)
        if requested_target_gb > 0 and current_size_gb < requested_target_gb:
            expanded = _expand_network_volume(api_key, volume_id, requested_target_gb)
            return {
                "ok": True,
                "expanded": expanded,
                "current_size_gb": current_size_gb,
                "target_size_gb": requested_target_gb,
                "message": (
                    f"Expanded storage '{label}' to {requested_target_gb} GB"
                    if expanded
                    else f"Expansion failed for storage '{label}', continuing anyway"
                ),
            }

        if current_size_gb >= 100:
            return {
                "ok": True,
                "expanded": False,
                "current_size_gb": current_size_gb,
                "target_size_gb": current_size_gb,
                "message": f"Storage '{label}' has adequate capacity ({current_size_gb} GB)",
            }

        target_size_gb = current_size_gb + min_free_gb
        expanded = _expand_network_volume(api_key, volume_id, target_size_gb)
        return {
            "ok": True,
            "expanded": expanded,
            "current_size_gb": current_size_gb,
            "target_size_gb": target_size_gb,
            "message": (
                f"Expanded storage '{label}' to {target_size_gb} GB"
                if expanded
                else f"Expansion failed for storage '{label}', continuing anyway"
            ),
        }
    except Exception as exc:
        return {
            "ok": True,
            "expanded": False,
            "current_size_gb": None,
            "target_size_gb": None,
            "message": f"Error checking storage space: {exc}",
        }


def get_storage_volume_id(api_key: str, storage_name: str | None) -> str | None:
    """Resolve a storage volume name to its RunPod network volume ID."""
    if not storage_name:
        return None

    volumes = get_network_volumes(api_key)
    for volume in volumes:
        if volume.get("name") == storage_name:
            return volume.get("id")
    return None


def _parse_size(size_value: str) -> int:
    normalized = size_value.strip()
    if normalized.endswith("T"):
        return int(float(normalized[:-1]) * 1024)
    if normalized.endswith("G"):
        return int(float(normalized[:-1]))
    if normalized.endswith("M"):
        return int(float(normalized[:-1]) / 1024)
    if normalized.endswith("K"):
        return 0
    return int(float(normalized))


def parse_df_output(raw_output: str) -> dict[str, int]:
    """Parse the `/workspace` `df` output into normalized GiB values."""
    parsed = {
        "total_gb": 0,
        "used_gb": 0,
        "free_gb": 0,
        "percent_used": 0,
    }

    for line in raw_output.splitlines():
        stripped = line.strip()
        if "===" in stripped or "/workspace" not in stripped:
            continue

        parts = stripped.split()
        if len(parts) < 5:
            continue

        try:
            parsed["total_gb"] = _parse_size(parts[1])
            parsed["used_gb"] = _parse_size(parts[2])
            parsed["free_gb"] = _parse_size(parts[3])
            parsed["percent_used"] = int(parts[4].rstrip("%"))
            break
        except (ValueError, IndexError):
            continue

    return parsed


def evaluate_storage_health(
    parsed: dict[str, int],
    api_total_gb: int | None,
    min_free_gb: int,
    max_percent_used: int,
) -> dict[str, Any]:
    """Evaluate parsed storage metrics and decide whether expansion is needed."""
    del api_total_gb
    free_gb = parsed.get("free_gb", 0)
    percent_used = parsed.get("percent_used", 0)

    if percent_used >= max_percent_used:
        return {
            "healthy": False,
            "needs_expansion": True,
            "message": f"CRITICAL: {percent_used}% used, only {free_gb}GB free!",
        }
    if free_gb < min_free_gb:
        return {
            "healthy": False,
            "needs_expansion": True,
            "message": f"LOW SPACE: Only {free_gb}GB free (need {min_free_gb}GB)",
        }
    return {
        "healthy": True,
        "needs_expansion": False,
        "message": f"OK: {free_gb}GB free ({100 - percent_used}% available)",
    }


__all__ = [
    "STORAGE_CHECK_COMMAND",
    "_expand_network_volume",
    "check_and_expand_storage",
    "evaluate_storage_health",
    "get_storage_volume_id",
    "parse_df_output",
]
