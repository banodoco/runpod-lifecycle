"""Unpaid, bounded checks of client egress, before requesting a pod."""
from __future__ import annotations

import os
import socket

import httpx

SSH_UNAVAILABLE = "SSH egress unavailable; use --transport https"
PROXY_PROBE = "https://astrid-egress-preflight-8000.proxy.runpod.net/health"


def preflight_transport(transport: str, *, timeout: float = 5) -> None:
    """Check an egress route, not capacity or readiness of an unallocated pod.

    A configured HTTPS proxy is treated conservatively as HTTPS-only. Never
    attempt raw SSH or CONNECT-to-SSH through it. Direct clients probe RunPod's
    SSH gateway (override the endpoint for a known site-specific TCP test).
    """
    if transport == "ssh":
        if any(os.environ.get(name) for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")):
            raise RuntimeError(SSH_UNAVAILABLE)
        host = os.environ.get("RUNPOD_SSH_PREFLIGHT_HOST", "ssh.runpod.io")
        port = int(os.environ.get("RUNPOD_SSH_PREFLIGHT_PORT", "22"))
        try:
            with socket.create_connection((host, port), timeout=timeout):
                pass
        except OSError:
            raise RuntimeError(SSH_UNAVAILABLE) from None
    elif transport == "https":
        try:
            # A provider 404 for this nonexistent pod proves HTTPS reaches the
            # proxy. Authentication/egress denials must never count as success.
            response = httpx.get(PROXY_PROBE, timeout=timeout, follow_redirects=False)
            if response.status_code not in (200, 404):
                raise RuntimeError("HTTPS pod proxy egress unavailable (HTTP %s)" % response.status_code)
        except httpx.HTTPError:
            raise RuntimeError("HTTPS pod proxy egress unavailable") from None
    else:
        raise ValueError("transport must be ssh or https")
