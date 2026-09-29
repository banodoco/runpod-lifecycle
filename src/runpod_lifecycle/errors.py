"""Exception types for the standalone RunPod lifecycle API."""

from __future__ import annotations


class RunPodError(Exception):
    """Base class for package-level RunPod lifecycle failures."""


class LaunchFailure(RunPodError):
    """Raised when a pod cannot be provisioned or becomes unusable while starting."""


class AllocationUnknown(LaunchFailure):
    """A create request may have allocated a pod; reconcile before retrying."""

    status = "allocation_unknown"
    reconciliation_required = True

    def __init__(
        self, request_name: str, gpu_type: str, ram_tier: int,
        storage_name: str | None, storage_volume_id: str | None,
    ) -> None:
        self.request_name = request_name
        self.gpu_type = gpu_type
        self.ram_tier = ram_tier
        self.storage_name = storage_name
        self.storage_volume_id = storage_volume_id
        super().__init__(
            f"allocation_unknown for request {request_name!r}; reconcile the provider "
            "allocation before retrying (pod ID is unknown)"
        )


class NotReadyTimeout(RunPodError):
    """Raised when a pod fails to reach a ready state before the timeout."""


class SSHError(RunPodError):
    """Raised when SSH setup or execution fails."""


class TerminateError(RunPodError):
    """Raised when pod termination fails."""
