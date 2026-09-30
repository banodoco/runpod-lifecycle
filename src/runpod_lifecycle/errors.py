"""Exception types for the standalone RunPod lifecycle API."""

from __future__ import annotations


class RunPodError(Exception):
    """Base class for package-level RunPod lifecycle failures."""


class LaunchFailure(RunPodError):
    """Raised when a pod cannot be provisioned or becomes unusable while starting."""

    retryable = True

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.callback_error: Exception | None = None
        self.callback_diagnostic: str | None = None


class AllocationUnknown(LaunchFailure):
    """A create request may have allocated a pod; reconcile before retrying."""

    status = "allocation_unknown"
    reconciliation_required = True
    retryable = False

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


class PostCreateHookFailure(LaunchFailure):
    """A pod is known, but its post-create state notification failed.

    The error remains a ``LaunchFailure`` for callers that already handle the
    package's launch error surface, while ``retryable=False`` prevents a
    capacity retry from creating a second pod.  ``pod_id`` is retained so the
    consumer can reconcile or otherwise recover the known allocation.
    """

    retryable = False

    def __init__(self, pod_id: str, callback_error: Exception) -> None:
        super().__init__(
            f"pod {pod_id!r} was created, but its post-create state hook failed: "
            f"{type(callback_error).__name__}: {callback_error}; reconcile the known allocation"
        )
        self.pod_id = pod_id
        self.callback_error = callback_error
        self.callback_diagnostic = f"{type(callback_error).__name__}: {callback_error}"


class NotReadyTimeout(RunPodError):
    """Raised when a pod fails to reach a ready state before the timeout."""


class SSHError(RunPodError):
    """Raised when SSH setup or execution fails."""


class TerminateError(RunPodError):
    """Raised when pod termination fails."""
