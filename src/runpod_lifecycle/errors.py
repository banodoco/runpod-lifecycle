"""Exception types for the standalone RunPod lifecycle API."""

from __future__ import annotations


class RunPodError(Exception):
    """Base class for package-level RunPod lifecycle failures."""


class LaunchFailure(RunPodError):
    """Raised when a pod cannot be provisioned or becomes unusable while starting."""


class NotReadyTimeout(RunPodError):
    """Raised when a pod fails to reach a ready state before the timeout."""


class SSHError(RunPodError):
    """Raised when SSH setup or execution fails."""


class TerminateError(RunPodError):
    """Raised when pod termination fails."""
