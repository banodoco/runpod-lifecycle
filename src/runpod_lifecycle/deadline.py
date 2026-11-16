"""Validation for provider-side absolute pod termination deadlines."""

import re
from datetime import datetime, timezone


def validate_terminate_after(value: str | None) -> str | None:
    """Return a normalized future UTC timestamp, or reject it before a provider call."""
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|\+00:00)", value
    ):
        raise ValueError("terminate_after must be an absolute UTC timestamp")
    try:
        deadline = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("terminate_after must be an absolute UTC timestamp") from exc
    if deadline <= datetime.now(timezone.utc):
        raise ValueError("terminate_after must be in the future")
    return deadline.isoformat().replace("+00:00", "Z")
