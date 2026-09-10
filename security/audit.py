"""Minimal structured audit event helper."""

from __future__ import annotations

from datetime import datetime, timezone


def audit_event(action: str, *, actor: str = "local-user", **details):
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": str(action),
        "actor": str(actor),
        "details": dict(details),
    }
