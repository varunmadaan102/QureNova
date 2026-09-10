"""Serializable experiment schema helpers."""

from __future__ import annotations

from typing import Any


def validate_experiment_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["Experiment payload must be a dictionary."]
    if not payload.get("experiment_id"):
        errors.append("experiment_id is required")
    if "result" not in payload:
        errors.append("result is required")
    return errors


def json_safe(value: Any) -> Any:
    """Convert common numpy/pandas scalar containers into JSON-safe values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    try:
        import pandas as pd
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
    except Exception:
        pass
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return str(value)
