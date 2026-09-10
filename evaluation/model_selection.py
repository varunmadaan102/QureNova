"""Model-selection utility that accounts for quality and compute cost."""

from __future__ import annotations


def utility_score(metrics: dict, *, weights: dict | None = None) -> float:
    """Return a transparent 0-1 model utility score.

    This is a research ranking heuristic, not a clinical decision rule.
    """
    weights = weights or {
        "roc_auc": 0.30,
        "f1": 0.25,
        "sensitivity": 0.20,
        "specificity": 0.15,
        "stability": 0.05,
        "runtime_efficiency": 0.05,
    }
    total = 0.0
    used = 0.0
    for key, weight in weights.items():
        value = metrics.get(key)
        if value is None:
            continue
        try:
            total += float(value) * float(weight)
            used += float(weight)
        except (TypeError, ValueError):
            continue
    return total / used if used else 0.0
