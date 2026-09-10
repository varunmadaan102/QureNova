"""Clinical-facing interpretation helpers.

These thresholds are deliberately experimental. They translate model outputs
into review queues; they do not represent validated clinical risk cutoffs.
"""

from __future__ import annotations

from collections import Counter

from config.settings import RISK_THRESHOLDS


def risk_category(probability: float | None) -> str:
    if probability is None:
        return "UNAVAILABLE"
    probability = float(probability)
    if probability < RISK_THRESHOLDS["low_upper"]:
        return "LOW"
    if probability < RISK_THRESHOLDS["moderate_upper"]:
        return "MODERATE"
    return "ELEVATED"


def priority_for_risk(category: str) -> str:
    return {
        "ELEVATED": "HIGH",
        "MODERATE": "MEDIUM",
        "LOW": "ROUTINE",
    }.get(category, "REVIEW")


def agreement_summary(predictions: list[int]) -> str:
    """Describe consensus without hiding contradictory model outputs."""
    if not predictions:
        return "UNAVAILABLE"
    counts = Counter(int(value) for value in predictions)
    share = max(counts.values()) / len(predictions)
    if share == 1:
        return "HIGH AGREEMENT"
    if share >= 2 / 3:
        return "MODERATE AGREEMENT"
    return "LOW AGREEMENT"
