"""Input reliability checks relative to the reference training population."""

from __future__ import annotations

import numpy as np
import pandas as pd


def assess_input_reliability(reference: pd.DataFrame, patient: pd.DataFrame) -> dict:
    """Return a conservative distribution-shift signal for one patient row.

    The score is a standardized distance from the reference medians using
    interquartile ranges. It is a model-input quality warning, not an anomaly
    or medical abnormality detector.
    """
    reference = reference.apply(pd.to_numeric, errors="coerce")
    patient = patient.apply(pd.to_numeric, errors="coerce")
    missing = int(patient.isna().sum(axis=1).iloc[0])
    if reference.empty or patient.empty:
        return {"level": "LOW", "distance": None, "reason": "No reference values available."}

    median = reference.median()
    spread = (reference.quantile(0.75) - reference.quantile(0.25)).replace(0, np.nan)
    spread = spread.fillna(reference.std(ddof=0)).replace(0, 1.0).fillna(1.0)
    distances = ((patient.iloc[0] - median).abs() / spread).replace(
        [np.inf, -np.inf], np.nan
    )
    distance = float(distances.fillna(0).mean())
    if missing or distance >= 4:
        level = "LOW"
    elif distance >= 2:
        level = "MODERATE"
    else:
        level = "HIGH"

    reasons = []
    if missing:
        reasons.append(f"{missing} missing feature(s) will be imputed")
    if distance >= 2:
        reasons.append("some measurements differ substantially from the reference population")
    reason = "; ".join(reasons) if reasons else "input is within the reference distribution used for this check"
    return {"level": level, "distance": distance, "reason": reason}
