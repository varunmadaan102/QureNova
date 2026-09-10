"""Deterministic, stratified dataset splits shared by benchmark runners."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def common_final_split(X, y, *, test_size=0.20, random_state=42):
    """Create one untouched stratified test set for fair future benchmarking."""
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=float(test_size),
        stratify=y,
        random_state=int(random_state),
    )
    return np.asarray(train_idx), np.asarray(test_idx)
