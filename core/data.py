import re
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import streamlit as st
except ImportError:  # core data utilities remain testable without the UI dependency
    st = None

def _cache_data(func):
    if st is not None:
        return st.cache_data(show_spinner=False)(func)
    return func

def clean_dataframe(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def _schema_key(value):
    """Return a forgiving key for common biomedical CSV header conventions."""
    key = re.sub(r"[\s_]+", " ", str(value).strip().lower())
    # Accept the common Wisconsin/UCI suffix convention, e.g.
    # radius_mean, radius_se, and radius_worst.
    if key.endswith(" mean"):
        key = "mean " + key[:-5]
    elif key.endswith(" se"):
        key = key[:-3] + " error"
    elif key.endswith(" worst"):
        key = "worst " + key[:-6]
    return key


def _column_lookup(columns):
    lookup = {}
    duplicates = []
    for column in columns:
        key = _schema_key(column)
        if key in lookup:
            duplicates.append(str(column))
        else:
            lookup[key] = column
    return lookup, duplicates


def validate_schema(
    df,
    required_features,
    target_column=None,
    numeric_features=None,
    allow_extra=True,
):
    """Validate headers and numeric values before model preprocessing."""
    frame = clean_dataframe(df)
    required_features = list(required_features or [])
    numeric_features = list(numeric_features or required_features)
    lookup, duplicate_columns = _column_lookup(frame.columns)
    required_keys = {_schema_key(name) for name in required_features}
    target_key = _schema_key(target_column) if target_column else None

    missing = [name for name in required_features if _schema_key(name) not in lookup]
    extras = []
    if required_features:
        extras = [
            str(column)
            for column in frame.columns
            if _schema_key(column) not in required_keys
            and _schema_key(column) != target_key
        ]
    invalid_numeric = []
    for feature in numeric_features:
        source = lookup.get(_schema_key(feature))
        if source is None:
            continue
        values = frame[source].replace(r"^\s*$", np.nan, regex=True)
        converted = pd.to_numeric(values, errors="coerce")
        if ((values.notna()) & converted.isna()).any():
            invalid_numeric.append(feature)

    warnings, errors = [], []
    if missing:
        errors.append("Missing required features: " + ", ".join(missing[:10]))
    if duplicate_columns:
        errors.append(
            "Duplicate or ambiguous column names: "
            + ", ".join(duplicate_columns[:10])
        )
    if invalid_numeric:
        errors.append(
            "Non-numeric values found in: " + ", ".join(invalid_numeric[:10])
        )
    if extras and allow_extra:
        warnings.append(
            f"Ignoring {len(extras)} extra column(s): " + ", ".join(extras[:8])
        )
    elif extras:
        errors.append("Unexpected extra columns: " + ", ".join(extras[:10]))
    if target_column and target_key not in lookup:
        warnings.append(
            f"Target column '{target_column}' was not found. "
            "This is expected for prediction CSVs."
        )

    present_order = [
        lookup[_schema_key(feature)]
        for feature in required_features
        if _schema_key(feature) in lookup
    ]
    actual_feature_order = [
        column for column in frame.columns if _schema_key(column) in required_keys
    ]
    reordered = present_order != actual_feature_order
    if reordered:
        warnings.append("Feature columns were reordered to the canonical schema.")

    return {
        "valid": not errors,
        "missing_features": missing,
        "extra_features": extras,
        "invalid_numeric_features": invalid_numeric,
        "duplicate_columns": duplicate_columns,
        "reordered": reordered,
        "warnings": warnings,
        "errors": errors,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_cells": int(frame.isna().sum().sum()),
    }


def align_features(df, feature_names, allow_extra=True, return_report=False):
    """Normalize and align a frame to the persisted training feature order."""
    frame = clean_dataframe(df)
    feature_names = list(feature_names)
    report = validate_schema(
        frame,
        required_features=feature_names,
        numeric_features=feature_names,
        allow_extra=allow_extra,
    )
    if not report["valid"]:
        raise ValueError("; ".join(report["errors"]))

    lookup, _ = _column_lookup(frame.columns)
    aligned = pd.DataFrame(index=frame.index)
    for feature in feature_names:
        source = lookup[_schema_key(feature)]
        aligned[feature] = pd.to_numeric(frame[source], errors="coerce")
    return (aligned, report) if return_report else aligned


@_cache_data
def load_demo_dataset():
    """Load the checked-in 30-feature demo dataset used throughout the app."""
    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "demo"
        / "qurenova_demo_biomedical.csv"
    )
    return clean_dataframe(pd.read_csv(path))


def _local_path_for_breast_cancer_wisconsin_csv() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "data"
        / "datasets"
        / "breast_cancer_wisconsin.csv"
    )


@_cache_data
def load_breast_cancer_wisconsin(*, allow_download: bool = False) -> pd.DataFrame:
    """Load the canonical WDBC benchmark with explicit B/M labels.

    The local file is preferred. If absent, an optional network download is attempted
    only when requested. Numeric 1..10 diagnosis codes are never interpreted here.
    """
    local_path = _local_path_for_breast_cancer_wisconsin_csv()
    if local_path.exists():
        df = clean_dataframe(pd.read_csv(local_path))
        return _validate_wdbc_frame(df)

    if not allow_download:
        raise FileNotFoundError(
            f"Breast Cancer Wisconsin dataset not found at {local_path}."
        )

    # Prefer the sklearn-maintained bundled copy when available; it is derived from WDBC
    # and avoids fragile network-dependent startup behaviour.
    try:
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        frame = pd.DataFrame(data.data, columns=data.feature_names)
        frame["diagnosis"] = pd.Series(data.target).map({0: "M", 1: "B"})
        local_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(local_path, index=False)
        return _validate_wdbc_frame(frame)
    except Exception as exc:
        st.error(f"Could not load the canonical WDBC dataset: {type(exc).__name__}: {exc}")
        raise

def _validate_wdbc_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Fail closed unless the WDBC target uses explicit B/M semantics."""
    if "diagnosis" not in df.columns:
        raise ValueError("WDBC dataset must contain an explicit 'diagnosis' column.")
    labels = set(df["diagnosis"].dropna().astype(str).str.strip().str.upper().unique())
    if labels != {"B", "M"}:
        raise ValueError(
            "Ambiguous WDBC diagnosis labels. Expected exactly {'B','M'}; "
            "QureNova will not infer medical meaning from numeric diagnosis codes."
        )
    df = df.copy()
    df["diagnosis"] = df["diagnosis"].astype(str).str.strip().str.upper()
    for col in df.columns:
        if col != "diagnosis":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def infer_target_column(df):
    preferred = ["diagnosis", "target", "label", "class", "outcome"]
    lower = {str(c).lower(): c for c in df.columns}
    for name in preferred:
        if name in lower:
            return lower[name]
    return None

def encode_binary_target(y, label_mapping=None):
    """Encode a binary target using an explicit semantic mapping.

    Medical labels must never be inferred from numeric ordering. Without an explicit
    mapping, only canonical 0/1 or B/M labels are accepted.
    """
    y = pd.Series(y).copy()
    if y.isna().any():
        raise ValueError("Target column contains blank or missing values.")
    if y.nunique(dropna=True) != 2:
        raise ValueError("QureNova currently supports binary classification experiments.")

    if label_mapping is not None:
        mapping = dict(label_mapping)
        observed = set(y.tolist())
        if observed != set(mapping.keys()):
            raise ValueError(
                "Target labels do not exactly match the declared mapping. "
                f"Observed={sorted(map(str, observed))}, declared={sorted(map(str, mapping))}."
            )
        if set(mapping.values()) != {0, 1}:
            raise ValueError("Binary target mapping must map classes to exactly 0 and 1.")
        return y.map(mapping).astype(int), mapping

    # Safe built-in semantic encodings only.
    if pd.api.types.is_numeric_dtype(y):
        unique = set(y.tolist())
        if unique == {0, 1}:
            mapping = {0: 0, 1: 1}
            return y.astype(int), mapping
        raise ValueError(
            "Numeric target labels require an explicit label_mapping; "
            "QureNova will not infer medical meaning from numeric ordering."
        )

    normalized = y.astype(str).str.strip().str.upper()
    if set(normalized.unique()) == {"B", "M"}:
        mapping = {"B": 0, "M": 1}
        return normalized.map(mapping).astype(int), mapping

    raise ValueError(
        "Categorical target labels require an explicit label_mapping. "
        "QureNova will not infer medical meaning from label ordering."
    )

def numeric_feature_columns(df, target_column=None):
    cols = [c for c in df.columns if c != target_column]
    numeric = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    return numeric
