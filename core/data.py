import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

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


@st.cache_data(show_spinner=False)
def load_demo_dataset():
    """Load the checked-in 30-feature demo dataset used throughout the app."""
    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "demo"
        / "qurenova_demo_biomedical.csv"
    )
    return clean_dataframe(pd.read_csv(path))

def infer_target_column(df):
    preferred = ["diagnosis", "target", "label", "class", "outcome"]
    lower = {str(c).lower(): c for c in df.columns}
    for name in preferred:
        if name in lower:
            return lower[name]
    return None

def encode_binary_target(y):
    y = pd.Series(y).copy()
    if y.isna().any():
        raise ValueError("Target column contains blank or missing values.")
    if y.nunique(dropna=True) != 2:
        raise ValueError("QureNova currently supports binary classification experiments.")
    if not pd.api.types.is_numeric_dtype(y):
        classes = sorted(y.dropna().astype(str).unique().tolist())
        mapping = {classes[0]: 0, classes[1]: 1}
        return y.astype(str).map(mapping).astype(int), mapping
    unique = sorted(y.dropna().unique().tolist())
    mapping = {unique[0]: 0, unique[1]: 1}
    return y.map(mapping).astype(int), mapping

def numeric_feature_columns(df, target_column=None):
    cols = [c for c in df.columns if c != target_column]
    numeric = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    return numeric
