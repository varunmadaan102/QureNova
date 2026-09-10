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


def _local_path_for_breast_cancer_wisconsin_csv() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "data"
        / "datasets"
        / "breast_cancer_wisconsin.csv"
    )


@st.cache_data(show_spinner=False)
def load_breast_cancer_wisconsin(*, allow_download: bool = False) -> pd.DataFrame:
    """Load Breast Cancer Wisconsin dataset from local CSV.

    Expected format: one row per sample, numeric feature columns, and a binary
    target column (named 'diagnosis' or 'target' by default if present).

    If the local CSV is missing and allow_download=True, we attempt a graceful
    download using UCI if internet is available. If not available, we raise a
    Streamlit-visible error.
    """
    local_path = _local_path_for_breast_cancer_wisconsin_csv()
    if local_path.exists():
        df = clean_dataframe(pd.read_csv(local_path))
        # Ensure diagnosis is binary for the experiment pipeline.
        if "diagnosis" in df.columns:
            def _map_diag(v):
                s = str(v).strip().upper()
                if s in ("M", "MALIGNANT", "1"):
                    return 1
                if s in ("B", "BENIGN", "0"):
                    return 0
                # Legacy UCI diagnosis encoding is typically an integer 1..10.
                # We map {2,4,6,8,10} as malignant(1) and {1,3,5,7,9} as benign(0)
                # only if the value is in 1..10.
                try:
                    v_int = int(s)
                    if v_int in range(1, 11):
                        return 1 if v_int % 2 == 0 else 0
                    return pd.NA
                except Exception:
                    return pd.NA
            df["diagnosis"] = df["diagnosis"].map(_map_diag)
        # Coerce numeric features.
        for col in df.columns:
            if col == "diagnosis":
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    if not allow_download:
        st.error(
            "Breast Cancer Wisconsin CSV was not found locally. "
            "Please place it at data/datasets/breast_cancer_wisconsin.csv or enable download."
        )
        raise FileNotFoundError(str(local_path))

    # Optional / graceful download path.
    # Avoid hard dependency on network. If unavailable, fail with clear Streamlit error.
    try:
        import pandas as _pd
        import urllib.request as _rq

        # UCI Breast Cancer Wisconsin (Diagnostic) dataset.
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
        )
        st.info("Downloading Breast Cancer Wisconsin from UCI...")
        with _rq.urlopen(url, timeout=20) as resp:
            raw = resp.read().decode("utf-8")

        # UCI file is whitespace- and comma-mixed with no headers; first column is diagnosis.
        # We map the canonical Wisconsin feature names used in many references.
        columns = [
            "id",
            "diagnosis",
            "radius_mean",
            "texture_mean",
            "perimeter_mean",
            "area_mean",
            "smoothness_mean",
            "compactness_mean",
            "concavity_mean",
            "concave_points_mean",
            "symmetry_mean",
            "fractal_dimension_mean",
            "radius_se",
            "texture_se",
            "perimeter_se",
            "area_se",
            "smoothness_se",
            "compactness_se",
            "concavity_se",
            "concave_points_se",
            "symmetry_se",
            "fractal_dimension_se",
            "radius_worst",
            "texture_worst",
            "perimeter_worst",
            "area_worst",
            "smoothness_worst",
            "compactness_worst",
            "concavity_worst",
            "concave_points_worst",
            "symmetry_worst",
            "fractal_dimension_worst",
        ]

        # The UCI file is comma-separated but can contain placeholders like '?'.
        df = _pd.read_csv(
            _pd.io.common.StringIO(raw),
            header=None,
            names=columns,
        )
        # Remove id if present; keep diagnosis and numeric features.
        if "id" in df.columns:
            df = df.drop(columns=["id"])
        df = clean_dataframe(df)

        # Convert raw UCI 'diagnosis' (B/M or other legacy encodings) into
        # a clean binary target in {0,1}.
        if "diagnosis" in df.columns:
            def _map_diag(v):
                s = str(v).strip().upper()
                if s in ("M", "MALIGNANT", "1"):
                    return 1
                if s in ("B", "BENIGN", "0"):
                    return 0
                # If the dataset already uses numeric encodings 1..10, fall
                # back to binary by treating {4,5,6,7,8,9,10} as malignant is
                # not safe; instead coerce to NaN and let downstream validation
                # handle it.
                try:
                    return int(s)
                except Exception:
                    return _pd.NA

            df["diagnosis"] = df["diagnosis"].map(_map_diag)

        # Coerce all numeric feature columns to numeric (placeholders to NaN).
        for col in df.columns:
            if col == "diagnosis":
                continue
            df[col] = _pd.to_numeric(df[col], errors="coerce")

        # The canonical UCI file may include placeholders like '?'.
        # Convert numeric feature columns to numeric and coerce invalids to NaN.
        for col in df.columns:
            if col == 'diagnosis':
                continue
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Persist for next run.
        local_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(local_path, index=False)
        return df
    except Exception as exc:
        st.error(
            "Could not load Breast Cancer Wisconsin from UCI. "
            f"No local CSV found and download failed: {type(exc).__name__}: {exc}"
        )
        raise
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
