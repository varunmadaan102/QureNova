from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from core.data import (
    align_features,
    clean_dataframe,
    encode_binary_target,
    numeric_feature_columns,
    validate_schema,
)
from core.preprocessing import build_preprocessor
from models.classical import build_classical_models


_ROOT = Path(__file__).resolve().parents[1]
_MODEL_DIRS = (_ROOT / "results" / "models", _ROOT / "models" / "trained")
_ARTIFACT_NAMES = {
    "Logistic Regression": "logistic_regression.joblib",
    "SVM (RBF)": "svm_rbf.joblib",
    "XGBoost": "xgboost.joblib",
}


def _read_json(path):
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _load_persisted_models(target, model_dir=None):
    """Load a complete, metadata-backed persisted model bundle if available."""
    directories = (Path(model_dir),) if model_dir else _MODEL_DIRS
    for directory in directories:
        metadata_path = directory / "feature_metadata.json"
        if not metadata_path.exists():
            continue
        try:
            import json

            metadata = json.loads(_read_json(metadata_path))
            if metadata.get("target_column") != target:
                continue
            feature_names = list(metadata["feature_names"])
            models = {}
            for name, filename in _ARTIFACT_NAMES.items():
                artifact = directory / filename
                if artifact.exists():
                    models[name] = joblib.load(artifact)
            if models:
                return (
                    models,
                    feature_names,
                    dict(metadata.get("target_mapping", {})),
                    f"persisted artifacts ({directory})",
                )
        except (OSError, ValueError, KeyError, TypeError, ImportError):
            # A partial or incompatible artifact must not block the fallback.
            continue
    return None


@st.cache_resource(show_spinner=False)
def train_models_for_prediction(
    reference_df,
    target,
    model_dir=None,
    prefer_persisted=True,
    return_source=False,
):
    reference_df = clean_dataframe(reference_df)
    if target not in reference_df.columns:
        raise ValueError(f"Reference dataset is missing target column '{target}'.")

    if prefer_persisted:
        persisted = _load_persisted_models(target, model_dir=model_dir)
        if persisted is not None:
            models, features, mapping, source = persisted
            reference_schema = validate_schema(
                reference_df,
                required_features=features,
                target_column=target,
                numeric_features=features,
                allow_extra=True,
            )
            if reference_schema["valid"]:
                result = (models, None, features, mapping)
                return result + (source,) if return_source else result

    feature_names = numeric_feature_columns(reference_df, target)
    if not feature_names:
        raise ValueError("Reference dataset has no numeric predictive features.")
    X = align_features(reference_df, feature_names)
    y, mapping = encode_binary_target(reference_df[target])

    # This is intentionally a visible fallback when no persisted bundle exists.
    prep = build_preprocessor(None)
    Xp = prep.fit_transform(X)
    trained = {}
    for name, model in build_classical_models().items():
        model.fit(Xp, y)
        trained[name] = model

    result = (trained, prep, feature_names, mapping)
    return result + ("trained in memory fallback",) if return_source else result


def predict_patients(
    reference_df,
    target,
    patient_df,
    model_dir=None,
    prefer_persisted=True,
    return_source=False,
):
    models, prep, features, mapping, source = train_models_for_prediction(
        reference_df,
        target,
        model_dir=model_dir,
        prefer_persisted=prefer_persisted,
        return_source=True,
    )
    patient_df = clean_dataframe(patient_df)
    schema = validate_schema(
        patient_df,
        required_features=features,
        numeric_features=features,
        allow_extra=True,
    )
    if not schema["valid"]:
        raise ValueError("Patient CSV schema error: " + "; ".join(schema["errors"]))
    aligned, _ = align_features(patient_df, features, return_report=True)
    Xp = aligned if prep is None else prep.transform(aligned)
    rows = []

    for idx in range(len(patient_df)):
        row = {"Patient": int(idx + 1)}
        votes = []
        probabilities = []

        for name, model in models.items():
            one = Xp.iloc[idx : idx + 1] if hasattr(Xp, "iloc") else Xp[idx : idx + 1]
            pred = int(model.predict(one)[0])
            prob = (
                float(model.predict_proba(one)[0, 1])
                if hasattr(model, "predict_proba")
                else None
            )
            row[f"{name} Prediction"] = pred
            row[f"{name} Probability"] = prob
            votes.append(pred)
            if prob is not None:
                probabilities.append(prob)

        row["Consensus Prediction"] = int(round(np.mean(votes)))
        row["Average Probability"] = (
            float(np.mean(probabilities)) if probabilities else None
        )
        row["Model Agreement"] = "High" if len(set(votes)) == 1 else "Mixed"
        if row["Average Probability"] is not None:
            distance = abs(row["Average Probability"] - 0.5)
            row["Model Confidence"] = (
                "High"
                if distance >= 0.25
                else "Moderate"
                if distance >= 0.05
                else "Low"
            )
        rows.append(row)

    result = pd.DataFrame(rows)
    return (
        (result, mapping, features, source)
        if return_source
        else (result, mapping, features)
    )
