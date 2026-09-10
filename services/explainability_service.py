import pandas as pd
import numpy as np
from models.classical import build_classical_models
from core.preprocessing import build_preprocessor
from config.constants import DEMO_FEATURE_NAMES
from core.data import (
    align_features,
    encode_binary_target,
    numeric_feature_columns,
    validate_schema,
)


def _model_features(reference_df, target):
    # Prefer the canonical demo schema if it matches; otherwise derive numeric
    # feature columns dynamically so explainability stays dataset-agnostic.
    schema = validate_schema(
        reference_df,
        required_features=DEMO_FEATURE_NAMES,
        target_column=target,
        numeric_features=DEMO_FEATURE_NAMES,
    )
    if not schema["missing_features"]:
        return DEMO_FEATURE_NAMES

    features = numeric_feature_columns(reference_df, target)
    if not features:
        raise ValueError("No numeric predictive features are available for explanation.")
    return features

def explain_linear_model(reference_df, target, patient_row):
    features = _model_features(reference_df, target)
    X = align_features(reference_df, features)
    y, _ = encode_binary_target(reference_df[target])

    prep = build_preprocessor(None)
    Xp = prep.fit_transform(X)

    model = build_classical_models()["Logistic Regression"]
    model.fit(Xp, y)

    patient = prep.transform(align_features(patient_row, features))
    contributions = patient[0] * model.coef_[0]

    frame = pd.DataFrame({
        "Feature": features,
        "Contribution": contributions,
        "Absolute Contribution": np.abs(contributions),
    }).sort_values("Absolute Contribution", ascending=False)

    return frame

def explain_xgboost(reference_df, target):
    try:
        import shap
    except Exception:
        return None, "SHAP is not available."

    features = _model_features(reference_df, target)
    X = align_features(reference_df, features)
    y, _ = encode_binary_target(reference_df[target])

    models = build_classical_models()
    if "XGBoost" not in models:
        return None, "XGBoost is not available."

    model = models["XGBoost"]
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(X)

    importance = np.abs(values).mean(axis=0)
    return pd.DataFrame({
        "Feature": features,
        "Mean |SHAP|": importance
    }).sort_values("Mean |SHAP|", ascending=False), None
