"""Train and persist the classical QureAI prediction models.

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --input data/demo/qureai_demo_biomedical.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_MODEL_NAMES = ("Logistic Regression", "SVM (RBF)", "XGBoost")

from config.constants import DEMO_FEATURE_NAMES
from config.settings import CV_FOLDS, DEFAULT_PCA_COMPONENTS, RANDOM_STATE
from core.data import (
    clean_dataframe,
    encode_binary_target,
    infer_target_column,
    numeric_feature_columns,
    align_features,
)
from core.evaluation import evaluate_predictions, summarize_fold_metrics
from core.preprocessing import build_preprocessor, safe_pca_components
from core.validation import validate_dataset
from models.classical import build_classical_models


def _score(model, frame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(frame)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(frame)
    return None


def _slug(name):
    return {
        "Logistic Regression": "logistic_regression",
        "SVM (RBF)": "svm_rbf",
        "XGBoost": "xgboost",
    }.get(name, name.lower().replace(" ", "_"))


def _cross_validate(X, y, models, folds, pca_components):
    splitter = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=RANDOM_STATE
    )
    results = {}
    for name, base_model in models.items():
        fold_metrics = []
        started = time.perf_counter()
        for train_idx, test_idx in splitter.split(X, y):
            # The preprocessor is inside each fold so no validation statistics
            # (including PCA components) are learned from held-out rows.
            pipeline = Pipeline(
                [
                    ("preprocessor", build_preprocessor(pca_components)),
                    ("model", clone(base_model)),
                ]
            )
            pipeline.fit(X.iloc[train_idx], y[train_idx])
            pred = pipeline.predict(X.iloc[test_idx])
            score = _score(pipeline, X.iloc[test_idx])
            fold_metrics.append(
                evaluate_predictions(y[test_idx], pred, score)
            )
        results[name] = {
            "fold_metrics": fold_metrics,
            "summary": summarize_fold_metrics(fold_metrics),
            "timing_seconds": time.perf_counter() - started,
            "preprocessing_fit": "inside each CV fold",
        }
    return results


def train_and_persist(
    input_path,
    output_dir,
    target_column=None,
    pca_components=DEFAULT_PCA_COMPONENTS,
    cv_folds=CV_FOLDS,
    include_xgboost=True,
):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = clean_dataframe(pd.read_csv(input_path))
    target_column = target_column or infer_target_column(df)
    if not target_column or target_column not in df.columns:
        raise ValueError("A target column is required (for example: diagnosis).")
    demo_schema = validate_dataset(
        df, target_column=target_column, required_features=DEMO_FEATURE_NAMES
    )
    feature_names = (
        DEMO_FEATURE_NAMES
        if not demo_schema["missing_features"]
        else numeric_feature_columns(df, target_column)
    )
    if not feature_names:
        raise ValueError("No numeric predictive features were found.")
    validation = validate_dataset(
        df, target_column=target_column, required_features=feature_names
    )
    if not validation["valid"]:
        raise ValueError("; ".join(validation["errors"]))

    X = align_features(df, feature_names)
    y, target_mapping = encode_binary_target(df[target_column])
    folds = min(int(cv_folds), int(y.value_counts().min()))
    if folds < 2:
        raise ValueError("Each target class needs at least two rows for CV.")
    components = safe_pca_components(pca_components, len(X), len(feature_names))

    models = build_classical_models(RANDOM_STATE)
    if not include_xgboost:
        models.pop("XGBoost", None)
    for name, filename in {
        key: _slug(key) + ".joblib" for key in _MODEL_NAMES
    }.items():
        if name not in models:
            stale = output_dir / filename
            if stale.exists():
                stale.unlink()
    evaluation = _cross_validate(
        X, y.to_numpy(), models, folds=folds, pca_components=components
    )

    artifacts = {}
    for name, model in models.items():
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(components)),
                ("model", clone(model)),
            ]
        )
        pipeline.fit(X, y)
        artifact_path = output_dir / f"{_slug(name)}.joblib"
        dump(pipeline, artifact_path)
        artifacts[name] = artifact_path.name

    feature_metadata = {
        "schema_version": "1.0",
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "numeric_features": feature_names,
        "target_column": target_column,
        "target_mapping": target_mapping,
        "pca_components": components,
        "preprocessing": "median imputation -> StandardScaler -> PCA",
    }
    dataset_metadata = {
        "source": str(input_path),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "feature_count": len(feature_names),
        "target_column": target_column,
        "class_distribution": {
            str(key): int(value)
            for key, value in df[target_column].value_counts().items()
        },
        "schema_validation": validation,
    }
    (output_dir / "evaluation_results.json").write_text(
        json.dumps(
            {
                "models": evaluation,
                "cv_folds": folds,
                "pca_components": components,
                "artifacts": artifacts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "feature_metadata.json").write_text(
        json.dumps(feature_metadata, indent=2), encoding="utf-8"
    )
    (output_dir / "dataset_metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2), encoding="utf-8"
    )
    return artifacts, evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "demo" / "qureai_demo_biomedical.csv"),
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "results" / "models")
    )
    parser.add_argument("--target", default=None)
    parser.add_argument("--pca-components", type=int, default=DEFAULT_PCA_COMPONENTS)
    parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    parser.add_argument(
        "--disable-xgboost",
        action="store_true",
        help="Persist only Logistic Regression and SVM.",
    )
    args = parser.parse_args()
    artifacts, evaluation = train_and_persist(
        args.input,
        args.output_dir,
        target_column=args.target,
        pca_components=args.pca_components,
        cv_folds=args.cv_folds,
        include_xgboost=not args.disable_xgboost,
    )
    print(f"Saved {len(artifacts)} model artifact(s) to {args.output_dir}")
    for name, path in artifacts.items():
        print(f"- {name}: {path}")
    print(
        "Best mean F1: "
        + max(
            evaluation,
            key=lambda name: evaluation[name]["summary"]["f1"]["mean"] or -1,
        )
    )


if __name__ == "__main__":
    main()
