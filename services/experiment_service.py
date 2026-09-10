import time
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV

from config.settings import RANDOM_STATE, CV_FOLDS, DEFAULT_PCA_COMPONENTS, DEFAULT_QUANTUM_QUBITS, MAX_QUANTUM_TRAIN_SAMPLES
from config.constants import DEMO_FEATURE_NAMES
from core.data import (
    clean_dataframe,
    infer_target_column,
    encode_binary_target,
    numeric_feature_columns,
    align_features,
    validate_schema,
)
from core.validation import validate_dataset
from core.datasets import DATASET_PROFILES, resolve_target_definition
from core.preprocessing import build_preprocessor, safe_pca_components
from core.evaluation import evaluate_predictions, summarize_fold_metrics, build_confusion
from models.registry import build_registered_models
from models.quantum import build_precomputed_qsvc
from quantum.feature_maps import qiskit_available
from quantum.kernel import compute_quantum_kernel, kernel_diagnostics
from quantum.alignment import kernel_target_alignment

# Quantum is executed in an isolated worker process because native backends (e.g., Aer/QML)
# can segfault the interpreter. Subprocess isolation lets Streamlit remain alive.
import json
import os
import subprocess
import tempfile

def _score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None

def _classical_cv(X, y, folds, random_state, pca_components=None):
    results = {}
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    for name, base_model in build_registered_models(random_state).items():
        fold_metrics, all_true, all_pred = [], [], []
        started = time.perf_counter()

        for train_idx, test_idx in splitter.split(X, y):
            # Fit every preprocessing step on the training fold only. Fitting
            # once before CV leaks validation-fold statistics into the model.
            preprocessor = build_preprocessor(pca_components)
            X_train_source = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
            X_test_source = X.iloc[test_idx] if hasattr(X, "iloc") else X[test_idx]
            X_train = preprocessor.fit_transform(X_train_source)
            X_test = preprocessor.transform(X_test_source)
            model = clone(base_model)
            model.fit(X_train, y[train_idx])
            pred = model.predict(X_test)
            score = _score(model, X_test)
            fold_metrics.append(evaluate_predictions(y[test_idx], pred, score))
            all_true.extend(y[test_idx].tolist())
            all_pred.extend(pred.tolist())

        results[name] = {
            "fold_metrics": fold_metrics,
            "summary": summarize_fold_metrics(fold_metrics),
            "timing_seconds": time.perf_counter() - started,
            "validation": f"Stratified {folds}-fold CV",
            "confusion_matrix": build_confusion(all_true, all_pred),
        }
    return results

def _classical_pca_cv(X, y, folds, random_state, pca_components):
    """Classical models with PCA - control for quantum kernel comparison."""
    results = {}
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    for name, base_model in build_registered_models(random_state).items():
        fold_metrics, all_true, all_pred = [], [], []
        started = time.perf_counter()

        for train_idx, test_idx in splitter.split(X, y):
            preprocessor = build_preprocessor(pca_components)
            X_train_source = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
            X_test_source = X.iloc[test_idx] if hasattr(X, "iloc") else X[test_idx]
            X_train = preprocessor.fit_transform(X_train_source)
            X_test = preprocessor.transform(X_test_source)
            model = clone(base_model)
            model.fit(X_train, y[train_idx])
            pred = model.predict(X_test)
            score = _score(model, X_test)
            fold_metrics.append(evaluate_predictions(y[test_idx], pred, score))
            all_true.extend(y[test_idx].tolist())
            all_pred.extend(pred.tolist())

        results[name] = {
            "fold_metrics": fold_metrics,
            "summary": summarize_fold_metrics(fold_metrics),
            "timing_seconds": time.perf_counter() - started,
            "validation": f"Stratified {folds}-fold CV (with PCA control)",
            "confusion_matrix": build_confusion(all_true, all_pred),
        }
    return results

def _quantum_holdout_isolated(X, y, qubits, feature_map):
    """Run quantum kernel QSVC in a separate process to prevent native segfaults."""
    # IMPORTANT: keep classical path alive even if quantum segfaults.
    if not qiskit_available():
        return {
            "available": False,
            "reason": "Qiskit is not installed or failed to import. Classical experiments still completed."
        }

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    import json
    import subprocess
    import tempfile
    import sys

    started = time.perf_counter()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    # Quantum kernels scale quadratically. Keep the demonstration bounded.
    if len(X_train) > MAX_QUANTUM_TRAIN_SAMPLES:
        from sklearn.model_selection import train_test_split as _stratified_sample
        X_train, _, y_train, _ = _stratified_sample(
            X_train, y_train, train_size=MAX_QUANTUM_TRAIN_SAMPLES,
            stratify=y_train, random_state=RANDOM_STATE
        )

    components = safe_pca_components(qubits, len(X_train), X_train.shape[1])

    # Quantum feature maps require finite numeric inputs; ensure NaNs are
    # handled inside the worker-safe preprocessing.
    from sklearn.impute import SimpleImputer

    prep = __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=components, random_state=RANDOM_STATE)),
    ])
    X_train_q = prep.fit_transform(X_train)
    X_test_q = prep.transform(X_test)

    # Worker expects concatenated [X_train_q; X_test_q]
    X_worker = np.concatenate([X_train_q, X_test_q], axis=0)

    worker_out_dir = tempfile.mkdtemp(prefix="quantum_worker_")
    out_path = os.path.join(worker_out_dir, "result.json")

    worker_script = os.path.join(os.path.dirname(__file__), "..", "scripts", "quantum_holdout_worker.py")
    worker_script = os.path.abspath(worker_script)

    cmd = [
        sys.executable,
        worker_script,
        "--x",
        json.dumps(X_worker.tolist()),
        "--y_train",
        json.dumps(np.asarray(y_train).tolist()),
        "--y_test",
        json.dumps(np.asarray(y_test).tolist()),
        "--feature_map",
        str(feature_map),
        "--num_qubits",
        str(int(components)),
        "--out",
        out_path,
    ]

    # IMPORTANT: allow enough time for quantum to run, but don't hang the Streamlit request.
    # 120s is a reasonable cap; adjust if your deployment is slower.
    timeout_sec = int(os.environ.get("QURENOVA_QUANTUM_TIMEOUT_SECONDS", "120"))

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)

    if proc.returncode != 0:
        return {
            "available": False,
            "reason": f"Quantum worker failed (exit code {proc.returncode}).",
            "worker_stdout": proc.stdout[-4000:],
            "worker_stderr": proc.stderr[-4000:],
            "timing_seconds": time.perf_counter() - started,
        }

    if not os.path.exists(out_path):
        return {
            "available": False,
            "reason": "Quantum worker did not produce result.json.",
            "worker_stdout": proc.stdout[-4000:],
            "worker_stderr": proc.stderr[-4000:],
            "timing_seconds": time.perf_counter() - started,
        }

    with open(out_path, "r", encoding="utf-8") as f:
        q = json.load(f)

    q["configuration"] = q.get("configuration") or {
        "backend": "Qiskit fidelity quantum kernel simulator",
        "feature_map": feature_map,
        "num_qubits": components,
        "validation": "Stratified holdout (quantum demonstration)",
        "comparison_status": "Contextual only; not directly comparable to CV means",
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
    }
    q["timing_seconds"] = time.perf_counter() - started

    return q


def _quantum_skipped(reason):
    return {
        "available": False,
        "status": "not_run",
        "reason": reason,
    }

def run_experiment(df, target_column=None, config=None):
    config = config or {}
    df = clean_dataframe(df)

    # Prefer explicit target column. If not provided, infer from columns.
    target_column = target_column or infer_target_column(df)

    if not target_column:
        raise ValueError("No target column found. Select a binary target column before running an experiment.")

    # Validate schema against the demo feature contract if possible.
    # If the dataset doesn't match, we fall back to all numeric feature columns.
    demo_schema = validate_schema(
        df, DEMO_FEATURE_NAMES, target_column=target_column
    )

    feature_names = (
        DEMO_FEATURE_NAMES
        if (not demo_schema["missing_features"] and len(demo_schema["invalid_numeric_features"]) == 0)
        else numeric_feature_columns(df, target_column)
    )
    validation = validate_dataset(
        df, target_column=target_column, required_features=feature_names
    )
    if not validation["valid"]:
        raise ValueError("; ".join(validation["errors"]))
    # If the dataset uses the raw UCI Wisconsin format, 'diagnosis' may contain
    # multiple numeric encodings. We require exactly two classes for binary
    # classification.
    if df[target_column].nunique(dropna=True) != 2:
        return {
            "class_distribution": {str(k): int(v) for k, v in df[target_column].value_counts().to_dict().items()},
            "class_distribution_note": "Quantum/classical experiment requires exactly 2 classes; returning without running models.",
            "quantum_results": _quantum_skipped("Dataset target is not binary; exactly two classes required."),
            "available": False,
        }

    if not feature_names:
        raise ValueError("No numeric predictive features were found.")

    X = align_features(df, feature_names)
    # Resolve medical target semantics explicitly. Do not infer meaning from numeric codes.
    dataset_profile_name = config.get("dataset_profile")
    target_definition = None
    if dataset_profile_name:
        target_definition = resolve_target_definition(dataset_profile_name)
        if target_column != target_definition.column:
            raise ValueError(
                f"Dataset profile expects target '{target_definition.column}', got '{target_column}'."
            )
        y, mapping = encode_binary_target(
            df[target_column], target_definition.label_mapping
        )
    else:
        y, mapping = encode_binary_target(df[target_column])

    pca_components = safe_pca_components(
        config.get("pca_components", DEFAULT_PCA_COMPONENTS),
        len(X), len(feature_names)
    )

    folds = min(int(config.get("cv_folds", CV_FOLDS)), int(y.value_counts().min()))
    if folds < 2:
        raise ValueError("Each class needs at least two samples for cross-validation.")

    started = time.perf_counter()
    classical = _classical_cv(
        X,
        y.to_numpy(),
        folds,
        RANDOM_STATE,
        pca_components=None,
    )

    classical_pca = _classical_pca_cv(
        X,
        y.to_numpy(),
        folds,
        RANDOM_STATE,
        pca_components=pca_components,
    )

    if config.get("run_quantum", False):
        quantum = _quantum_holdout_isolated(
            X.to_numpy(),
            y.to_numpy(),
            int(config.get("quantum_qubits", DEFAULT_QUANTUM_QUBITS)),
            config.get("feature_map", "zz"),
        )
    else:
        quantum = _quantum_skipped(
            "Quantum simulation was not run. Enable 'Enable quantum kernel computation' "
            "to execute the optional Qiskit workflow."
        )

    return {
        "metadata": {
            "experiment_version": "2.0.0",
            "runtime_seconds": time.perf_counter() - started,
            "target_column": target_column,
            "target_mapping": mapping,
            "target_semantics": (
                {"task_type": target_definition.task_type, "class_names": target_definition.class_names, "source": target_definition.source}
                if target_definition else None
            ),
            "dataset_profile": dataset_profile_name,
            "platform_scope": "Biomedical research MVP; dataset profiles can be extended",
        },
        "dataset": {
            "rows": int(len(df)),
            "features": int(len(feature_names)),
            "feature_names": feature_names,
            "class_distribution": {str(k): int(v) for k, v in df[target_column].value_counts().to_dict().items()},
            "target_classes": (target_definition.class_names if target_definition else None),
        },
        "validation": validation,
        "preprocessing": {
            "pipeline": "Fold-fitted median imputation → StandardScaler (classical); StandardScaler → PCA (quantum)",
            "pca_components": pca_components,
            "fit_scope": "Each CV training fold",
            "feature_selection": "Canonical validated feature set",
        },
        "classical_results": classical,
        "classical_pca_results": classical_pca,
        "quantum_results": quantum,
        "artifacts": {
            "feature_names": feature_names,
            "reference_df": X,
        },
    }
