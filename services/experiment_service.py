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
from core.preprocessing import build_preprocessor, safe_pca_components
from core.evaluation import evaluate_predictions, summarize_fold_metrics, build_confusion
from models.classical import build_classical_models
from models.quantum import build_precomputed_qsvc
from quantum.feature_maps import qiskit_available
from quantum.kernel import compute_quantum_kernel, kernel_diagnostics
from quantum.alignment import kernel_target_alignment

def _score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None

def _classical_cv(X, y, folds, random_state, pca_components=None):
    results = {}
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)

    for name, base_model in build_classical_models(random_state).items():
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

    for name, base_model in build_classical_models(random_state).items():
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

def _quantum_holdout(X, y, qubits, feature_map):
    if not qiskit_available():
        return {
            "available": False,
            "reason": "Qiskit is not installed or failed to import. Classical experiments still completed."
        }

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    started = time.perf_counter()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    # Quantum kernels scale quadratically. Keep the demonstration bounded.
    if len(X_train) > MAX_QUANTUM_TRAIN_SAMPLES:
        rng = np.random.default_rng(RANDOM_STATE)
        selected = rng.choice(len(X_train), MAX_QUANTUM_TRAIN_SAMPLES, replace=False)
        X_train, y_train = X_train[selected], y_train[selected]

    components = safe_pca_components(qubits, len(X_train), X_train.shape[1])
    prep = __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=components, random_state=RANDOM_STATE)),
    ])
    X_train_q = prep.fit_transform(X_train)
    X_test_q = prep.transform(X_test)

    train_kernel, test_kernel, circuit = compute_quantum_kernel(
        X_train_q, X_test_q,
        {"feature_map": feature_map, "num_qubits": components}
    )

    model = build_precomputed_qsvc()
    model.fit(train_kernel, y_train)
    pred = model.predict(test_kernel)
    score = model.decision_function(test_kernel)

    metrics = evaluate_predictions(y_test, pred, score)
    alignment = kernel_target_alignment(train_kernel, y_train)

    circuit_text = None
    circuit_reason = None
    try:
        circuit_text = str(circuit.draw(output="text"))
        if len(circuit_text) > 50000:
            circuit_reason = f"Circuit text rendering exceeded max length (50000 chars)."
            circuit_text = None
    except Exception as e:
        circuit_reason = f"Circuit text rendering failed: {type(e).__name__}: {e}"
        circuit_text = None

    return {
        "available": True,
        "configuration": {
            "backend": "Qiskit fidelity quantum kernel simulator",
            "feature_map": feature_map,
            "num_qubits": components,
            "validation": "Stratified holdout (quantum demonstration)",
            "comparison_status": "Contextual only; not directly comparable to CV means",
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
        },
        "metrics": metrics,
        "timing_seconds": time.perf_counter() - started,
        "kernel_diagnostics": kernel_diagnostics(train_kernel),
        "kernel_target_alignment": alignment,
        "confusion_matrix": build_confusion(y_test, pred),
        "kernel_preview": train_kernel[:30, :30].tolist(),
        "circuit": circuit_text,
        "circuit_reason": circuit_reason,
    }


def _quantum_skipped(reason):
    return {
        "available": False,
        "status": "not_run",
        "reason": reason,
    }

def run_experiment(df, target_column=None, config=None):
    config = config or {}
    df = clean_dataframe(df)
    target_column = target_column or infer_target_column(df)

    if not target_column:
        raise ValueError("No target column found. Select a binary target column before running an experiment.")

    demo_schema = validate_schema(
        df, DEMO_FEATURE_NAMES, target_column=target_column
    )
    feature_names = (
        DEMO_FEATURE_NAMES
        if not demo_schema["missing_features"]
        else numeric_feature_columns(df, target_column)
    )
    validation = validate_dataset(
        df, target_column=target_column, required_features=feature_names
    )
    if not validation["valid"]:
        raise ValueError("; ".join(validation["errors"]))
    if df[target_column].nunique(dropna=True) != 2:
        raise ValueError("The selected target must contain exactly two classes.")

    if not feature_names:
        raise ValueError("No numeric predictive features were found.")

    X = align_features(df, feature_names)
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
        quantum = _quantum_holdout(
            X.to_numpy(),
            y.to_numpy(),
            int(config.get("quantum_qubits", DEFAULT_QUANTUM_QUBITS)),
            config.get("feature_map", "zz"),
        )
    else:
        quantum = _quantum_skipped(
            "Quantum simulation was not run. Enable 'Run experimental quantum kernel' "
            "to execute the optional, slower Qiskit workflow."
        )

    return {
        "metadata": {
            "experiment_version": "2.0.0",
            "runtime_seconds": time.perf_counter() - started,
            "target_column": target_column,
            "target_mapping": mapping,
            "platform_scope": "Breast-cancer tabular MVP; dataset profiles can be extended",
        },
        "dataset": {
            "rows": int(len(df)),
            "features": int(len(feature_names)),
            "feature_names": feature_names,
            "class_distribution": {str(k): int(v) for k, v in df[target_column].value_counts().to_dict().items()},
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
