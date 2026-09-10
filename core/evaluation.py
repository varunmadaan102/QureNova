import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
    "f1",
    "roc_auc",
)

def evaluate_predictions(y_true, y_pred, y_score=None):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) else 0.0
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        result["roc_auc"] = float(roc_auc_score(y_true, y_score if y_score is not None else y_pred))
    except ValueError:
        result["roc_auc"] = None
    return result

def summarize_fold_metrics(folds):
    frame = pd.DataFrame(folds)
    summary = {}
    for metric in METRICS:
        values = pd.to_numeric(frame.get(metric), errors="coerce").dropna()
        summary[metric] = {
            "mean": float(values.mean()) if len(values) else None,
            "std": float(values.std(ddof=0)) if len(values) else None,
        }
    return summary

def model_result_table(results):
    rows = []
    for name, result in results.items():
        row = {"Model": name}
        for metric in METRICS:
            m = result.get("summary", {}).get(metric, {})
            mean, std = m.get("mean"), m.get("std")
            row[metric.upper()] = None if mean is None else round(mean, 4)
            row[f"{metric.upper()} STD"] = None if std is None else round(std, 4)
        row["Time (s)"] = round(float(result.get("timing_seconds", 0)), 3)
        row["Validation"] = result.get("validation", "Stratified CV")
        row["Comparison"] = "CV baseline"
        rows.append(row)
    return pd.DataFrame(rows)

def build_confusion(y_true, y_pred):
    return confusion_matrix(y_true, y_pred).tolist()
