"""Persistence helpers for compact, comparable experiment records."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


DEFAULT_HISTORY_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "experiments"
)


def _metric_summary(summary):
    return {
        name: {
            "mean": values.get("mean"),
            "std": values.get("std"),
        }
        for name, values in summary.items()
    }


def build_experiment_record(result, experiment_id=None, timestamp=None):
    """Create a JSON-safe summary without persisting training data."""
    if not isinstance(result, dict):
        raise TypeError("Experiment result must be a dictionary.")

    metadata = result.get("metadata", {})
    dataset = result.get("dataset", {})
    preprocessing = result.get("preprocessing", {})
    classical = result.get("classical_results", {})
    quantum = result.get("quantum_results", {})
    recorded_at = timestamp or datetime.now(timezone.utc).isoformat()
    record_id = experiment_id or (
        f"exp-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    )

    return {
        "experiment_id": record_id,
        "timestamp": recorded_at,
        "runtime_seconds": metadata.get("runtime_seconds"),
        "dataset": {
            "rows": dataset.get("rows"),
            "features": dataset.get("features"),
            "feature_names": list(dataset.get("feature_names", [])),
            "class_distribution": dataset.get("class_distribution", {}),
            "target_column": metadata.get("target_column"),
        },
        "preprocessing": {
            "pipeline": preprocessing.get("pipeline"),
            "pca_components": preprocessing.get("pca_components"),
        },
        "classical_models": {
            name: {
                "summary": _metric_summary(values.get("summary", {})),
                "timing_seconds": values.get("timing_seconds"),
                "validation": values.get("validation"),
            }
            for name, values in classical.items()
        },
        "quantum": {
            "available": bool(quantum.get("available")),
            "reason": quantum.get("reason"),
            "configuration": quantum.get("configuration", {}),
            "metrics": quantum.get("metrics", {}),
            "timing_seconds": quantum.get("timing_seconds"),
        },
    }


def save_experiment_record(record, directory=None):
    """Persist one record and return its path."""
    if not isinstance(record, dict) or not record.get("experiment_id"):
        raise ValueError("A record with an experiment_id is required.")
    history_dir = Path(directory or DEFAULT_HISTORY_DIR)
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / f"{record['experiment_id']}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return path


def load_experiment_records(directory=None):
    """Load valid experiment records newest first."""
    history_dir = Path(directory or DEFAULT_HISTORY_DIR)
    if not history_dir.exists():
        return []
    records = []
    for path in sorted(history_dir.glob("exp-*.json"), reverse=True):
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        if not isinstance(record, dict) or not record.get("experiment_id"):
            raise ValueError(f"Invalid experiment record: {path.name}")
        records.append(record)
    return records
