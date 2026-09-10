"""Application-facing experiment entry point.

This is deliberately a thin boundary around the current engine. The existing
`services.experiment_service.run_experiment` remains the implementation until
the benchmark protocol is upgraded to a common final test set.
"""

from __future__ import annotations

from uuid import uuid4
from datetime import datetime, timezone

from artifacts.store import ArtifactStore
from domain.contracts import ExperimentArtifact, ExperimentSpec
from services.experiment_service import run_experiment


def new_experiment_id() -> str:
    return f"exp-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"


def execute(df, target_column: str, config: dict | None = None, *, persist: bool = True):
    config = dict(config or {})
    experiment_id = config.pop("experiment_id", None) or new_experiment_id()
    spec = ExperimentSpec(
        experiment_id=experiment_id,
        dataset_profile=config.get("dataset_profile"),
        target_column=target_column,
        cv_folds=int(config.get("cv_folds", 5)),
        seeds=tuple(config.get("seeds", [42])),
        pca_components=int(config.get("pca_components", 4)),
        run_quantum=bool(config.get("run_quantum", False)),
        quantum_qubits=int(config.get("quantum_qubits", 4)),
        feature_map=str(config.get("feature_map", "zz")),
        quantum_backend=str(config.get("quantum_backend", "auto")),
        tuning_enabled=bool(config.get("tuning_enabled", False)),
    )
    result = run_experiment(df, target_column=target_column, config=config)
    result.setdefault("metadata", {})["experiment_id"] = experiment_id
    artifact = ExperimentArtifact(experiment_id=experiment_id, spec=spec, result=result)
    if persist:
        ArtifactStore().save(artifact.to_dict())
    return result, experiment_id
