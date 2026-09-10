"""Domain-level contracts for QureNova.

These structures contain no Streamlit or persistence concerns. They define the
stable vocabulary shared by the UI, services, model runners and future API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class DatasetRef:
    profile: str | None
    name: str
    source: str | None
    rows: int
    features: int
    target_column: str
    target_mapping: dict[str, int] = field(default_factory=dict)
    content_hash: str | None = None


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    dataset_profile: str | None
    target_column: str
    cv_folds: int = 5
    seeds: tuple[int, ...] = (42,)
    pca_components: int = 4
    run_quantum: bool = False
    quantum_qubits: int = 4
    feature_map: str = "zz"
    quantum_backend: str = "auto"
    tuning_enabled: bool = False


@dataclass
class ExperimentArtifact:
    """Canonical completed-experiment envelope.

    `result` contains the backwards-compatible engine output. Metadata and
    provenance live beside it so future API/database layers can reuse the same
    object without depending on Streamlit session state.
    """

    experiment_id: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    spec: ExperimentSpec | None = None
    dataset: DatasetRef | None = None
    result: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "spec": self.spec.__dict__ if self.spec else None,
            "dataset": self.dataset.__dict__ if self.dataset else None,
            "result": self.result,
            "warnings": list(self.warnings),
        }
