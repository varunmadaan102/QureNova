"""Central model registry.

The registry keeps model construction in one place while allowing the rest of
QureNova to distinguish baseline, experimental and future tuned candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

from .classical import build_classical_models


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    stage: str
    factory: Callable[[int], Any]
    enabled: bool = True
    description: str = ""


_REGISTRY = (
    ModelSpec(
        "Logistic Regression", "classical", "baseline", lambda seed: build_classical_models(seed)["Logistic Regression"],
        description="Regularized linear baseline for calibrated binary classification.",
    ),
    ModelSpec(
        "SVM (RBF)", "classical", "baseline", lambda seed: build_classical_models(seed)["SVM (RBF)"],
        description="Balanced RBF SVM with probability calibration.",
    ),
    ModelSpec(
        "XGBoost", "classical", "experimental", lambda seed: build_classical_models(seed).get("XGBoost"),
        description="Optional gradient-boosted tree baseline when XGBoost is installed.",
    ),
)


def list_model_specs(*, enabled_only: bool = True) -> list[ModelSpec]:
    return [spec for spec in _REGISTRY if (spec.enabled or not enabled_only)]


def build_registered_models(random_state: int = 42) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for spec in list_model_specs():
        model = spec.factory(random_state)
        if model is not None:
            models[spec.name] = model
    return models


def get_model_spec(name: str) -> ModelSpec:
    for spec in _REGISTRY:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown model: {name}")
