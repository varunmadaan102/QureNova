"""Evaluation protocol definitions.

Keeping protocol metadata explicit prevents accidentally comparing metrics from
incompatible validation designs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationProtocol:
    name: str
    kind: str
    folds: int | None = None
    outer_test_size: float | None = None
    seed: int = 42
    comparable: bool = True
    description: str = ""


CLASSICAL_CV = EvaluationProtocol(
    name="stratified_cv",
    kind="cross_validation",
    folds=5,
    comparable=True,
    description="Stratified K-fold CV with fold-fitted preprocessing.",
)

QUANTUM_HOLDOUT = EvaluationProtocol(
    name="quantum_holdout",
    kind="holdout",
    outer_test_size=0.25,
    comparable=False,
    description="Bounded stratified holdout used for interactive quantum-kernel experiments.",
)

COMMON_FINAL_TEST = EvaluationProtocol(
    name="common_final_test",
    kind="fixed_external_test",
    comparable=True,
    description="One untouched final test set shared by classical and quantum candidates.",
)


def protocol_summary(protocol: EvaluationProtocol) -> dict:
    return {
        "name": protocol.name,
        "kind": protocol.kind,
        "folds": protocol.folds,
        "outer_test_size": protocol.outer_test_size,
        "seed": protocol.seed,
        "comparable": protocol.comparable,
        "description": protocol.description,
    }
