import pytest

from core.evaluation import evaluate_predictions
from quantum.alignment import kernel_target_alignment


def test_evaluation_reports_sensitivity_specificity_and_balanced_accuracy():
    result = evaluate_predictions(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        [0.1, 0.8, 0.7, 0.9],
    )

    assert result["sensitivity"] == pytest.approx(1.0)
    assert result["specificity"] == pytest.approx(0.5)
    assert result["balanced_accuracy"] == pytest.approx(0.75)


def test_quantum_alignment_is_invariant_to_binary_encoding():
    kernel = [[1.0, 0.2], [0.2, 1.0]]

    assert kernel_target_alignment(kernel, [0, 1]) == pytest.approx(
        kernel_target_alignment(kernel, [-1, 1])
    )
