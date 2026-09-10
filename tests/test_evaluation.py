import pytest

from core.evaluation import evaluate_predictions
from core.clinical import agreement_summary, risk_category
from core.reliability import assess_input_reliability
from quantum.alignment import kernel_target_alignment
import pandas as pd


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


def test_clinical_labels_are_explicit_and_distribution_check_is_conservative():
    assert risk_category(0.2) == "LOW"
    assert risk_category(0.5) == "MODERATE"
    assert risk_category(0.9) == "ELEVATED"
    assert agreement_summary([0, 0, 1]) == "MODERATE AGREEMENT"

    reference = pd.DataFrame({"a": [1.0, 1.1, 0.9], "b": [10.0, 10.2, 9.8]})
    result = assess_input_reliability(reference, pd.DataFrame({"a": [1.0], "b": [10.0]}))
    assert result["level"] == "HIGH"
