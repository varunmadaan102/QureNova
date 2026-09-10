import pandas as pd
import pytest

from core.data import load_demo_dataset
from services.prediction_service import predict_patients


def test_prediction_accepts_string_metadata_column():
    reference = load_demo_dataset()
    target = "diagnosis"
    patient = reference.drop(columns=[target]).head(1).copy()
    patient.insert(0, "sample_id", "SYN-0001")

    result, mapping, features, source = predict_patients(
        reference,
        target,
        patient,
        return_source=True,
    )

    assert len(result) == 1
    assert result["Model Agreement"].iloc[0] in {
        "HIGH AGREEMENT",
        "MODERATE AGREEMENT",
        "LOW AGREEMENT",
    }
    assert result["Risk Category"].iloc[0] in {"LOW", "MODERATE", "ELEVATED"}
    assert result["Prediction Reliability"].iloc[0] in {"HIGH", "MODERATE", "LOW"}
    assert len(mapping) == 2
    assert len(features) == 30
    assert source


def test_prediction_rejects_missing_feature():
    reference = load_demo_dataset()
    patient = reference.drop(columns=["diagnosis"]).head(1).drop(
        columns=["mean radius"]
    )

    with pytest.raises(ValueError, match="Missing required features"):
        predict_patients(reference, "diagnosis", patient)
