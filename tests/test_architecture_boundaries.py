from pathlib import Path

from artifacts.store import ArtifactStore
from evaluation.model_selection import utility_score
from security.privacy import find_direct_identifiers
from security.upload_policy import validate_upload_name


def test_artifact_store_round_trip(tmp_path):
    store = ArtifactStore(tmp_path)
    payload = {"experiment_id": "exp-test-001", "result": {"ok": True}}
    path = store.save(payload)
    assert path.exists()
    assert store.load("exp-test-001")["result"]["ok"] is True


def test_privacy_identifier_detection():
    assert "patient_name" in find_direct_identifiers(["patient_name", "radius"])
    assert "radius" not in find_direct_identifiers(["patient_name", "radius"])


def test_csv_upload_policy():
    validate_upload_name("dataset.csv")


def test_model_utility_score_is_bounded_for_valid_inputs():
    score = utility_score({
        "roc_auc": 1,
        "f1": 0.9,
        "sensitivity": 0.9,
        "specificity": 0.9,
        "stability": 0.8,
        "runtime_efficiency": 0.7,
    })
    assert 0 <= score <= 1
