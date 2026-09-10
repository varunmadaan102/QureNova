import json

from core.history import (
    build_experiment_record,
    load_experiment_records,
    save_experiment_record,
)


def test_experiment_history_round_trip(tmp_path):
    result = {
        "metadata": {
            "runtime_seconds": 1.25,
            "target_column": "diagnosis",
        },
        "dataset": {
            "rows": 20,
            "features": 3,
            "feature_names": ["a", "b", "c"],
            "class_distribution": {"0": 10, "1": 10},
        },
        "preprocessing": {"pipeline": "fold-fitted", "pca_components": 2},
        "classical_results": {
            "Logistic Regression": {
                "summary": {"f1": {"mean": 0.8, "std": 0.1}},
                "timing_seconds": 0.4,
                "validation": "Stratified 3-fold CV",
            }
        },
        "quantum_results": {
            "available": False,
            "reason": "disabled",
        },
        "artifacts": {"reference_df": object()},
    }

    record = build_experiment_record(
        result,
        experiment_id="exp-test",
        timestamp="2026-01-01T00:00:00+00:00",
    )
    path = save_experiment_record(record, tmp_path)

    assert path.name == "exp-test.json"
    assert json.loads(path.read_text())["dataset"]["rows"] == 20
    loaded = load_experiment_records(tmp_path)
    assert loaded[0]["experiment_id"] == "exp-test"
    assert loaded[0]["classical_models"]["Logistic Regression"]["summary"]["f1"]["mean"] == 0.8
