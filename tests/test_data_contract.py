import pandas as pd
import pytest

from core.data import align_features, encode_binary_target, validate_schema
from core.validation import validate_dataset


FEATURES = ["mean radius", "mean texture", "mean perimeter"]


def test_schema_normalizes_headers_and_aligns_feature_order():
    frame = pd.DataFrame(
        {
            "mean_texture": [2.0, 3.0],
            "mean_radius": [1.0, 1.5],
            "mean_perimeter": [4.0, 5.0],
            "sample_id": ["A", "B"],
        }
    )

    report = validate_schema(frame, FEATURES, allow_extra=True)
    aligned = align_features(frame, FEATURES)

    assert report["valid"]
    assert report["reordered"]
    assert report["extra_features"] == ["sample_id"]
    assert aligned.columns.tolist() == FEATURES
    assert aligned.iloc[0].tolist() == [1.0, 2.0, 4.0]


def test_schema_accepts_common_uci_suffix_headers():
    frame = pd.DataFrame(
        {
            "radius_mean": [1.0],
            "texture_mean": [2.0],
            "perimeter_mean": [4.0],
        }
    )

    aligned = align_features(frame, FEATURES)

    assert aligned.columns.tolist() == FEATURES
    assert aligned.iloc[0].tolist() == [1.0, 2.0, 4.0]


def test_schema_rejects_duplicate_and_nonnumeric_features():
    frame = pd.DataFrame(
        [
            ["bad", 2.0, 4.0],
        ],
        columns=["mean_radius", "mean radius", "mean_perimeter"],
    )

    report = validate_dataset(frame, required_features=FEATURES)

    assert not report["valid"]
    assert report["duplicate_columns"]
    assert "mean radius" in report["invalid_numeric_features"]


def test_binary_target_rejects_missing_values():
    with pytest.raises(ValueError, match="missing values"):
        encode_binary_target(pd.Series(["benign", None, "malignant"]))


def test_numeric_medical_labels_fail_closed_without_mapping():
    with pytest.raises(ValueError, match="explicit label_mapping"):
        encode_binary_target(pd.Series([1, 2, 1, 2]))


def test_explicit_label_mapping_is_used():
    encoded, mapping = encode_binary_target(
        pd.Series(["benign", "malignant", "benign"]),
        {"benign": 0, "malignant": 1},
    )
    assert encoded.tolist() == [0, 1, 0]
    assert mapping == {"benign": 0, "malignant": 1}


def test_wdbc_semantic_labels_are_supported():
    encoded, mapping = encode_binary_target(pd.Series(["B", "M", "B"]))
    assert encoded.tolist() == [0, 1, 0]
    assert mapping == {"B": 0, "M": 1}
