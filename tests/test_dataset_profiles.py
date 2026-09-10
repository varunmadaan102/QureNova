import pandas as pd
import pytest
from core.data import encode_binary_target
from core.datasets import resolve_target_definition

def test_wdbc_profile_is_explicit():
    target = resolve_target_definition("breast_cancer_wisconsin_diagnostic")
    assert target.column == "diagnosis"
    assert target.label_mapping == {"B": 0, "M": 1}
    assert target.class_names == {0: "Benign", 1: "Malignant"}

def test_unknown_numeric_codes_are_rejected():
    target = resolve_target_definition("breast_cancer_wisconsin_diagnostic")
    with pytest.raises(ValueError, match="do not exactly match"):
        encode_binary_target(pd.Series([1, 2]), target.label_mapping)
