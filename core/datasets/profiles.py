from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

@dataclass(frozen=True)
class TargetDefinition:
    column: str
    task_type: str
    positive_class: Any
    negative_class: Any
    label_mapping: Dict[Any, int]
    class_names: Dict[int, str]
    source: str

@dataclass(frozen=True)
class DatasetProfile:
    name: str
    source: str
    task_type: str
    target: TargetDefinition
    expected_samples: Optional[int] = None
    expected_features: Optional[int] = None

WDBC_TARGET = TargetDefinition(
    column="diagnosis",
    task_type="binary_classification",
    positive_class="M",
    negative_class="B",
    label_mapping={"B": 0, "M": 1},
    class_names={0: "Benign", 1: "Malignant"},
    source="UCI Breast Cancer Wisconsin (Diagnostic) dataset definition",
)

DATASET_PROFILES = {
    "breast_cancer_wisconsin_diagnostic": DatasetProfile(
        name="Breast Cancer Wisconsin (Diagnostic)",
        source="UCI Machine Learning Repository; bundled public WDBC representation",
        task_type="binary_classification",
        target=WDBC_TARGET,
        expected_samples=569,
        expected_features=30,
    )
}

def get_dataset_profile(name: str) -> DatasetProfile:
    try:
        return DATASET_PROFILES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown QureNova dataset profile: {name}") from exc

def resolve_target_definition(profile_name: str) -> TargetDefinition:
    return get_dataset_profile(profile_name).target

def profile_dict(profile: DatasetProfile) -> Dict[str, Any]:
    return asdict(profile)
