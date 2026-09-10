"""Dataset catalog facade.

Profiles remain backward compatible under `core.datasets` while this module
provides the future root-level dataset registry expected by the platform.
"""

from core.datasets.profiles import DATASET_PROFILES, DatasetProfile, TargetDefinition, resolve_target_definition

__all__ = ["DATASET_PROFILES", "DatasetProfile", "TargetDefinition", "resolve_target_definition"]
