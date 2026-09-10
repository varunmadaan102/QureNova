import pandas as pd
from core.data import validate_schema


def validate_dataset(df, target_column=None, required_features=None):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        return {
            "valid": False,
            "missing_features": [],
            "extra_features": [],
            "invalid_numeric_features": [],
            "duplicate_columns": [],
            "reordered": False,
            "warnings": ["Dataset is empty."],
            "errors": ["No rows found."],
            "rows": 0,
            "columns": len(df.columns),
            "missing_cells": 0,
        }

    frame = df
    required_features = required_features or []
    schema = validate_schema(
        frame,
        required_features=required_features,
        target_column=target_column,
        numeric_features=required_features,
    )
    warnings, errors = list(schema["warnings"]), list(schema["errors"])

    duplicate_count = int(frame.duplicated().sum())
    if duplicate_count:
        warnings.append(f"{duplicate_count} duplicate rows detected.")

    return {
        "valid": len(errors) == 0,
        "missing_features": schema["missing_features"],
        "extra_features": schema["extra_features"],
        "invalid_numeric_features": schema["invalid_numeric_features"],
        "duplicate_columns": schema["duplicate_columns"],
        "warnings": warnings,
        "errors": errors,
        "rows": len(frame),
        "columns": len(frame.columns),
        "missing_cells": int(frame.isna().sum().sum()),
    }
