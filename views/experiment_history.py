import pandas as pd
import streamlit as st

from components.theme import dataframe, disclaimer, panel, section_header
from core.history import load_experiment_records


def _classical_rows(record):
    rows = []
    for model, details in record.get("classical_models", {}).items():
        row = {
            "Experiment": record["experiment_id"],
            "Timestamp": record.get("timestamp", ""),
            "Model": model,
            "Validation": details.get("validation", ""),
            "Time (s)": details.get("timing_seconds"),
        }
        for metric, values in details.get("summary", {}).items():
            row[metric.upper()] = values.get("mean")
        rows.append(row)
    return rows


def render():
    section_header(
        "Experiment history",
        "Review persisted experiment summaries without storing patient rows or model artifacts.",
    )
    try:
        records = load_experiment_records()
    except (OSError, ValueError) as exc:
        st.error(f"Experiment history could not be loaded: {exc}")
        return

    if not records:
        st.info("No saved experiments yet. Run an experiment in Experiment Lab first.")
        return

    history = pd.DataFrame(
        [
            {
                "Experiment": record["experiment_id"],
                "Timestamp": record.get("timestamp", ""),
                "Rows": record.get("dataset", {}).get("rows"),
                "Features": record.get("dataset", {}).get("features"),
                "Runtime (s)": record.get("runtime_seconds"),
                "Quantum": "Available"
                if record.get("quantum", {}).get("available")
                else "Not run",
            }
            for record in records
        ]
    )
    dataframe(history)

    selected = st.multiselect(
        "Compare experiments",
        [record["experiment_id"] for record in records],
        default=[records[0]["experiment_id"]],
        max_selections=5,
    )
    selected_records = [
        record for record in records if record["experiment_id"] in selected
    ]
    if selected_records:
        with panel("Classical comparison"):
            comparison = pd.DataFrame(
                row
                for record in selected_records
                for row in _classical_rows(record)
            )
            dataframe(comparison)

        with panel("Experiment configuration"):
            configuration = pd.DataFrame(
                [
                    {
                        "Experiment": record["experiment_id"],
                        "Target": record.get("dataset", {}).get("target_column"),
                        "PCA components": record.get("preprocessing", {}).get(
                            "pca_components"
                        ),
                        "Quantum samples": record.get("quantum", {})
                        .get("configuration", {})
                        .get("train_samples"),
                        "Quantum feature map": record.get("quantum", {})
                        .get("configuration", {})
                        .get("feature_map"),
                    }
                    for record in selected_records
                ]
            )
            dataframe(configuration)

        disclaimer(
            "History contains experiment metadata and aggregate metrics only. "
            "It is not a clinical record and does not establish model validity."
        )
