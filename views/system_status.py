"""System and artifact status view."""

import importlib.util
import platform
import sys
from pathlib import Path

import streamlit as st

from core.data import load_demo_dataset
from components.theme import panel, section_header, status_badge


ROOT = Path(__file__).resolve().parents[1]


def _available(module_name):
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def render():
    section_header("System status", "Runtime and artifact checks for the research prototype.")

    demo_path = ROOT / "data" / "demo" / "qurenova_demo_biomedical.csv"
    try:
        demo = load_demo_dataset()
        demo_status = f"Ready ({len(demo)} rows × {len(demo.columns)} columns)"
    except Exception as exc:
        demo_status = f"Unavailable: {exc}"

    artifact_dirs = [ROOT / "results" / "models", ROOT / "models" / "trained"]
    artifact_files = []
    for directory in artifact_dirs:
        if directory.exists():
            artifact_files.extend(
                path.name
                for path in directory.glob("*.joblib")
                if path.is_file()
            )

    with panel("Core runtime"):
        st.table(
            {
                "Check": ["Python", "Demo dataset", "Persisted models"],
                "Status": [
                    platform.python_version(),
                    demo_status if demo_path.exists() else "Missing CSV",
                    f"{len(artifact_files)} artifact(s) available"
                    if artifact_files
                    else "None — prediction will train a fallback",
                ],
            }
        )
        status_badge("RUNTIME CHECK COMPLETE", "ok")

    optional = {
        "XGBoost": "xgboost",
        "SHAP": "shap",
        "Qiskit": "qiskit",
        "Qiskit Machine Learning": "qiskit_machine_learning",
    }
    with panel("Optional integrations"):
        st.table(
            {
                "Integration": list(optional),
                "Status": [
                    "Available" if _available(module) else "Not installed (graceful fallback)"
                    for module in optional.values()
                ],
            }
        )
    st.caption(f"Interpreter: {sys.executable}")
    if artifact_files:
        st.write("Model artifacts:", ", ".join(sorted(artifact_files)))
