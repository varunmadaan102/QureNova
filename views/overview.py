import streamlit as st
from components.theme import disclaimer, metric_strip, panel, section_header, status_badge
from components.pipeline import pipeline
from config.settings import CLASSICAL_MODELS

def render():
    section_header(
        "Overview",
        "A guided, non-diagnostic research console for tabular biomedical classification.",
    )
    with panel():
        st.markdown("### A measurement-first research console")
        st.write(
            "QureAI compares classical baselines and bounded quantum-kernel experiments "
            "without assuming quantum advantage. The checked-in example uses a synthetic "
            "Wisconsin-style 30-feature schema; read Guide & Methodology for the CSV "
            "contract, normalized headers, and safe interpretation boundaries."
        )
        status_badge("REPRODUCIBLE WORKFLOW", "info")

    with panel("SIH 26139 alignment"):
        st.markdown(
            """
            **Evidence-backed MVP:** breast-cancer tabular early-detection
            classification with classical baselines, leakage-safe preprocessing,
            an opt-in fidelity-quantum-kernel QSVC, explainability, and unified
            evaluation.

            **Extensible platform:** the validation, preprocessing, experiment,
            artifact, and quantum-backend boundaries are separated so future
            cardiovascular, neurological, imaging, or genomics profiles can be
            added without changing the core workflow.

            **Evidence boundary:** this prototype reports research metrics; it
            does not claim quantum advantage, clinical accuracy, or deployment
            readiness without external validation and governance.
            """
        )

    pipeline([
        "CSV Data",
        "Validation",
        "Shared Preprocessing",
        "Classical ML",
        "Quantum Kernel",
        "Benchmark",
        "Patient Analysis",
    ])

    metric_strip([
        ("Classical models", str(len(CLASSICAL_MODELS)), "Configured classical baselines"),
        ("Quantum workflow", "Kernel + QSVC", "Available when optional quantum dependencies are installed"),
        ("Validation", "Stratified CV", "Configured validation strategy"),
    ])

    section_header("Operating sequence", "Keep data, artifacts, and interpretation connected.")
    with panel():
        st.markdown("""
1. Upload or generate a dataset in **Data Workspace**.
2. Choose a binary target (or use inspection-only mode) and inspect validation.
3. Run the experiment in **Experiment Lab**.
4. Compare measured outputs in **Benchmark**.
5. Upload compatible patient rows for individual predictions.
6. Review feature-level explanations.
7. Use **Guide & Methodology** for input examples, model terms, troubleshooting,
   and the non-diagnostic research boundary.
""")

    disclaimer("Research prototype only. Results are not clinical diagnoses and are not clinically validated.")
