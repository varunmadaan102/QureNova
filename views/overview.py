import streamlit as st
from components.theme import (
    disclaimer, hero, metric_strip, panel, section_header, status_badge,
    page_header, empty_state, stat_row, section_divider
)
from components.pipeline import pipeline
from config.settings import CLASSICAL_MODELS

def render():
    hero(
        "Make biomedical ML legible.",
        "A measurement-first workspace for comparing classical baselines and bounded quantum-kernel experiments—designed for researchers, students, and jury walkthroughs.",
        chip="MVP READY · RESEARCH-ONLY · NON-DIAGNOSTIC",
    )

    # System Status Section
    section_header("System Status", "Current state of QureNova and loaded artifacts")
    
    dataset = st.session_state.get("dataset")
    experiment = st.session_state.get("experiment_result")
    
    if dataset is None and experiment is None:
        empty_state(
            "📊",
            "No Data Loaded",
            "Upload a biomedical dataset to begin analysis. Visit Data Workspace to get started.",
            "→ Go to Data Workspace"
        )
        return
    
    # Dataset Status
    with panel("Dataset"):
        if dataset is not None:
            cols = st.columns(3)
            with cols[0]:
                st.metric("Samples", len(dataset))
            with cols[1]:
                st.metric("Features", len(dataset.columns) - 1)
            with cols[2]:
                target_col = dataset.columns[-1]
                unique_targets = dataset[target_col].nunique() if target_col in dataset.columns else 0
                st.metric("Target Classes", unique_targets)
        else:
            st.write("No dataset loaded")
    
    # Workflow Information
    with panel("Research Workflow"):
        st.markdown("""
        **QureNova** compares classical machine-learning baselines against a bounded quantum-kernel workflow:
        
        - **Classical Models:** Logistic Regression, Calibrated SVM, XGBoost
        - **Quantum Workflow:** Fidelity quantum-kernel QSVC (simulator-backed)
        - **Evaluation:** Stratified cross-validation for classical, stratified holdout for quantum
        - **Boundary:** Research prototype—no quantum advantage claims, no clinical deployment
        """)
        status_badge("REPRODUCIBLE WORKFLOW", "info")
    
    section_divider()

    # Pipeline Visualization
    section_header("Analysis Pipeline", "Standard workflow from data to interpretation")
    pipeline([
        "CSV Data",
        "Validation",
        "Shared Preprocessing",
        "Classical ML",
        "Quantum Kernel",
        "Benchmark",
        "Patient Analysis",
    ])

    section_divider()

    # Key Metrics
    section_header("Platform Configuration", "Available models and validation strategies")
    cols = st.columns(3)
    with cols[0]:
        st.metric("Classical Models", len(CLASSICAL_MODELS))
    with cols[1]:
        st.metric("Quantum Workflow", "Kernel + QSVC")
    with cols[2]:
        st.metric("Validation", "Stratified CV")

    section_divider()

    # Next Steps
    section_header("Next Steps", "Continue your analysis workflow")
    
    if experiment is None:
        with panel("Ready to Begin"):
            st.markdown("""
            1. **Data Workspace** — Upload or inspect your dataset
            2. **Experiment Lab** — Configure and run classical + quantum experiments
            3. **Benchmark** — Compare model performance side-by-side
            4. **Patient Analysis** — Get individual predictions and explanations
            """)
    else:
        with panel("Continue Analysis"):
            st.markdown("""
            - **Benchmark** — Review latest experiment results
            - **Patient Analysis** — Analyze individual predictions
            - **Quantum Analysis** — Inspect circuit and kernel details
            - **Explainability** — Understand feature contributions
            """)

    section_divider()

    # Boundaries and Disclaimers
    section_header("Research Boundaries", "Important limitations and use cases")
    
    with panel():
        st.markdown("""
        **This is a research prototype:**
        - Results are research estimates, not clinical diagnoses
        - No quantum advantage claims or guarantees
        - Not clinically validated or approved for clinical use
        - Designed for demonstration, education, and research exploration
        
        **Read Guide & Methodology for:**
        - CSV data contract and feature requirements
        - Preprocessing and leakage-control details
        - Model configuration and training protocols
        - Troubleshooting and FAQ
        """)
    
    disclaimer("Research prototype only. Results are not clinical diagnoses and are not clinically validated.")

