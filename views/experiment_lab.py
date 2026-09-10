import streamlit as st
from components.theme import (
    dataframe, disclaimer, panel, section_header, status_badge,
    page_header, empty_state, loading_state, error_state, section_divider
)
from experiments.orchestrator import execute
from core.evaluation import model_result_table
from quantum.feature_maps import qiskit_available
from core.history import build_experiment_record, save_experiment_record

def render():
    page_header(
        "Experiment Lab",
        "Configure and run classical baseline and quantum kernel experiments",
        "Experiment configuration · execution · result comparison"
    )

    df = st.session_state.get("dataset")
    target = st.session_state.get("target_column")

    if df is None or target is None:
        empty_state(
            "⚙️",
            "No Dataset Loaded",
            "Load a dataset and select a binary target in Data Workspace to configure experiments.",
            "→ Go to Data Workspace"
        )
        return

    # Experiment Configuration
    section_header("Experiment Configuration", "Set parameters for classical and quantum models")
    
    with panel("Model Parameters"):
        col1, col2, col3 = st.columns(3)
        with col1:
            folds = st.selectbox(
                "CV Folds",
                [3, 5],
                index=1,
                help="Number of folds for stratified cross-validation (classical models only)"
            )
        with col2:
            qubits = st.selectbox(
                "Quantum PCA Dimensions",
                [2, 4, 6],
                index=1,
                help="Number of qubits for quantum kernel computation"
            )
        with col3:
            fmap = st.selectbox(
                "Feature Map",
                ["zz", "z"],
                help="Quantum feature map type"
            )

    # Quantum Configuration
    section_header("Quantum Workflow", "Optional quantum kernel QSVC")
    
    if qiskit_available():
        run_quantum = st.checkbox(
            "Enable quantum kernel computation",
            value=False,
            help="Quantum simulation can take 30-60 seconds. Classical analysis remains available.",
        )
        st.caption("✓ Quantum dependencies are available in this environment")
    else:
        run_quantum = False
        st.warning(
            "Quantum simulation is unavailable in this environment. "
            "Install the packages in requirements.txt to enable quantum workflows."
        )

    st.caption(
        "**Evaluation Protocol:** Classical models use stratified cross-validation. "
        "Quantum simulation uses stratified holdout (train/test split) due to computational constraints."
    )

    section_divider()

    # Run Button
    run_in_progress = bool(st.session_state.get("run_in_progress", False))
    
    if st.button("Run Experiment", type="primary", disabled=run_in_progress, use_container_width=True):
        st.session_state["run_in_progress"] = True
        try:
            progress_container = st.container()
            with progress_container:
                if run_quantum:
                    st.info("🔬 Quantum kernel computation in progress (30-60 seconds)...")
                else:
                    st.info("⏳ Running classical baselines...")
                
                result, experiment_id = execute(df, target, {
                    "dataset_profile": st.session_state.get("dataset_profile"),
                    "cv_folds": folds,
                    "quantum_qubits": qubits,
                    "feature_map": fmap,
                    "run_quantum": run_quantum,
                }, persist=True)
                st.session_state["experiment_result"] = result
                st.session_state["last_experiment_id"] = experiment_id
                try:
                    record = build_experiment_record(result, experiment_id=experiment_id)
                    save_experiment_record(record)
                except (OSError, TypeError, ValueError) as exc:
                    st.error(f"Experiment completed, but history could not be saved: {exc}")
                
                st.success("✓ Experiment completed successfully")
                
        except Exception as exc:
            error_state(
                "Experiment Failed",
                "The experiment could not be completed. Verify dataset validation and try again."
            )
            with st.expander("Technical details"):
                st.code(str(exc), language="text")
            return
        finally:
            st.session_state["run_in_progress"] = False

    section_divider()

    # Results Display
    result = st.session_state.get("experiment_result")
    if result:
        section_header("Experiment Results", "Classical baseline and quantum comparison")
        
        # Classical Results
        with panel("Classical Baselines"):
            st.caption(f"Trained with {folds}-fold stratified cross-validation")
            dataframe(model_result_table(result["classical_results"]))

        # Quantum Results
        q = result["quantum_results"]
        with panel("Quantum Kernel QSVC"):
            if q.get("available"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Configuration", f"{qubits} qubits · {fmap} map")
                with col2:
                    st.metric("Runtime (sec)", f"{q['timing_seconds']:.2f}")
                
                st.subheader("Performance Metrics")
                st.json({
                    "metrics": q["metrics"],
                    "kernel_target_alignment": q["kernel_target_alignment"],
                })
            elif q.get("status") == "not_run":
                st.info(q["reason"])
            else:
                st.warning(q.get("reason", "Quantum unavailable"))
        
        section_divider()
        disclaimer("Experiment outputs are research measurements. Results do not claim quantum advantage or clinical utility.")

