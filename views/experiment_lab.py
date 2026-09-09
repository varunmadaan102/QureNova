import streamlit as st
from components.theme import dataframe, disclaimer, panel, section_header, status_badge
from services.experiment_service import run_experiment
from core.evaluation import model_result_table

def render():
    section_header("Experiment lab", "Run the configured classical and quantum comparison workflow.")
    df = st.session_state.get("dataset")
    target = st.session_state.get("target_column")

    if df is None or target is None:
        st.info("Load a dataset first in Data Workspace.")
        return

    c1, c2, c3 = st.columns(3)
    folds = c1.selectbox("CV folds", [3, 5], index=1)
    qubits = c2.selectbox("Quantum PCA dimensions / qubits", [2, 4, 6], index=1)
    fmap = c3.selectbox("Feature map", ["zz", "z"])

    run_quantum = st.checkbox(
        "Run experimental quantum kernel",
        value=False,
        help="Quantum simulation is optional and can take significantly longer than the classical baselines.",
    )
    st.caption(
        "Classical models use stratified cross-validation. Quantum simulation is "
        "opt-in because circuit-fidelity evaluation is computationally expensive."
    )

    if st.button("Run Experiment", type="primary"):
        with st.spinner("Running classical and quantum experiments..."):
            try:
                result = run_experiment(df, target, {
                    "cv_folds": folds,
                    "quantum_qubits": qubits,
                    "feature_map": fmap,
                    "run_quantum": run_quantum,
                })
                st.session_state["experiment_result"] = result
                status_badge("EXPERIMENT COMPLETED", "ok")
            except Exception as exc:
                st.error(f"Experiment failed: {exc}")
                return

    result = st.session_state.get("experiment_result")
    if result:
        with panel("Classical results"):
            dataframe(model_result_table(result["classical_results"]))

        q = result["quantum_results"]
        with panel("Quantum result"):
            if q.get("available"):
                st.json({
                    "configuration": q["configuration"],
                    "metrics": q["metrics"],
                    "kernel_target_alignment": q["kernel_target_alignment"],
                    "timing_seconds": q["timing_seconds"],
                })
            elif q.get("status") == "not_run":
                st.info(q["reason"])
            else:
                st.warning(q.get("reason"), icon=":material/warning:")
        disclaimer("Experiment outputs are research measurements, not evidence of clinical utility or quantum advantage.")
