import streamlit as st
from components.theme import dataframe, disclaimer, panel, section_header, status_badge
from services.experiment_service import run_experiment
from core.evaluation import model_result_table
from quantum.feature_maps import qiskit_available
from core.history import build_experiment_record, save_experiment_record

def render():
    section_header("Experiment lab", "Run the configured classical and quantum comparison workflow.")
    df = st.session_state.get("dataset")
    target = st.session_state.get("target_column")

    if df is None or target is None:
        st.info("Load a dataset first in Data Workspace.")
        return

    with st.container():
        folds = st.selectbox("CV folds", [3, 5], index=1)
        qubits = st.selectbox("Quantum PCA dimensions / qubits", [2, 4, 6], index=1)
        fmap = st.selectbox("Feature map", ["zz", "z"])

    if qiskit_available():
        run_quantum = st.checkbox(
            "Run experimental quantum kernel",
            value=False,
            help="Quantum simulation can take significantly longer than the classical baselines.",
        )
    else:
        run_quantum = False
        st.info(
            "Quantum simulation is unavailable in this environment. Install the "
            "packages listed in requirements.txt, then restart the app."
        )
    st.caption(
        "Classical models use stratified cross-validation. Quantum simulation is "
        "opt-in because circuit-fidelity evaluation is computationally expensive."
    )

    run_in_progress = bool(st.session_state.get("run_in_progress", False))

    if st.button("Run Experiment", type="primary", disabled=run_in_progress):
        st.session_state["run_in_progress"] = True
        try:
            with st.spinner("Running classical and quantum experiments..."):
                if run_quantum:
                    st.info("Quantum kernel computation takes 30-60 seconds. The app is working - please wait.")
                result = run_experiment(df, target, {
                    "cv_folds": folds,
                    "quantum_qubits": qubits,
                    "feature_map": fmap,
                    "run_quantum": run_quantum,
                })
                st.session_state["experiment_result"] = result
                try:
                    record = build_experiment_record(result)
                    save_experiment_record(record)
                    st.session_state["last_experiment_id"] = record["experiment_id"]
                except (OSError, TypeError, ValueError) as exc:
                    st.error(f"Experiment completed, but its history could not be saved: {exc}")
                status_badge("EXPERIMENT COMPLETED", "ok")
        except Exception as exc:
            st.error("The experiment could not be completed. Review the dataset validation and try again.")
            with st.expander("Technical details"):
                st.code(str(exc), language="text")
            return
        finally:
            st.session_state["run_in_progress"] = False

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
