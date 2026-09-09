import streamlit as st
from components.theme import dataframe, disclaimer, panel, section_header
from services.explainability_service import explain_xgboost, explain_linear_model

def render():
    section_header("Explainability", "Clinician-first feature attribution with explicit research boundaries.")
    reference = st.session_state.get("dataset")
    target = st.session_state.get("target_column")

    if reference is None or target is None:
        st.info("Load a dataset first.")
        return

    with panel("Global feature importance"):
        if st.button("Compute XGBoost SHAP importance"):
            with st.spinner("Computing SHAP values..."):
                frame, error = explain_xgboost(reference, target)
            if error:
                st.warning(error)
            else:
                dataframe(frame.head(20))
                st.bar_chart(frame.set_index("Feature").head(15))

    with panel("Patient-level linear contributions"):
        patient_index = st.number_input("Reference patient row", min_value=0, max_value=len(reference)-1, value=0, step=1)
        if st.button("Explain selected reference patient"):
            patient = reference.iloc[[int(patient_index)]].drop(columns=[target])
            try:
                frame = explain_linear_model(reference, target, patient)
                dataframe(frame.head(15))
            except ValueError as exc:
                st.error(f"Explanation could not be computed: {exc}")

    disclaimer("Feature contributions explain model behaviour. They do not establish biological causation or clinical causality.")
