import streamlit as st
import pandas as pd
from components.theme import dataframe, disclaimer, section_header, status_badge
from services.prediction_service import predict_patients
from core.data import infer_target_column, load_demo_dataset

def render():
    section_header("Patient analysis", "Clinician-first review of individual model outputs against the reference schema.")
    reference = st.session_state.get("dataset")
    target = st.session_state.get("target_column")

    with st.expander("How patient analysis works", expanded=False):
        st.markdown(
            """
            First load a compatible reference dataset in **Data Workspace**, then
            upload one or more patient rows containing the same 30 numeric diagnostic
            feature names (or safe normalized equivalents, including common
            `radius_mean`/`radius_se`/`radius_worst` headers). The patient CSV does not
            need a target column: the reference target supplies the trained labels.
            Headers may be reordered and extra columns are ignored, while missing
            features, duplicate headers, or non-numeric values are reported as schema
            errors. Blank numeric cells can be imputed by the model pipeline, but
            inspect missingness before interpreting any output.
            """
        )
        st.info(
            "Suggested demo flow: load the built-in dataset → choose its binary "
            "`diagnosis` target → return here → Load Example Patient."
        )

    if reference is None:
        reference = load_demo_dataset()
        target = infer_target_column(reference)
        st.info("Using the checked-in demo dataset. Load another dataset in Data Workspace to replace it.")
    if target is None:
        st.error("The reference dataset has no target column.")
        return

    st.caption(
        "Upload one or more patient rows using the reference feature columns. "
        "Columns may be reordered, and extra columns are ignored; missing or "
        "non-numeric features are reported as schema errors. A target column is "
        "not expected in this prediction input."
    )

    uploaded = st.file_uploader("Upload patient CSV", type=["csv"], key="patient_upload")
    use_example = st.button("Load Example Patient")
    if uploaded is None and not use_example:
        template = reference.drop(columns=[target]).head(3)
        st.markdown("### Example patient CSV structure")
        dataframe(template)
        st.caption("The first example row can be copied into a CSV for a quick prediction smoke test.")
        return

    if uploaded is not None:
        try:
            patients = pd.read_csv(uploaded)
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
            st.error(f"Could not read the uploaded patient CSV: {exc}")
            return
        input_source = "uploaded CSV"
    else:
        patients = reference.drop(columns=[target]).head(1).copy()
        input_source = "built-in example row"
    st.caption(f"Input source: {input_source}")
    dataframe(patients)
    with st.expander("Input checks before running", expanded=False):
        st.markdown(
            """
            Check that each row represents one record and that numeric values use
            decimal notation (not embedded units such as `12 mm`). Empty cells are
            allowed by the preprocessing pipeline but can reduce reliability. The
            model output is a research estimate against the reference model—not a
            diagnosis, treatment recommendation, or substitute for clinical review.
            """
        )
    try:
        with st.spinner("Training reference models and analysing patients..."):
            result, mapping, features, source = predict_patients(
                reference, target, patients, return_source=True
            )
        if source.startswith("trained in memory"):
            st.info(
                "No compatible persisted model bundle was found; models were "
                "trained in memory for this analysis."
            )
        else:
            st.caption(f"Model source: {source}")
        status_badge(f"ANALYSED · {len(result)} PATIENT RECORD(S)", "ok")
        dataframe(result)

        selected = st.selectbox("Inspect patient", result["Patient"].tolist())
        row = result[result["Patient"] == selected].iloc[0]
        st.json(row.to_dict())

        disclaimer("Predictions are experimental model outputs, not clinical diagnoses.")
    except ValueError as exc:
        st.error(f"Schema validation failed: {exc}")
    except Exception as exc:
        st.error("Patient analysis could not be completed. Check the input schema and try again.")
        with st.expander("Technical details"):
            st.code(str(exc), language="text")
