import streamlit as st
import pandas as pd
from components.theme import dataframe, disclaimer, panel, section_header, status_badge
from services.prediction_service import predict_patients
from services.patient_experience_service import (
    get_patient_interpretation,
    get_feature_contribution_summary,
    get_model_calibration_summary,
)
from core.data import infer_target_column, load_demo_dataset
from config.settings import RISK_THRESHOLDS

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
        st.info(
            "Risk categories and priorities are experimental review labels based on "
            f"probability thresholds (<{RISK_THRESHOLDS['low_upper']:.2f} low, "
            f"<{RISK_THRESHOLDS['moderate_upper']:.2f} moderate, otherwise elevated). "
            "They are not clinically validated thresholds."
        )
        display_columns = [
            column for column in [
                "Patient", "Risk Category", "Priority", "Average Probability",
                "Prediction Reliability", "Model Agreement", "Model Confidence",
            ] if column in result.columns
        ]
        queue = result.sort_values(
            by=["Priority", "Average Probability"],
            ascending=[True, False],
            na_position="last",
        )
        dataframe(queue[display_columns])

        selected = st.selectbox("Inspect patient", result["Patient"].tolist())
        row = result[result["Patient"] == selected].iloc[0]
        
        # Enhanced interpretation layer
        interpretation = get_patient_interpretation(
            row,
            row.to_dict(),
            reference,
            features
        )
        
        # Primary assessment
        with panel("Patient assessment"):
            st.write(interpretation["primary_assessment"])
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", interpretation["confidence_level"])
            with col2:
                st.metric("Model agreement", interpretation["model_agreement"])
            with col3:
                st.metric("Reliability", row["Prediction Reliability"])
        
        # Reliability context
        with panel("Reliability & context"):
            st.write(interpretation["reliability_context"])
            if interpretation["key_observations"]:
                st.subheader("Key observations")
                for obs in interpretation["key_observations"]:
                    st.write(f"• {obs}")
        
        # Important caveats
        with panel("Important caveats"):
            for caveat in interpretation["important_caveats"]:
                st.write(f"⚠ {caveat}")
        
        # Suggested next steps
        with panel("Suggested next steps"):
            for step in interpretation["suggested_next_steps"]:
                st.write(f"→ {step}")

        with st.expander("Detailed metrics & model breakdown", expanded=False):
            left, right = st.columns(2)
            with left:
                st.metric("Risk category", row["Risk Category"])
                st.metric(
                    "Model probability",
                    "Unavailable"
                    if pd.isna(row["Average Probability"])
                    else f"{row['Average Probability']:.3f}",
                )
                st.metric("Priority", row["Priority"])
            with right:
                st.metric("Prediction reliability", row["Prediction Reliability"])
                st.metric("Model confidence", row.get("Model Confidence", "N/A"))
                st.caption(row["Reliability Note"])

            st.subheader("Model-by-model assessment")
            model_columns = [
                column for column in result.columns
                if column.endswith(" Prediction") or column.endswith(" Probability")
            ]
            dataframe(pd.DataFrame([row[model_columns].to_dict()]))
            
        if row["Model Agreement"] == "LOW AGREEMENT":
            st.warning(
                "Models produced inconsistent assessments. Automated assessment "
                "reliability may be reduced; qualified clinical review is required."
            )
        if row["Prediction Reliability"] != "HIGH":
            st.warning(
                "The input profile differs from the reference population used for "
                "this model. Treat the estimate as lower reliability, not as a "
                "statement that the patient is abnormal."
            )

        disclaimer("Predictions are experimental model outputs, not clinical diagnoses.")
    except ValueError as exc:
        st.error(f"Schema validation failed: {exc}")
    except Exception as exc:
        st.error("Patient analysis could not be completed. Check the input schema and try again.")
        with st.expander("Technical details"):
            st.code(str(exc), language="text")
