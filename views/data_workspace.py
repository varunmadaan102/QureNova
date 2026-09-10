import streamlit as st
import pandas as pd
from components.charts import feature_space_figure
from components.theme import (
    dataframe, disclaimer, metric_strip, panel, section_header, status_badge,
    page_header, empty_state, loading_state, error_state, section_divider
)
from config.constants import DEMO_FEATURE_NAMES
from security.privacy import find_direct_identifiers, research_notice
from security.upload_policy import validate_upload_name, validate_upload_size
from core.data import (
    clean_dataframe,
    infer_target_column,
    load_demo_dataset,
    load_breast_cancer_wisconsin,
    numeric_feature_columns,
    validate_schema,
)
from core.validation import validate_dataset
from services.dataset_intelligence_service import (
    get_dataset_profile,
    get_data_quality_warnings,
)

def _demo():
    return load_demo_dataset()

def render():
    page_header(
        "Data Workspace",
        "Upload biomedical datasets, inspect profiles, and prepare experiments",
        "CSV ingestion · schema validation · quality assessment"
    )

    # Upload Section
    section_header("Dataset Upload", "Load your data or use a built-in dataset")

    selector = st.radio(
        "Dataset source",
        options=["Demo dataset", "Breast Cancer Wisconsin (local/optional download)", "Upload CSV"],
        index=0,
        horizontal=True,
        help="Choose a built-in dataset or upload your own CSV.",
    )

    if selector == "Demo dataset":
        if st.button("Load Demo Dataset", use_container_width=True):
            st.session_state["dataset"] = _demo()
            st.session_state["dataset_profile"] = None

    elif selector == "Breast Cancer Wisconsin (local/optional download)":
        if st.button("Load Breast Cancer Wisconsin", use_container_width=True):
            st.session_state["dataset"] = load_breast_cancer_wisconsin(allow_download=True)
            st.session_state["dataset_profile"] = "breast_cancer_wisconsin_diagnostic"

    else:
        uploaded = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            help="Comma-separated file with one row per sample",
        )
        if uploaded is not None:
            try:
                validate_upload_name(uploaded.name)
                validate_upload_size(uploaded.size)
                df = clean_dataframe(pd.read_csv(uploaded))
                st.session_state["dataset"] = df
                st.session_state["dataset_profile"] = None
            except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
                error_state("CSV Upload Error", f"Could not accept the uploaded file: {exc}")
                return

    # If a dataset is already loaded from a prior interaction, keep it.
    # Initialize target selection based on the currently loaded dataset.
    if "dataset" not in st.session_state:
        st.session_state["dataset"] = None

    df = st.session_state.get("dataset")
    if df is None:
        empty_state(
            "📤",
            "No Dataset Loaded",
            "Upload a CSV file or load the demonstration dataset to begin.",
            "Use the uploader above or click 'Load Demo Dataset'"
        )
        return

    section_divider()

    # Dataset Status
    section_header("Dataset Status", "Overview of loaded data")
    status_badge(f"LOADED · {len(df)} ROWS × {len(df.columns)} COLUMNS", "ok")
    st.caption(research_notice())
    direct_identifiers = find_direct_identifiers(df.columns)
    if direct_identifiers:
        st.warning("Potential direct identifiers detected: " + ", ".join(direct_identifiers[:8]) + ". Remove or de-identify them before research use.")
    
    # Preview
    with panel("Data Preview"):
        st.caption(f"Showing first {min(20, len(df))} rows")
        dataframe(df.head(20))

    section_divider()

    # Target Selection
    section_header("Target Configuration", "Select a binary classification target")

    # In medically-grounded binary mode, we keep target column in session
    # so downstream experiment + explainability stays consistent.
    target_guess = infer_target_column(df)
    target_options = ["No target (inspection only)"] + list(df.columns)
    target_default = (
        target_guess if target_guess in df.columns else target_options[0]
    )
    target = st.selectbox(
        "Target column",
        target_options,
        index=target_options.index(target_default),
        help="Targets are optional for inspection, but experiments require a binary target.",
    )
    target = None if target == target_options[0] else target
    st.session_state["target_column"] = target

    if st.session_state.get("dataset_profile") == "breast_cancer_wisconsin_diagnostic":
        with panel("Medical Target Definition"):
            st.info("Breast Cancer Wisconsin (Diagnostic): B = Benign, M = Malignant. This semantic mapping comes from the dataset definition; QureNova never infers medical meaning from numeric label ordering.")

    section_divider()

    # Validation Report
    section_header("Data Validation", "Schema and quality assessment")
    
    is_wdbc = st.session_state.get("dataset_profile") == "breast_cancer_wisconsin_diagnostic"
    validation_features = DEMO_FEATURE_NAMES if is_wdbc else [
        c for c in df.select_dtypes(include="number").columns if c != target
    ]
    report = validate_dataset(
        df,
        target,
        required_features=validation_features,
    )
    
    # Validation status metrics
    cols = st.columns(3)
    with cols[0]:
        st.metric("Rows", report["rows"])
    with cols[1]:
        st.metric("Columns", report["columns"])
    with cols[2]:
        st.metric("Missing Cells", report["missing_cells"])

    # Validation messages
    if report["errors"]:
        for error in report["errors"]:
            error_state("Validation Error", error)
    
    if target is None:
        st.info("ℹ️ No target selected. Inspection and visualization are available; experiments require a binary target.")
    elif df[target].nunique(dropna=True) != 2 or df[target].isna().any():
        error_state(
            "Invalid Target",
            "Selected target is not binary. QureNova currently supports binary classification only."
        )
    else:
        st.success(f"✓ Binary target detected: {df[target].nunique()} classes")

    if report["warnings"]:
        for warning in report["warnings"]:
            st.warning(f"⚠️ {warning}")

    section_divider()

    # Dataset Profile
    section_header("Dataset Profile", "Detailed statistics and feature ranges")
    
    quality_warnings = get_data_quality_warnings(df)
    if quality_warnings:
        with panel("Data Quality Notes"):
            for warning in quality_warnings:
                st.warning(f"• {warning}")

    profile = get_dataset_profile(df)
    
    with panel("Summary Statistics"):
        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Samples", profile["total_samples"])
        with cols[1]:
            st.metric("Total Features", profile["total_features"])
        with cols[2]:
            st.metric("Missing Values", profile["missing_values"]["count"])
        with cols[3]:
            st.metric("Duplicate Rows", profile["duplicates"])
        
        if profile["feature_statistics"]:
            st.subheader("Feature Ranges (min, median, max)")
            stats_data = []
            for feat_name in profile["numeric_features"][:15]:
                if feat_name in profile["feature_statistics"]:
                    stats = profile["feature_statistics"][feat_name]
                    stats_data.append({
                        "Feature": feat_name,
                        "Min": f"{stats['min']:.3f}" if stats['min'] is not None else "-",
                        "Median": f"{stats['median']:.3f}" if stats['median'] is not None else "-",
                        "Max": f"{stats['max']:.3f}" if stats['max'] is not None else "-",
                        "Std Dev": f"{stats['std']:.3f}" if stats['std'] is not None else "-",
                    })
            if stats_data:
                dataframe(pd.DataFrame(stats_data))

    section_divider()

    # Feature Space Visualization
    section_header("Feature Space Orientation", "PCA-based 3D visualization of numeric features")
    
    numeric_columns = numeric_feature_columns(df, target)
    
    if len(numeric_columns) < 3:
        st.info("ℹ️ At least three numeric features are needed for the feature-space view.")
    else:
        with panel():
            st.caption(
                "This lightweight visualization uses up to 300 records for responsiveness. "
                "It is visual orientation of your data distribution—not clinical evidence."
            )
            
            selected_features = st.multiselect(
                "Numeric features for PCA",
                numeric_columns,
                default=numeric_columns[: min(10, len(numeric_columns))],
                key="feature_space_columns",
                help="Changing this selection changes the view only; it does not affect experiment preprocessing.",
            )
            
            if len(selected_features) < 3:
                st.warning("Select at least three numeric features to render the view.")
            else:
                try:
                    st.plotly_chart(
                        feature_space_figure(
                            df,
                            target_column=target,
                            feature_columns=selected_features,
                            max_rows=300,
                        ),
                        use_container_width=True,
                        config={"displaylogo": False},
                    )
                    st.caption("This 3D projection shows mathematical similarity in selected numeric features; it is not anatomy or clinical analysis.")
                except ValueError as exc:
                    st.warning(f"Feature-space view unavailable: {exc}")
    
    section_divider()
    disclaimer("Research insight only: this tool is for educational exploration of machine-learning models and does not provide clinical diagnosis or medical advice. Any results are not a substitute for professional judgment.")

