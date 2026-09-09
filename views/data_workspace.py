import streamlit as st
import pandas as pd
from components.charts import feature_space_figure
from components.theme import dataframe, disclaimer, metric_strip, panel, section_header, status_badge
from config.constants import DEMO_FEATURE_NAMES
from core.data import (
    clean_dataframe,
    infer_target_column,
    load_demo_dataset,
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
    section_header("Data workspace", "Load data, inspect schema, and prepare an experiment.")

    with st.expander("How to use this page and prepare a CSV", expanded=False):
        st.markdown(
            """
            Start with the built-in demonstration dataset, or upload a comma-separated
            file with one row per record. The diagnostic demo schema has 30 numeric
            feature columns plus an optional target such as `diagnosis`. Headers are
            matched case-insensitively after trimming whitespace and normalizing spaces
            and underscores. Both canonical names such as `mean_radius` and common
            Wisconsin/UCI names such as `radius_mean`, `radius_se`, and
            `radius_worst` are accepted.
            Columns may be reordered; extra columns are ignored with a warning. Missing
            required features, duplicate headers, and non-numeric feature values stop
            model preparation. Blank numeric cells are treated as missing and imputed
            during preprocessing. Choose **No target** for inspection-only data; an
            experiment still needs a binary target.
            """
        )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    use_demo = st.button("Load built-in demonstration dataset")

    if uploaded is not None:
        try:
            df = clean_dataframe(pd.read_csv(uploaded))
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
            st.error(f"Could not read the uploaded CSV: {exc}")
            return
        st.session_state["dataset"] = df
    elif use_demo:
        st.session_state["dataset"] = _demo()

    df = st.session_state.get("dataset")
    if df is None:
        st.info("Upload a CSV or load the demonstration dataset.")
        return

    status_badge(f"DATASET LOADED · {len(df)} ROWS × {len(df.columns)} COLUMNS", "ok")
    dataframe(df.head(20))

    target_guess = infer_target_column(df)
    target_options = ["No target (inspection only)"] + list(df.columns)
    target_default = (
        target_guess if target_guess in df.columns else target_options[0]
    )
    target = st.selectbox(
        "Target column",
        target_options,
        index=target_options.index(target_default),
        help="Targets are optional for inspection and visualization, but experiments and supervised predictions require a binary target.",
    )
    target = None if target == target_options[0] else target
    st.session_state["target_column"] = target

    report = validate_dataset(
        df,
        target,
        required_features=DEMO_FEATURE_NAMES,
    )
    metric_strip([
        ("Rows", report["rows"]),
        ("Columns", report["columns"]),
        ("Missing cells", report["missing_cells"]),
    ])

    if target is None:
        st.info("No target selected. Inspection and the feature-space orientation remain available; supervised experiments require a binary target.")
    elif df[target].nunique(dropna=True) != 2 or df[target].isna().any():
        st.error("Selected target is not binary. QureNova currently supports binary classification.")
    else:
        st.success(f"Binary target detected: {df[target].nunique()} classes")

    for error in report["errors"]:
        st.error(error)
    if report["warnings"]:
        for warning in report["warnings"]:
            st.warning(warning)

    # Dataset quality warnings
    quality_warnings = get_data_quality_warnings(df)
    if quality_warnings:
        with panel("Data quality observations"):
            for warning in quality_warnings:
                st.warning(warning)

    # Dataset profile
    with panel("Dataset profile"):
        profile = get_dataset_profile(df)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Samples", profile["total_samples"])
        with col2:
            st.metric("Features", profile["total_features"])
        with col3:
            st.metric("Missing values", profile["missing_values"]["count"])
        with col4:
            st.metric("Duplicate rows", profile["duplicates"])
        
        if profile["feature_statistics"]:
            st.subheader("Feature ranges (min, median, max)")
            stats_data = []
            for feat_name in profile["numeric_features"][:15]:  # Limit to first 15 for readability
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

    numeric_columns = numeric_feature_columns(df, target)
    with panel("3D feature-space orientation"):
        st.caption(
            "This lightweight Plotly view uses the loaded numeric rows, capped at 300 "
            "records for responsiveness. It is visual orientation—not anatomy, "
            "evidence, or a clinical prediction."
        )
        if len(numeric_columns) < 3:
            st.info("At least three usable numeric feature columns are needed for the orientation view.")
        else:
            selected_features = st.multiselect(
                "Numeric features used for PCA",
                numeric_columns,
                default=numeric_columns[: min(10, len(numeric_columns))],
                key="feature_space_columns",
                help="Changing this selection changes the view only; it does not change experiment preprocessing.",
            )
            if len(selected_features) < 3:
                st.warning("Select at least three numeric features to draw the orientation.")
            else:
                try:
                    st.plotly_chart(
                        feature_space_figure(
                            df,
                            target_column=target,
                            feature_columns=selected_features,
                            max_rows=300,
                        ),
                        width="stretch",
                        config={"displaylogo": False},
                    )
                    st.caption("The 3D projection shows mathematical similarity in selected numeric features; it is not anatomy or a clinical result.")
                except ValueError as exc:
                    st.warning(f"Feature-space view unavailable: {exc}")
    disclaimer("The feature-space chart is an educational orientation of tabular values; it does not depict anatomy or establish clinical evidence.")
