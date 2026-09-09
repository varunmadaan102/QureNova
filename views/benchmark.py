import streamlit as st
import pandas as pd
from components.theme import dataframe, disclaimer, panel, section_header
from core.evaluation import model_result_table
from components.charts import benchmark_bar, stability_chart
from services.comparison_service import (
    merge_all_results,
    get_comparison_narrative,
    get_pca_effect_analysis,
    get_quantum_vs_classical_summary,
)

def render():
    section_header("Model benchmark", "Compare measured performance, stability, and runtime across available artifacts.")
    result = st.session_state.get("experiment_result")
    if not result:
        st.info("Run an experiment first.")
        return

    # Get fair comparison
    comparison_df = merge_all_results(result)
    narrative = get_comparison_narrative(result)
    
    # Display fair comparison title and summary
    st.subheader(narrative["title"])
    st.info(narrative["summary"])
    
    # Protocol warning
    st.warning(narrative["protocol_warning"])
    
    # Display unified comparison table
    st.subheader("Unified Comparison")
    display_cols = [
        "Model",
        "Variant",
        "Protocol",
        "Accuracy",
        "BalancedAccuracy",
        "F1",
        "ROC_AUC",
        "Time(s)",
    ]
    display_cols = [col for col in display_cols if col in comparison_df.columns]
    
    # Format numeric columns for display
    display_df = comparison_df[display_cols].copy()
    for col in ["Accuracy", "BalancedAccuracy", "F1", "ROC_AUC"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
    for col in ["Time(s)"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    
    dataframe(display_df)
    
    st.caption(
        "Unified comparison showing all three experiment variants: "
        "(1) Classical baseline without dimensionality reduction, "
        "(2) Classical with PCA as a control to isolate quantum effects, "
        "(3) Quantum kernel experiment. "
        "Note: Different protocols (CV vs holdout) mean results are not directly comparable by rank."
    )
    
    # PCA effect analysis
    pca_analysis = get_pca_effect_analysis(result)
    if pca_analysis:
        st.subheader("PCA Effect Analysis")
        st.info(
            "The PCA control experiment isolates the effect of dimensionality reduction. "
            "This helps distinguish whether performance differences come from PCA or from the quantum kernel."
        )
        pca_df = pd.DataFrame(pca_analysis)
        pca_display = pca_df[[
            "model",
            "accuracy_baseline",
            "accuracy_pca",
            "accuracy_delta",
            "f1_baseline",
            "f1_pca",
            "f1_delta",
        ]].copy()
        pca_display.columns = ["Model", "Acc (no PCA)", "Acc (PCA)", "Δ Acc", "F1 (no PCA)", "F1 (PCA)", "Δ F1"]
        for col in ["Acc (no PCA)", "Acc (PCA)", "Δ Acc", "F1 (no PCA)", "F1 (PCA)", "Δ F1"]:
            pca_display[col] = pca_display[col].apply(lambda x: f"{x:.4f}")
        dataframe(pca_display)
        st.caption(
            "Δ (delta) shows the change in performance from adding PCA. "
            "Negative values indicate PCA reduced performance on this dataset."
        )
    
    # Quantum assessment
    quantum_summary = get_quantum_vs_classical_summary(result)
    if quantum_summary.get("status") == "completed":
        st.subheader("Quantum Assessment")
        st.info(narrative["quantum_assessment"])
        
        q_vs_c = quantum_summary["quantum_vs_best_classical"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Best Classical Accuracy",
                f"{q_vs_c['classical_accuracy']:.4f}",
                label_visibility="visible"
            )
        with col2:
            st.metric(
                "Quantum Accuracy",
                f"{q_vs_c['quantum_accuracy']:.4f}",
                delta=f"{q_vs_c['accuracy_delta']:+.4f}",
                label_visibility="visible"
            )
        with col3:
            st.metric(
                "Comparison",
                "Holdout" if q_vs_c['accuracy_delta'] != 0 else "N/A",
                label_visibility="visible"
            )
        
        st.caption(
            f"Best classical model: {q_vs_c['classical_model']} (stratified 5-fold CV). "
            f"Quantum used holdout validation with different sample size. "
            f"Kernel alignment: {quantum_summary['quantum_metadata'].get('kernel_alignment', 'N/A')}"
        )

    # Classical benchmark charts
    classical = result["classical_results"]
    
    with panel("Cross-validation performance"):
        st.plotly_chart(benchmark_bar(classical))
        st.caption("Bars show mean cross-validation F1; error bars show fold-to-fold variation.")

    with panel("F1 stability"):
        st.plotly_chart(stability_chart(classical))
        st.caption("This stability view compares the distribution of fold-level F1 scores across classical models.")

    best = max(
        classical.items(),
        key=lambda x: x[1]["summary"]["f1"]["mean"] or -1
    )
    st.success(f"Best classical model by mean F1: {best[0]} ({best[1]['summary']['f1']['mean']:.3f})", icon=":material/check_circle:")

    disclaimer("Model selection should consider stability, computational cost, calibration, and external validation. No quantum advantage claim is inferred from a single experiment.")
