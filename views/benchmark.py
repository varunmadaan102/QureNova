import streamlit as st
import pandas as pd
from components.theme import dataframe, disclaimer, panel, section_header
from core.evaluation import model_result_table
from components.charts import benchmark_bar, stability_chart

def render():
    section_header("Model benchmark", "Compare measured performance, stability, and runtime across available artifacts.")
    result = st.session_state.get("experiment_result")
    if not result:
        st.info("Run an experiment first.")
        return

    classical = result["classical_results"]
    table = model_result_table(classical)

    q = result["quantum_results"]
    if q.get("available"):
        row = {"Model": f'QSVC ({q["configuration"]["feature_map"].upper()}-{q["configuration"]["num_qubits"]}Q)'}
        for key, value in q["metrics"].items():
            row[key.upper()] = value
        row["Time (s)"] = q["timing_seconds"]
        row["Validation"] = q["configuration"]["validation"]
        row["Comparison"] = q["configuration"].get(
            "comparison_status",
            "Contextual only; not directly comparable to CV means",
        )
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)

    summary_columns = [
        "Model",
        "ACCURACY",
        "BALANCED_ACCURACY",
        "F1",
        "ROC_AUC",
        "SENSITIVITY",
        "SPECIFICITY",
        "Time (s)",
        "Validation",
    ]
    visible_columns = [column for column in summary_columns if column in table.columns]
    dataframe(table[visible_columns])
    st.caption(
        "This compact view shows mean scores for quick comparison; fold variability "
        "remains available in the detailed experiment artifact. Sensitivity is the "
        "positive-class recall; specificity is the true-negative rate. These are "
        "dataset evaluation metrics, not clinical guarantees. "
        "Classical rows are cross-validation means; the bounded quantum row is a "
        "contextual holdout demonstration and must not be ranked directly against them."
    )

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
