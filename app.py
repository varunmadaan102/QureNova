import streamlit as st
from components.theme import apply_theme, sidebar_brand, top_status
from views import (
    overview,
    data_workspace,
    experiment_lab,
    quantum_analysis,
    benchmark,
    experiment_history,
    patient_analysis,
    explainability,
    limitations,
    system_status,
    guide,
)

st.set_page_config(
    page_title="QureNova",
    page_icon=":material/biotech:",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

PAGES = [
    st.Page(overview.render, title="Overview", url_path="overview", icon=":material/biotech:"),
    st.Page(data_workspace.render, title="Data Workspace", url_path="data-workspace", icon=":material/table_view:"),
    st.Page(experiment_lab.render, title="Experiment Lab", url_path="experiment-lab", icon=":material/science:"),
    st.Page(quantum_analysis.render, title="Quantum Analysis", url_path="quantum-analysis", icon=":material/blur_on:"),
    st.Page(benchmark.render, title="Benchmark", url_path="benchmark", icon=":material/monitoring:"),
    st.Page(experiment_history.render, title="Experiment History", url_path="experiment-history", icon=":material/history:"),
    st.Page(patient_analysis.render, title="Patient Analysis", url_path="patient-analysis", icon=":material/person_search:"),
    st.Page(explainability.render, title="Explainability", url_path="explainability", icon=":material/psychology:"),
    st.Page(system_status.render, title="System Status", url_path="system-status", icon=":material/health_and_safety:"),
    st.Page(limitations.render, title="Limitations", url_path="limitations", icon=":material/warning:"),
    st.Page(guide.render, title="Guide & Methodology", url_path="guide", icon=":material/menu_book:"),
]

with st.sidebar:
    sidebar_brand()
    st.divider()
    page = st.navigation(PAGES, position="sidebar")
    st.divider()
    st.caption("v2.0.0 · local session")
    st.caption("Research prototype · not for clinical diagnosis")

top_status("QureNova research console")
page.run()
