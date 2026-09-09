import streamlit as st
from components.theme import apply_theme, sidebar_brand, top_status
from views import (
    overview,
    data_workspace,
    experiment_lab,
    quantum_analysis,
    benchmark,
    patient_analysis,
    explainability,
    limitations,
    system_status,
    guide,
)

st.set_page_config(
    page_title="QureAI",
    page_icon=":material/biotech:",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

PAGES = {
    "Overview": st.Page(overview.render, title="Overview", icon=":material/biotech:"),
    "Data Workspace": st.Page(data_workspace.render, title="Data Workspace", icon=":material/table_view:"),
    "Experiment Lab": st.Page(experiment_lab.render, title="Experiment Lab", icon=":material/science:"),
    "Quantum Analysis": st.Page(quantum_analysis.render, title="Quantum Analysis", icon=":material/blur_on:"),
    "Benchmark": st.Page(benchmark.render, title="Benchmark", icon=":material/monitoring:"),
    "Patient Analysis": st.Page(patient_analysis.render, title="Patient Analysis", icon=":material/person_search:"),
    "Explainability": st.Page(explainability.render, title="Explainability", icon=":material/psychology:"),
    "System Status": st.Page(system_status.render, title="System Status", icon=":material/health_and_safety:"),
    "Limitations": st.Page(limitations.render, title="Limitations", icon=":material/warning:"),
    "Guide & Methodology": st.Page(guide.render, title="Guide & Methodology", icon=":material/menu_book:"),
}

with st.sidebar:
    sidebar_brand()
    page = st.navigation(PAGES, position="sidebar")
    st.divider()
    st.caption("v2.0.0 · local session")
    st.caption("Research prototype · not for clinical diagnosis")

top_status(page.title)
page.run()
