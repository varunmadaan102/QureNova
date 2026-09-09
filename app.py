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
    "Overview": overview,
    "Data Workspace": data_workspace,
    "Experiment Lab": experiment_lab,
    "Quantum Analysis": quantum_analysis,
    "Benchmark": benchmark,
    "Patient Analysis": patient_analysis,
    "Explainability": explainability,
    "System Status": system_status,
    "Limitations": limitations,
    "Guide & Methodology": guide,
}

with st.sidebar:
    sidebar_brand()
    page_name = st.radio("Navigate", list(PAGES.keys()))
    st.divider()
    st.caption("v2.0.0 · local session")
    st.caption("Research prototype · not for clinical diagnosis")

top_status(page_name)
PAGES[page_name].render()
