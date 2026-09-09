import streamlit as st
from components.theme import disclaimer, panel, section_header

def render():
    section_header("Limitations", "Research boundaries that should remain visible alongside every result.")
    with panel("Current limitations"):
        st.markdown("""
### Current limitations

- Research prototype, not clinically validated.
- Binary classification only.
- Performance depends on dataset quality and representativeness.
- Classical and quantum validation procedures are currently different because quantum kernel simulation cost is substantially higher.
- A single experiment does not demonstrate quantum advantage.
- Simulator results do not establish real quantum hardware performance.
- Classical results use stratified cross-validation while the bounded quantum
  demonstration uses a stratified holdout; the benchmark labels these as
  contextual rather than directly comparable.
- The interactive anatomical view is a stylized orientation graphic and is not
  patient anatomy, lesion localization, or a model input.
- The Biomedical Visual Lab uses procedural geometry and synthetic grayscale
  projections; it is not medical imaging, pathology, tumor segmentation, or
  radiology interpretation.
- Probability outputs should not be interpreted as calibrated clinical risk unless calibration and external validation are performed.
- Distribution shift and out-of-distribution data remain major deployment concerns.
""")

    with panel("Responsible interpretation"):
        st.markdown("""
### Responsible interpretation

QureAI is designed to compare machine-learning approaches and surface measurable
trade-offs. The synthetic Wisconsin-style feature schema and benign/malignant
labels are educational dataset categories, not clinical validation or evidence
about prevalence, treatment, or an individual's diagnosis. Use the Guide &
Methodology page for the CSV contract, leakage reminders, model-term definitions,
and troubleshooting. Do not use this prototype to diagnose patients, replace
clinicians, or make treatment decisions.
""")
    disclaimer("Research-only outputs require clinical expertise, calibration, external validation, and governed use before any real-world consideration.")
