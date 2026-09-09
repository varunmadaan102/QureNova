# QureAI v2

QureAI is a research prototype for the SIH 26139 hybrid quantum machine
learning platform concept. Its evidence-backed MVP compares classical
machine-learning models and a bounded quantum-kernel QSVC workflow on
breast-cancer-style biomedical tabular data.

## Quick start

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Train persisted prediction models (the checked-in 30-feature demo CSV is the
default input):

```powershell
python scripts/train_models.py
```

Artifacts are written to `results/models/` (`*.joblib`,
`evaluation_results.json`, `feature_metadata.json`, and
`dataset_metadata.json`). Install `requirements-optional.txt` to enable
XGBoost, SHAP, or the quantum demonstration; the app detects missing optional
packages and keeps classical workflows available.

If PowerShell blocks `pip.exe`, always use:

```powershell
python -m pip install -r requirements.txt
```

For a lightweight deployment, install only `requirements.txt`. The optional
integrations in `requirements-optional.txt` add XGBoost, SHAP, and Qiskit; the
application detects missing packages and keeps the classical workflow
available. Quantum execution is deliberately opt-in and is not required for
startup or for the default jury walkthrough.

## Workflow

CSV → validation → preprocessing → stratified cross-validation → classical models + QSVC → benchmark → patient prediction → explainability.

## SIH 26139 alignment

The current implementation satisfies the demonstrable MVP requirements:

- hybrid classical preprocessing plus an opt-in Qiskit fidelity-quantum-kernel
  QSVC workflow;
- Logistic Regression, calibrated SVM, and optional XGBoost baselines;
- schema validation, median imputation, scaling, PCA, persisted pipelines, and
  leakage-safe cross-validation;
- patient-row inference, model behavior explanations, benchmark charts, and
  reproducible experiment metadata;
- accuracy, balanced accuracy, precision, sensitivity/recall, specificity,
  F1, ROC-AUC, confusion matrices, runtime, kernel diagnostics, and target
  alignment.

Breast-cancer tabular classification is the validated MVP. The data, model,
artifact, and quantum-backend boundaries are designed for future cardiovascular,
neurological, imaging, or genomics profiles, but those domains are roadmap
capabilities rather than completed evidence. The prototype does not claim
quantum advantage, clinical accuracy, or real-world early-detection benefit.

## Guide and input format

Open **Guide & Methodology** in the app before running an experiment. It
documents every page, the synthetic Wisconsin-style 30-feature diagnostic
schema, safe header normalization, optional targets, missing/extra/reordered
columns, numeric values, demo flows, model terms, troubleshooting, and the
non-diagnostic research boundary.

For a presentation or jury demonstration, use the built-in five-minute flow:
**Overview → Data Workspace → Load built-in demonstration dataset → 3D
orientation → Experiment Lab → Model Benchmark → Patient Analysis → Load
Example Patient**. Medical researchers can use the guide's research checklist
to review provenance, missingness, leakage controls, validation design,
calibration, subgroup performance, and external generalization before drawing
any conclusion.

Training CSVs should contain one row per record, a header row, the 30 numeric
feature names used by `data/demo/qureai_demo_biomedical.csv`, and optionally a
binary target such as `diagnosis`. Headers may vary in case and use spaces or
underscores; the common Wisconsin/UCI suffix style (`radius_mean`, `radius_se`,
and `radius_worst`) is also accepted and mapped to the canonical schema.
Feature order does not matter. Extra columns are ignored with a warning.
Patient prediction CSVs use the feature columns only and do not need a target.
Blank numeric cells are handled as missing values by preprocessing;
non-numeric feature values and missing required features are schema errors.

Data Workspace also includes a bounded Plotly 3D feature-space orientation using
loaded numeric rows and PCA. It is explicitly a visual aid—not anatomy,
evidence, or a clinical prediction.

Guide & Methodology also includes a **Biomedical Visual Lab** with generated
layered-tissue, synthetic-tumor, and X-ray-style projection views. These are
procedural educational simulations for presentation aesthetics only. They are
not real scans, pathology, patient anatomy, tumor segmentation, radiology
results, or inputs to the prediction models.

## Deployment and privacy checklist

1. Use Python 3.11 or newer and install dependencies from `requirements.txt`;
   install `requirements-optional.txt` only when the deployment has enough
   memory and the optional demonstrations are required.
2. Generate compatible persisted models with
   `python scripts/train_models.py` before relying on artifact-backed patient
   analysis. If artifacts are absent, the app transparently trains a bounded
   in-memory fallback.
3. Keep the built-in synthetic dataset for public demos. Upload only
   de-identified data in an approved environment; do not place identifiable
   health information in the repository, logs, screenshots, or public hosting.
4. Expect quantum simulation to be substantially slower than classical
   baselines. Leave it disabled for normal demonstrations.
5. Treat all outputs as research measurements. They require provenance,
   external validation, calibration, subgroup analysis, and expert review
   before any research conclusion or clinical use.

## Important

This is a research and demonstration system. Benign/malignant labels in the
synthetic example do not claim clinical validation, prevalence, treatment, or
diagnosis. Do not upload identifiable health information or use the application
as a diagnostic device.
