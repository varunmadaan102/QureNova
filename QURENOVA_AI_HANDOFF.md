# QureNova AI Handoff Context

## Purpose

QureNova is a Streamlit research prototype for the SIH 26139 problem statement:
**Hybrid Quantum Machine Learning Platform for Early Disease Detection**.
The validated MVP uses breast-cancer-style biomedical tabular data and compares
classical models with an opt-in, simulator-backed quantum-kernel workflow.

This file is intended for another AI coding model or teammate taking over the
repository. It records the product context, implementation decisions, known
boundaries, validation evidence, and current next steps.

## Product identity and repository

- Product name: **QureNova**
- GitHub repository: `varunmadaan102/QureNova`
- Default branch: `main`
- Local checkout currently lives at `QURE-AI-v2` because the active workspace
  holds a filesystem lock. Rename the folder to `QureNova` after closing the
  active Copilot/Streamlit process.
- Current published commit: `1e22517`
- Remote: `https://github.com/varunmadaan102/QureNova.git`
- Local-only research paper: `QureNova-Research-Paper.docx`
- The paper and generated/private artifacts are intentionally ignored by Git.

## User and research context

The intended audiences are:

1. Beginners and jury members who need a guided five-minute demonstration.
2. Medical researchers who need to inspect schema, preprocessing, validation,
   metrics, and model behavior.
3. Developers who may extend the workflow to cardiovascular, neurological,
   imaging, or genomics profiles.

The product must not claim clinical diagnosis, clinical accuracy, quantum
advantage, or real-world early-detection benefit. The demo labels are dataset
categories, not patient conclusions.

## User workflow

`Overview → Data Workspace → Experiment Lab → Benchmark → Patient Analysis → Explainability`

Supporting pages:

- **Quantum Analysis**: circuit, kernel matrix, diagnostics, backend metadata.
- **System Status**: runtime, demo data, artifacts, optional integrations.
- **Limitations**: validation, calibration, clinical, quantum, and visual limits.
- **Guide & Methodology**: beginner, researcher, jury, CSV, troubleshooting, and
  Biomedical Visual Lab guidance.

The app uses native `st.navigation()` and `st.Page()` with unique URL paths:
`overview`, `data-workspace`, `experiment-lab`, `quantum-analysis`, `benchmark`,
`patient-analysis`, `explainability`, `system-status`, `limitations`, and `guide`.

## Data contract

The checked-in demo is `data/demo/qurenova_demo_biomedical.csv`.
It contains 30 numeric biomedical-style features and a binary `diagnosis` target.
The canonical feature groups are:

- mean, error, and worst measurements;
- radius, texture, perimeter, area, smoothness, compactness, concavity,
  concave points, symmetry, and fractal dimension.

Input handling:

- headers are trimmed, lower-cased, and normalized for spaces/underscores;
- common Wisconsin/UCI names such as `radius_mean`, `radius_se`, and
  `radius_worst` are mapped safely;
- columns may be reordered;
- extra metadata columns such as `sample_id` are ignored with warnings;
- missing required features, duplicate headers, and nonnumeric values are errors;
- blank numeric cells are median-imputed by the preprocessing pipeline;
- target selection is optional for inspection but required for supervised runs;
- targets must contain exactly two non-null classes.

## Architecture

```text
app.py
  ├── components/theme.py        shared CSS, navigation-adjacent branding, panels
  ├── components/charts.py       Plotly charts and procedural biomedical visuals
  ├── core/data.py               cleaning, aliases, alignment, target encoding
  ├── core/validation.py         schema and dataset checks
  ├── core/preprocessing.py      imputation, scaling, PCA, safe bounds
  ├── core/evaluation.py         classification metrics and summary tables
  ├── models/classical.py        Logistic Regression, calibrated SVM, XGBoost
  ├── models/quantum.py          precomputed-kernel QSVC
  ├── quantum/                   feature maps, fidelity kernel, alignment
  ├── services/experiment_service.py
  ├── services/prediction_service.py
  ├── services/explainability_service.py
  └── views/                     page-level render() functions
```

No database, Flask backend, Supabase connection, job queue, or multi-user
service is required for this local/demo MVP. A future production workflow could
add FastAPI, a governed database, authentication, audit logs, and background
experiment jobs.

## Preprocessing and leakage controls

Classical experiments use fold-fitted:

1. median imputation;
2. `StandardScaler`;
3. PCA, normally four components in the persisted demo artifact;
4. model fitting inside each stratified fold.

The preprocessor is never fitted on held-out rows during evaluation. Persisted
prediction artifacts store preprocessing and estimator together in a Pipeline.
Patient inference aligns input columns to the persisted canonical feature order.

## Models and current tuning

The current classical model factory is `models/classical.py`.

### Logistic Regression

- `C=0.5`
- `max_iter=3000`
- strongest aggregate model on the checked-in demo

### Calibrated RBF SVM

- `C=1.0`
- `class_weight="balanced"`
- `CalibratedClassifierCV(method="sigmoid", cv=3, ensemble=False)`
- selected as the sensitivity-oriented option

### XGBoost

- `n_estimators=180`
- `max_depth=4`
- `learning_rate=0.1`
- `subsample=0.8`
- `colsample_bytree=1.0`
- `eval_metric="logloss"`
- `n_jobs=1`

These settings were selected using stratified five-fold cross-validation on the
demo schema. They are not universal clinical hyperparameters and must be
re-tuned for a new cohort or modality.

## Evidence from the current demo

Measured with leakage-safe stratified five-fold cross-validation:

| Model | Balanced accuracy | F1 | Sensitivity | Specificity | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9608 | 0.9537 | 0.9384 | 0.9831 | 0.9918 |
| Calibrated balanced SVM | 0.9632 | 0.9528 | 0.9573 | 0.9691 | 0.9917 |
| XGBoost | 0.9561 | 0.9455 | 0.9432 | 0.9691 | 0.9873 |

Interpretation:

- Logistic Regression is the best default aggregate/F1 model.
- SVM is preferable when sensitivity is prioritized, with a specificity tradeoff.
- XGBoost is a useful nonlinear baseline but is not the best current performer.
- These are demo-dataset measurements, not clinical performance claims.

The quantum workflow is deliberately bounded to at most 64 training rows and
uses a stratified holdout, so its result is explicitly contextual and not
directly comparable with classical cross-validation means.

## Caching and performance

- `load_demo_dataset()` uses `@st.cache_data`.
- Prediction model bundles use `@st.cache_resource`.
- PCA feature-space figures use `@st.cache_data`.
- Quantum simulation is opt-in and bounded because fidelity-kernel computation
  is quadratic and can be slow.
- Streamlit AppTest has loaded with zero exceptions; the live health endpoint
  returns HTTP 200.

## Accessibility and UI decisions

- Native Streamlit navigation supplies real active states and URL paths.
- The old radio `:has()` CSS dependency was removed.
- Captions are kept readable and chart captions summarize visual findings.
- Decorative glyphs in raw HTML use `aria-hidden="true"`.
- Status badges include text and do not rely on color alone.
- Metric strips wrap into two-column rows for narrow screens.
- A 480px breakpoint stacks the top status bar and reduces hero padding.
- The Overview hero is intentionally reserved for the landing page; dense pages
  use section headers.

## Visual Lab boundary

The Guide page includes procedural Plotly visualizations for layered tissue,
synthetic tumor/nodule geometry, and an X-ray-style projection. They are
educational simulations only: not scans, pathology, patient anatomy, tumor
segmentation, radiology results, or model inputs.

## Deployment state

`requirements.txt` now includes the integrations needed by the complete public
demo:

- Streamlit, pandas, NumPy, scikit-learn, Plotly, matplotlib, joblib;
- XGBoost;
- SHAP;
- Qiskit;
- Qiskit Machine Learning.

The app still detects missing optional imports gracefully, but a deployment
should reinstall dependencies after the requirements change. The previously
observed hosted screenshot showed a stale build with QureAI branding and missing
Qiskit; redeploy/reboot the Streamlit app from the renamed repository to load
QureNova and the current dependency set.

## Files that matter most

- `app.py`: page registry and native routing.
- `components/theme.py`: visual system and responsive CSS.
- `models/classical.py`: tuned classical model definitions.
- `services/experiment_service.py`: evaluation orchestration.
- `services/prediction_service.py`: persisted/in-memory inference.
- `core/data.py`: schema aliases and demo loading.
- `core/evaluation.py`: metrics including sensitivity and specificity.
- `views/guide.py`: educational and safety guidance.
- `README.md`: setup, workflow, deployment, privacy, and scope.
- `QureNova-Research-Paper.docx`: local-only report.

## Verification commands

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe -m compileall -q app.py components core models services views scripts
.\.venv\Scripts\python.exe scripts\train_models.py
streamlit run app.py
```

Expected baseline: 9 tests pass, compilation succeeds, and the app opens without
exceptions. Do not commit generated `.joblib` artifacts or the local DOCX.

## Safe next steps

1. Redeploy the renamed `QureNova` repository on Streamlit Cloud.
2. Confirm the hosted app title, sidebar brand, demo filename, and Qiskit status.
3. Add a separate holdout or external cohort before making any research claim.
4. Consider threshold tuning for an explicitly chosen sensitivity/specificity
   objective, with calibration and confidence intervals.
5. Add model/version metadata to every experiment artifact.
6. Keep future imaging/genomics work as separate profiles rather than silently
   mixing incompatible feature contracts.
