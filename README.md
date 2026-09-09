# QureNova

**A credible research console for hybrid quantum machine learning.**

QureNova is a measurement-first workspace designed for researchers, students, and juries to understand whether and where quantum machine learning can add value in biomedical applications. Rather than marketing quantum advantage, QureNova answers a harder question honestly: **WHERE, IF ANYWHERE, CAN QUANTUM MACHINE LEARNING ADD VALUE?**

The platform compares classical machine-learning baselines (Logistic Regression, SVM, XGBoost) against a bounded quantum-kernel QSVC workflow on breast-cancer-style biomedical tabular data, with transparent protocol labeling, fair benchmarking, dataset intelligence, patient-first interpretation, and explicit research boundaries.

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
`dataset_metadata.json`). The full `requirements.txt` installation includes
XGBoost, SHAP, and the quantum demonstration dependencies; the app still
detects missing packages gracefully and keeps classical workflows available.

If PowerShell blocks `pip.exe`, always use:

```powershell
python -m pip install -r requirements.txt
```

For a lightweight deployment, install `requirements.txt`. Quantum execution is
deliberately opt-in and is not required for startup or for the default jury
walkthrough. `requirements-optional.txt` remains available for environments
that intentionally want to layer those integrations separately.

## Core Principles

**Quantum is measured, not marketed.** QureNova displays results honestly regardless of outcome. If classical outperforms quantum, that's stated clearly. The platform answers whether quantum helps this specific problem, not whether quantum is inherently better.

**Transparency over confidence.** Every prediction includes reliability context, model agreement signals, and important caveats. The system never claims certainty it doesn't have.

**Separate model evidence from clinical interpretation.** What the algorithm predicts (model evidence) is distinct from what that prediction means in clinical context (human interpretation). Both are presented, neither is conflated.

**Distinguish simulation from hardware.** Quantum operations run on Qiskit simulators, not real QPU. The platform is explicit about this boundary and makes no claims about real hardware performance.

**Never fabricate clinical claims.** No "99% accuracy on breast cancer." Instead: "This is a research estimate from classifiers trained on synthetic reference data. Use only with external clinical validation."

## Workflow

**CSV → Validation → Preprocessing → Fair Comparison → Patient Analysis → Explainability**

1. **Data Workspace:** Load or upload a CSV with 30 numeric diagnostic features and an optional binary target. Inspect schema, missing values, duplicates, and feature ranges. View a 3D feature-space orientation.

2. **Experiment Lab:** Configure and run a controlled experiment comparing:
   - Classical baseline (5-fold cross-validation, no PCA)
   - Classical + PCA (control experiment to isolate PCA effects)
   - Quantum kernel QSVC (holdout 80/20 split for kernel complexity)
   
   Protocol differences (cross-validation vs. holdout) are explicitly labeled—results are not rank-comparable without accounting for this difference.

3. **Benchmark:** View fair comparison results with:
   - Unified metrics table (accuracy, balanced accuracy, precision, recall, F1, ROC-AUC)
   - Protocol transparency banner explaining why classical and quantum use different evaluation methods
   - PCA effect analysis showing whether classical gains come from dimensionality reduction or better baseline
   - Quantum vs. classical narrative: honest assessment of where quantum performed well or poorly

4. **Patient Analysis:** Upload compatible patient rows for individual predictions. Each prediction shows:
   - Primary assessment (algorithm output in plain language)
   - Confidence level (based on probability distance + model agreement)
   - Reliability context (how this patient differs from reference population)
   - Important caveats (limitations and disclaimers)
   - Suggested next steps (research or clinical actions)
   
   All predictions are experimental estimates, never diagnoses or treatment recommendations.

5. **Explainability:** Explore model behavior through:
   - Global feature importance (SHAP values or tree importance)
   - Per-patient feature contributions (linear model coefficients or tree paths)
   - Boundary cases and model agreement patterns

## Fair Benchmarking & Scientific Integrity (Phase 1)

QureNova implements a 3-way experimental comparison to isolate quantum effects:

- **Classical (no PCA):** Logistic Regression, SVM, XGBoost with standard preprocessing. 5-fold stratified cross-validation.
- **Classical + PCA:** Same classical models with optional PCA for dimensionality reduction. Helps quantify whether classical gains come from better baselines or dimensionality reduction.
- **Quantum (QSVC):** Qiskit fidelity quantum kernel with QSVC classifier. Holdout 80/20 split (different protocol due to quantum circuit complexity).

Protocol differences are **explicitly labeled** on the Benchmark view. Classical models use cross-validation; quantum uses holdout. Results cannot be directly ranked by accuracy—the evaluation methods differ.

The comparison narrative (`comparison_service.py`) generates honest assessments:
- If classical outperforms: "Classical baselines outperformed the evaluated quantum kernel configuration. This demonstrates why rigorous benchmarking is essential."
- If quantum performs better: "Quantum kernel showed measurable advantage on this configuration. This warrants further investigation with independent validation."
- If results are mixed: "Model agreement is mixed. No clear superiority emerges on this dataset."

## Dataset Transparency (Phase 2)

Data Workspace now displays comprehensive dataset intelligence:

- **Dataset profile:** Sample count, feature count, missing values, duplicates
- **Feature statistics:** Min, median, max, standard deviation for each numeric feature
- **Data quality warnings:** Alerts for sparse features, highly correlated columns, potential data drift
- **Preprocessing context:** What transformations will be applied to user data

This supports the principle that users should understand exactly what data is being trained on and how it's being prepared, **before** running experiments.

## Patient-First Interpretation (Phase 3)

Patient Analysis separates **model evidence** from **clinical interpretation**:

- **Model evidence:** What the algorithm predicts (probability, prediction class, model agreement)
- **Clinical interpretation:** What that prediction means in context (reliability, caveats, suggested actions)

Each patient receives:
- **Primary assessment** in plain language (not just a probability)
- **Confidence level** (High/Moderate/Low based on probability distance + model agreement)
- **Reliability context** (how this patient's profile differs from reference population)
- **Key observations** (notable patient characteristics vs. reference)
- **Important caveats** (limitations, disclaimers, research boundaries)
- **Suggested next steps** (recommended clinical or research actions)

This design ensures clinicians and researchers understand both what the model says and why they should (or shouldn't) trust it.

## UX Polish (Phase 4)

QureNova provides a consistent, legible research workflow:

- **Navigation components:** Consistent page headers, breadcrumb trails, workflow checkpoints
- **Error handling:** Structured validation errors with clear guidance and recovery suggestions
- **Feedback:** Operation status, loading states, data quality warnings, prediction reliability context
- **Workflow context:** Data status panel reminds users of current dataset, target, and experiment state on all pages

## QA & Documentation (Phase 5)

- **Test coverage:** 11 comprehensive tests covering data contracts, evaluation, clinical labels, experiment persistence, patient analysis, and visuals
- **Explicit disclaimers:** Research boundaries and non-diagnostic statements appear on every relevant page
- **Methodology documentation:** Guide & Methodology page explains schema, header normalization, preprocessing, model terms, and safe interpretation boundaries
- **Reproducible workflow:** Built-in demo flow (5 min) for presentations and jury walkthroughs

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
feature names used by `data/demo/qurenova_demo_biomedical.csv`, and optionally a
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
