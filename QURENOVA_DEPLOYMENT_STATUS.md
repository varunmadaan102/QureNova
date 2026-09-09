# QureNova Deployment Status

## Current state

- Repository: `https://github.com/varunmadaan102/QureNova`
- Branch: `main`
- Latest known published commit: `1e22517`
- Product branding in source: QureNova
- Local research paper: `QureNova-Research-Paper.docx`
- Local checkout folder: still `QURE-AI-v2` because the active workspace locks
  the directory name

## Required redeployment action

The hosted screenshot showed `qurenova.streamlit.app` rendering the older
QureAI build even though GitHub `main` contains QureNova. Trigger a Streamlit
Cloud reboot/redeploy from `varunmadaan102/QureNova` and verify:

- browser title is QureNova;
- sidebar brand says QureNova;
- Overview copy says QureNova;
- Data Workspace loads `qurenova_demo_biomedical.csv`;
- requirements install Qiskit, Qiskit Machine Learning, XGBoost, and SHAP;
- the quantum control is available when imports succeed;
- the app health endpoint returns `ok`.

## Local validation already completed

- 9 automated tests pass.
- Python compilation passes.
- Streamlit AppTest loads with zero exceptions.
- Streamlit health endpoint returned HTTP 200.
- Tuned persisted model artifacts were regenerated locally.

## Privacy and artifact rules

Keep the following local/ignored:

- `QureNova-Research-Paper.docx`
- generated `results/models/*.joblib`
- private reports, uploads, and identifiable data

Never upload identifiable health information, patient records, secrets, or
private research documents to the public repository or a public demo deployment.
