# QureNova Architecture

QureNova is a Streamlit-first research platform with a framework-neutral application boundary.
The design intentionally separates **UI**, **application services**, **domain contracts**, **model/quantum engines**, **evaluation**, and **persistence/security**.

## Directory responsibilities

| Directory | Responsibility | UI-independent? |
|---|---|---|
| `app.py` | Streamlit entry point and page registration | No |
| `views/` | Page-level presentation and interaction | No |
| `components/` | Reusable Streamlit visual components | No |
| `services/` | Current application orchestration and feature services | Mostly |
| `domain/` | Stable domain contracts and experiment envelopes | Yes |
| `schemas/` | Serialization/validation of domain payloads | Yes |
| `models/` | Classical model definitions and registry | Yes |
| `quantum/` | Quantum feature maps, kernels and alignment | Yes |
| `core/` | Existing data/preprocessing/clinical/reliability primitives | Yes |
| `evaluation/` | Protocols, fair splits and model-selection heuristics | Yes |
| `experiments/` | Stable application-facing experiment entry point | Yes |
| `artifacts/` | File-based experiment artifact persistence | Yes |
| `infrastructure/` | Optional local SQLite persistence | Yes |
| `security/` | Upload, privacy and audit guardrails | Yes |
| `api/` | Future FastAPI/HTTP boundary | Yes |
| `datasets/` | Root-level dataset catalog facade | Yes |
| `data/` | Checked-in datasets and runtime data locations | N/A |
| `results/` | Generated artifacts, figures, kernels and experiment history | N/A |
| `tests/` | Unit, integration, regression, quantum and UI tests | Yes/No |

## Planned experiment lifecycle

`Dataset → explicit target semantics → validation → leakage-safe preprocessing → classical baselines / quantum experiments → evaluation → explainability → reliability → model-selection score → immutable artifact`

The current engine remains backwards compatible. The `experiments.orchestrator.execute()` boundary is the intended path for future API, database and background-job integrations.

## Benchmark rule

The current quantum path is intentionally bounded and uses a contextual holdout. It is **not** a direct comparison with cross-validation means. `evaluation.protocols.COMMON_FINAL_TEST` and `evaluation.splits.common_final_split()` establish the target architecture for the next benchmark revision: one untouched final test set shared by classical and quantum candidates.

## Medical/data-safety rule

QureNova may calculate a model **prediction/risk estimate** and **model confidence** for research workflows. It must not present experimental thresholds as clinical cutoffs, infer medical label meaning from numeric ordering, or claim clinical validity or quantum advantage without evidence.
