# QureNova Directory Completion Status

## Completed

- `domain/` — stable experiment and dataset contracts
- `schemas/` — JSON-safe payload helpers and validation
- `models/registry.py` — centralized model registry
- `evaluation/` — evaluation protocols, fair split helper, model utility ranking
- `experiments/` — framework-neutral experiment entry point
- `artifacts/` — atomic file-based artifact store
- `infrastructure/` — optional SQLite experiment index
- `security/` — CSV/size policy, direct-identifier detection, audit events
- `api/` — framework-neutral future API service boundary
- `datasets/` — root-level catalog facade with backward compatibility

## Deliberately not activated yet

- Full common-test-set benchmark integration into the legacy experiment engine
- Multi-seed execution across every model
- Full FastAPI server
- Persistent patient database
- Real-QPU execution as a default workflow
- Mixture-of-experts / learned meta-selector

These remain explicit extension points rather than pretending they already exist.
