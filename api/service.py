"""Framework-neutral service interface for future HTTP APIs."""

from __future__ import annotations

from experiments.orchestrator import execute


def run_experiment_service(dataset, target_column: str, config: dict | None = None):
    """Return the same domain result used by the Streamlit UI.

    A future FastAPI layer can call this function without moving model logic
    into the web framework.
    """
    result, experiment_id = execute(dataset, target_column, config, persist=True)
    return {"experiment_id": experiment_id, "result": result}
