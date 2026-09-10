"""Experiment artifact persistence.

Default storage is file based. A SQLite adapter is provided separately for
teams that want indexed/local database storage without changing the domain API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schemas.experiment import json_safe, validate_experiment_payload


class ArtifactStore:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or Path(__file__).resolve().parents[1] / "results" / "experiments")
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, experiment_id: str) -> Path:
        safe_id = "".join(ch for ch in str(experiment_id) if ch.isalnum() or ch in "-_.")
        if not safe_id:
            raise ValueError("Invalid experiment_id")
        return self.root / f"{safe_id}.json"

    def save(self, payload: dict[str, Any]) -> Path:
        errors = validate_experiment_payload(payload)
        if errors:
            raise ValueError("; ".join(errors))
        path = self.path_for(payload["experiment_id"])
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        return path

    def load(self, experiment_id: str) -> dict[str, Any]:
        path = self.path_for(experiment_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def list_ids(self) -> list[str]:
        return [path.stem for path in sorted(self.root.glob("*.json"), reverse=True)]
