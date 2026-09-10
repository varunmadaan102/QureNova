"""Optional SQLite persistence for experiment metadata.

SQLite is intentionally standard-library only; it is a local indexing option,
not a requirement for the SIH deployment.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from schemas.experiment import json_safe


class SQLiteExperimentStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or Path(__file__).resolve().parents[1] / "results" / "qurenova.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    dataset_profile TEXT,
                    target_column TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )

    def upsert(self, payload: dict[str, Any]) -> None:
        safe = json_safe(payload)
        metadata = safe.get("metadata", {}) if isinstance(safe, dict) else {}
        dataset_profile = metadata.get("dataset_profile")
        target_column = metadata.get("target_column")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO experiments(experiment_id, created_at, dataset_profile, target_column, payload_json)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(experiment_id) DO UPDATE SET
                    created_at=excluded.created_at,
                    dataset_profile=excluded.dataset_profile,
                    target_column=excluded.target_column,
                    payload_json=excluded.payload_json
                """,
                (
                    safe.get("experiment_id"),
                    safe.get("created_at", ""),
                    dataset_profile,
                    target_column,
                    json.dumps(safe, sort_keys=True),
                ),
            )

    def get(self, experiment_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload_json FROM experiments WHERE experiment_id=?",
                (experiment_id,),
            ).fetchone()
        if row is None:
            raise KeyError(experiment_id)
        return json.loads(row[0])
