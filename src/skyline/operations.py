"""Durable local review state for a nightly Skyline ranking."""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional


VALID_DECISIONS = {"follow_up", "watch", "dismiss"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def snapshot_id(paths: Iterable[str], budget: int, model_label: str) -> str:
    """Stable ID for an immutable set of input files and ranking settings."""
    digest = hashlib.sha256()
    for item in paths:
        path = Path(item)
        digest.update(str(path.resolve()).encode())
        digest.update(path.read_bytes())
    digest.update(str(budget).encode())
    digest.update(model_label.encode())
    return digest.hexdigest()[:16]


class ReviewStore:
    def __init__(self, database_path: str):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS snapshot (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    model_label TEXT NOT NULL,
                    object_count INTEGER NOT NULL,
                    budget INTEGER NOT NULL,
                    ranked_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS review (
                    run_id TEXT NOT NULL,
                    object_id TEXT NOT NULL,
                    decision TEXT NOT NULL CHECK (decision IN ('follow_up', 'watch', 'dismiss')),
                    note TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, object_id),
                    FOREIGN KEY (run_id) REFERENCES snapshot(run_id)
                );
                """
            )

    def save_snapshot(self, run_id: str, model_label: str, object_count: int, budget: int, ranked: List[Dict[str, object]]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO snapshot(run_id, created_at, model_label, object_count, budget, ranked_json) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, utc_now(), model_label, object_count, budget, json.dumps(ranked, separators=(",", ":"))),
            )

    def get_snapshot(self, run_id: str) -> Optional[Dict[str, object]]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM snapshot WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return {**dict(row), "ranked": json.loads(row["ranked_json"])}

    def decisions(self, run_id: str) -> Dict[str, Dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT object_id, decision, note, updated_at FROM review WHERE run_id = ?", (run_id,)).fetchall()
        return {row["object_id"]: {"decision": row["decision"], "note": row["note"], "updated_at": row["updated_at"]} for row in rows}

    def record_review(self, run_id: str, object_id: str, decision: str, note: str = "") -> Dict[str, str]:
        if decision not in VALID_DECISIONS:
            raise ValueError(f"decision must be one of {sorted(VALID_DECISIONS)}")
        snapshot = self.get_snapshot(run_id)
        if snapshot is None:
            raise LookupError("unknown ranking snapshot")
        if object_id not in {item["object_id"] for item in snapshot["ranked"]}:
            raise LookupError("object does not belong to this ranking snapshot")
        record = {"decision": decision, "note": note.strip(), "updated_at": utc_now()}
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO review(run_id, object_id, decision, note, updated_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id, object_id) DO UPDATE SET decision=excluded.decision, note=excluded.note, updated_at=excluded.updated_at",
                (run_id, object_id, record["decision"], record["note"], record["updated_at"]),
            )
        return record
