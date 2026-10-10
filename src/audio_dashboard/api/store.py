import sqlite3
from pathlib import Path

from .schemas import RunRecord

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "runs.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    digest TEXT NOT NULL,
    ran TEXT NOT NULL,
    results TEXT NOT NULL,
    created_at TEXT NOT NULL,
    requested TEXT NOT NULL,
    force INTEGER NOT NULL,
    allow_billing INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'done', 'failed')),
    error TEXT,
    current TEXT,
    completed INTEGER NOT NULL,
    total INTEGER NOT NULL
);
"""


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    conn.close()


_runs: dict[str, RunRecord] = {}


def save_run(record: RunRecord) -> None:
    _runs[record.run_id] = record


def get_run(run_id: str) -> RunRecord | None:
    return _runs.get(run_id)
