import json
import sqlite3
from datetime import datetime
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

_SAVE = """
INSERT OR REPLACE INTO runs (
    run_id, digest, ran, results, created_at, requested, force, allow_billing, status, error, current, completed, total
) VALUES (
    :run_id, :digest, :ran, :results, :created_at, :requested, :force, :allow_billing, :status, :error, :current, :completed, :total
);"""

_GET = """
SELECT * FROM runs WHERE run_id = :run_id;"""


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    conn.close()


def save_run(record: RunRecord) -> None:
    values = record.model_dump()
    values["ran"] = json.dumps(record.ran)
    values["results"] = json.dumps(record.results)
    values["requested"] = json.dumps(record.requested)
    values["created_at"] = record.created_at.isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SAVE, values)
    conn.commit()
    conn.close()


def get_run(run_id: str) -> RunRecord | None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(_GET, {"run_id": run_id})
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    values = dict(row)
    values["ran"] = json.loads(values["ran"])
    values["results"] = json.loads(values["results"])
    values["requested"] = json.loads(values["requested"])
    values["created_at"] = datetime.fromisoformat(values["created_at"])
    return RunRecord(**values)
