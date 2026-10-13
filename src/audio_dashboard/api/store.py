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

_CREATE_DIGEST_INDEX = "CREATE INDEX IF NOT EXISTS idx_digest ON runs (digest);"

_SAVE = """
INSERT OR REPLACE INTO runs (
    run_id, digest, ran, results, created_at, requested, force, allow_billing, status, error, current, completed, total
) VALUES (
    :run_id, :digest, :ran, :results, :created_at, :requested, :force, :allow_billing, :status, :error, :current, :completed, :total
);"""

_GET = """
SELECT * FROM runs WHERE run_id = :run_id;"""

_LIST_FOR_DIGEST = """
SELECT * FROM runs WHERE digest = :digest ORDER BY created_at DESC;"""

# A background run lives inside the server process, so a restart kills it
# mid-way and nothing is left to mark it finished. Without this it would say
# "queued" or "running" forever and a client polling it would never stop.
_FAIL_INTERRUPTED = """
UPDATE runs
SET status = 'failed', error = 'Server restarted before the run finished.', current = NULL
WHERE status IN ('queued', 'running');"""


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    conn.execute(_CREATE_DIGEST_INDEX)
    conn.commit()
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


def _row_to_record(row):
    values = dict(row)
    values["ran"] = json.loads(values["ran"])
    values["results"] = json.loads(values["results"])
    values["requested"] = json.loads(values["requested"])
    values["created_at"] = datetime.fromisoformat(values["created_at"])
    return RunRecord(**values)


def get_run(run_id: str) -> RunRecord | None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(_GET, {"run_id": run_id})
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return _row_to_record(row)


def get_runs(digest: str) -> list[RunRecord]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(_LIST_FOR_DIGEST, {"digest": digest})
    rows = cursor.fetchall()
    conn.close()

    return [_row_to_record(row) for row in rows]


def fail_interrupted_runs() -> int:
    """Mark runs cut off by a restart as failed. Returns how many there were."""
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute(_FAIL_INTERRUPTED).rowcount
    conn.commit()
    conn.close()
    return count
