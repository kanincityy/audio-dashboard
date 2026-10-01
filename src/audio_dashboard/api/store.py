from .schemas import RunRecord

_runs: dict[str, RunRecord] = {}


def save_run(record: RunRecord) -> None:
    _runs[record.run_id] = record


def get_run(run_id: str) -> RunRecord | None:
    return _runs.get(run_id)
