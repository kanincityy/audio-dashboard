import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Response

from audio_dashboard import asr, audio_io, registry

from . import store
from .files import UPLOAD_DIR
from .schemas import RunIn, RunOut, RunRecord

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1")


def _execute_run(path: Path, run_id: str):
    record = store.get_run(run_id)
    record.status = "running"
    store.save_run(record)
    try:
        bundle = audio_io.load(path)
        results = registry.run(
            bundle, record.digest, record.requested, force=record.force
        )
    except Exception as exc:
        record.status = "failed"
        record.error = f"Run failed ({type(exc).__name__}). Check the server log."
        logger.exception("Run %s failed", run_id)
        store.save_run(record)
        return
    record.results = results
    record.ran = list(results)
    record.status = "done"
    store.save_run(record)
    return record


@router.post("/files/{digest}/runs", response_model=RunOut, status_code=202)
def create_run(
    digest: str, body: RunIn, response: Response, background_tasks: BackgroundTasks
):
    matches = list(UPLOAD_DIR.glob(f"{digest}.*"))
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No uploaded file with digest {digest}. Upload it with POST /v1/files first."
            ),
        )
    path = matches[0]
    unknown = [name for name in body.analyses if name not in registry.BY_NAME]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown analysis: {', '.join(unknown)}. "
                "See GET /v1/analyses for valid names."
            ),
        )
    order = registry.resolve(body.analyses)
    would_bill = "transcript" in order and not asr.is_cached(
        path, asr.TranscribeConfig()
    )
    if would_bill and not body.allow_billing:
        raise HTTPException(
            status_code=403,
            detail=(
                "transcript would be billed for this file; set allow_billing to run it."
            ),
        )
    run_id = uuid.uuid4().hex
    record = RunRecord(
        run_id=run_id,
        digest=digest,
        ran=[],
        results={},
        created_at=datetime.now(timezone.utc),
        requested=body.analyses,
        force=body.force,
        allow_billing=body.allow_billing,
        status="queued",
    )
    store.save_run(record)
    background_tasks.add_task(_execute_run, path, run_id)
    results_url = f"/v1/runs/{run_id}"
    response.headers["Location"] = results_url
    return RunOut(run_id=run_id, ran=order, results_url=results_url)


@router.get("/runs/{run_id}", response_model=RunRecord)
def get_run(run_id: str):
    record = store.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No run with ID {run_id} found.")
    return record
