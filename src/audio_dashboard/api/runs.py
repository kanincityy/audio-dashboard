import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Response

from audio_dashboard import asr, audio_io, registry

from . import store
from .files import UPLOAD_DIR
from .schemas import RunIn, RunOut, RunRecord

router = APIRouter(prefix="/v1")


@router.post("/files/{digest}/runs", response_model=RunOut, status_code=201)
def create_run(digest: str, body: RunIn, response: Response):
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
    bundle = audio_io.load(path)
    results = registry.run(bundle, digest, body.analyses, force=body.force)

    run_id = uuid.uuid4().hex
    record = RunRecord(
        run_id=run_id,
        digest=digest,
        ran=list(results),
        results=results,
        created_at=datetime.now(timezone.utc),
        requested=body.analyses,
        force=body.force,
        allow_billing=body.allow_billing,
        status="done"
    )

    store.save_run(record)
    results_url = f"/v1/runs/{run_id}"
    response.headers["Location"] = results_url
    return RunOut(run_id=run_id, ran=list(results), results_url=results_url)


@router.get("/runs/{run_id}", response_model=RunRecord)
def get_run(run_id: str):
    record = store.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No run with ID {run_id} found.")
    return record
