from fastapi import APIRouter, HTTPException

from audio_dashboard import registry

from .files import UPLOAD_DIR
from .schemas import RunIn

router = APIRouter(prefix="/v1")


@router.post("/files/{digest}/runs")
def create_run(digest: str, body: RunIn):
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

    return {"digest": digest, "path": str(path)}
