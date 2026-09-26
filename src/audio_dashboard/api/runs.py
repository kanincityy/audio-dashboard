from fastapi import APIRouter, HTTPException

from .files import UPLOAD_DIR

router = APIRouter(prefix="/v1")


@router.post("/files/{digest}/runs")
def create_run(digest: str):
    matches = list(UPLOAD_DIR.glob(f"{digest}.*"))
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No uploaded file with digest {digest}. Upload it with POST /v1/files first."
            ),
        )
    path = matches[0]
    return {"digest": digest, "path": str(path)}
