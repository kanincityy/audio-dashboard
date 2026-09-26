from fastapi import APIRouter, HTTPException

from .files import UPLOAD_DIR

router = APIRouter(prefix="/v1")


@router.post("/files/{digest}/runs")
def create_run(digest: str):
    return {"digest": digest}
