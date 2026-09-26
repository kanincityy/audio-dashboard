import hashlib
from pathlib import Path

from fastapi import APIRouter, HTTPException, Response, UploadFile

from audio_dashboard import audio_io

from .schemas import FileOut

UPLOAD_DIR = Path(__file__).resolve().parents[3] / "uploads"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


router = APIRouter(prefix="/v1")


def _reject_if_too_large(size_bytes: int) -> None:
    """Raise 413 if an upload exceeds the limit. One message, one place."""
    if size_bytes > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File is {size_bytes / 1024 / 1024:.1f} MB; "
                f"the limit is {MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB."
            ),
        )


@router.post("/files", response_model=FileOut, status_code=201)
async def upload_file(file: UploadFile, response: Response):
    if file.filename is None:
        raise HTTPException(
            status_code=400,
            detail="No filename in upload. Filename needed to determine file format.",
        )
    suffix = Path(file.filename).suffix.lower()
    if suffix not in audio_io.AUDIO_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type. Accepted: {', '.join(sorted(audio_io.AUDIO_SUFFIXES))}",
        )
    # Checked twice on purpose. file.size is what the client claims, and it is
    # checked first so an oversized upload is refused before it is read into
    # memory. It can be absent or wrong, so len(data) is checked after the read
    # as the measured truth.
    if file.size is not None:
        _reject_if_too_large(file.size)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    data = await file.read()
    _reject_if_too_large(len(data))
    digest = hashlib.sha256(data).hexdigest()[:16]
    dest = UPLOAD_DIR / f"{digest}{suffix}"
    existed = dest.exists()
    if not existed:
        dest.write_bytes(data)
    response.status_code = 200 if existed else 201
    return FileOut(digest=digest, size_bytes=len(data), filename=file.filename)
