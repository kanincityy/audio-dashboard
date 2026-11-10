import hashlib
import mimetypes
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Response, UploadFile
from fastapi.responses import FileResponse

from audio_dashboard import audio_io, waveform

from .schemas import FileOut, WaveformOut

UPLOAD_DIR = Path(__file__).resolve().parents[3] / "uploads"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


router = APIRouter(prefix="/v1")


def require_upload(digest: str) -> list[Path]:
    """The stored files for a digest, or a 404 that says how to get one."""
    matches = list(UPLOAD_DIR.glob(f"{digest}.*"))
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No uploaded file with digest {digest}. Upload it with POST /v1/files first."
            ),
        )
    return matches


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


@router.get("/files/{digest}/audio")
def get_audio(digest: str):
    """The uploaded audio itself, for a browser <audio> element."""
    path = require_upload(digest)[0]
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type)


@router.get("/files/{digest}/waveform", response_model=WaveformOut)
def get_waveform(digest: str, points: int = Query(4000, ge=10, le=20000)):
    """Min and max per bucket, enough to draw the waveform chart."""
    path = require_upload(digest)[0]
    try:
        bundle = audio_io.load(path)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not decode this file ({type(exc).__name__}).",
        ) from exc
    outline = waveform.envelope(bundle.samples, points)
    return WaveformOut(duration_s=bundle.duration, sample_rate=bundle.sr, **outline)
