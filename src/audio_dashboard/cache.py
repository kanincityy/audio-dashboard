"""On-disk cache for analysis results, keyed by file content.

Follows the same shape as the transcript cache in ``asr.py``: hash the file,
not its name, so re-uploading the same audio under a different filename is a
hit rather than a recompute.

The one addition is a version integer per analysis. Bumping it in the registry
invalidates that analysis alone, which matters because these extractors are
going to be tuned — changing the SNR estimator should not throw away every
spectrum and pitch track alongside it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "features"


def file_digest(path: Path) -> str:
    """First 16 hex chars of the file's sha256. Matches asr._file_digest."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()[:16]


def variant_key(satisfied: Sequence[str]) -> str:
    """Key fragment naming which optional inputs were available.

    Analyses that merely *prefer* an input still produce a different answer
    with it than without — pitch gains a per-speaker breakdown once there is a
    transcript, and signal-to-noise stops guessing once there is voice
    activity. Without this in the key, a run that has those inputs is served
    the earlier, worse answer from a run that did not.
    """
    return "+" + "+".join(sorted(satisfied)) if satisfied else ""


def cache_path(
    digest: str, analysis: str, version: int, satisfied: Sequence[str] = ()
) -> Path:
    return CACHE_DIR / f"{digest}-{analysis}-v{version}{variant_key(satisfied)}.json"


def read(
    digest: str, analysis: str, version: int, satisfied: Sequence[str] = ()
) -> dict[str, Any] | None:
    dest = cache_path(digest, analysis, version, satisfied)
    if not dest.exists():
        return None
    try:
        return json.loads(dest.read_text())
    except json.JSONDecodeError:
        # A truncated write from an interrupted run. Treat it as a miss rather
        # than crashing the app; the recompute will overwrite it.
        return None


def write(
    digest: str,
    analysis: str,
    version: int,
    result: dict[str, Any],
    satisfied: Sequence[str] = (),
) -> None:
    dest = cache_path(digest, analysis, version, satisfied)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2))
