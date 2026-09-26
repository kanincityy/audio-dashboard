"""The catalogue of analyses, and the runner that resolves and executes them.

Everything the dashboard can compute is registered here as an ``Analysis``.
The UI builds its checkbox list from this, the runner orders the selection by
dependency, and the cache keys off each entry's name and version. Adding a new
metric means writing an extractor and adding one line to ``ANALYSES`` — nothing
in the UI has to know about it specifically.

Extractors take the bundle and the results of everything already run, so a
metric that needs voice activity reads it out of that dict rather than
recomputing it. That is what keeps VAD, which is by far the slowest step, to
one run per file even though four metrics depend on it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import cache
from .audio_io import AudioBundle
from .features import formatmeta, quality, speaker, structure

# What running an analysis costs the user. "fast" is sub-second, "slow" means
# a model or a full-file transform, "paid" means it bills an API per call.
Cost = str


@dataclass(frozen=True)
class Analysis:
    name: str
    group: str
    label: str
    fn: Callable[[AudioBundle, dict[str, Any]], dict[str, Any]]
    note: str = ""
    requires: tuple[str, ...] = ()
    # Analyses this one uses when they are present but can do without. They
    # are never pulled in on its behalf; they only constrain ordering, so that
    # a better answer is used whenever the user happened to ask for both.
    prefers: tuple[str, ...] = ()
    cost: Cost = "fast"
    version: int = 1
    # Transcription maintains its own cache in asr.py, so caching the same
    # payload a second time here would only waste disk.
    cached: bool = True


# Group labels, in the order notes.md presents them.
GROUPS: list[tuple[str, str]] = [
    ("format", "1. Format and technical metadata"),
    ("quality", "2. Signal quality and acoustic metrics"),
    ("structure", "3. Speech structure and temporal dynamics"),
    ("speaker", "4. Speaker and demographic profiling"),
    ("transcript", "Transcription"),
]


ANALYSES: list[Analysis] = [
    Analysis(
        name="format",
        group="format",
        label="Format and codec",
        fn=formatmeta.extract,
        note="Sample rate, channels, codec, bitrate. Reads the container only.",
    ),
    Analysis(
        name="transcript",
        group="transcript",
        label="Transcribe (AssemblyAI)",
        fn=structure.transcript,
        note="Diarised transcript with per-word confidence. Cached per file.",
        cost="paid",
        cached=False,
    ),
    Analysis(
        name="levels",
        group="quality",
        label="Levels and dynamic range",
        fn=quality.levels,
        note="Peak and RMS dBFS, crest factor, dynamic range.",
    ),
    Analysis(
        name="clipping",
        group="quality",
        label="Clipping and distortion",
        fn=quality.clipping,
        note="Share of samples at full scale, and runs of flattened waveform.",
    ),
    Analysis(
        name="bandwidth",
        group="quality",
        label="Bandwidth and upsampling",
        fn=quality.bandwidth,
        note="Effective spectral cutoff, and whether the top half is empty.",
        version=2,
    ),
    Analysis(
        name="vad",
        group="structure",
        label="Voice activity detection",
        fn=structure.vad,
        note="Silero VAD. Needed by SNR, reverberation, pitch and density.",
        cost="slow",
    ),
    Analysis(
        name="snr",
        group="quality",
        label="Signal-to-noise ratio",
        fn=quality.snr,
        note="Speech level against noise floor. Far better with VAD selected.",
        prefers=("vad",),
    ),
    Analysis(
        name="rt60",
        group="quality",
        label="Reverberation time (RT60)",
        fn=quality.rt60,
        note="Blind estimate from decay after speech offsets. Approximate.",
        requires=("vad",),
        cost="slow",
    ),
    Analysis(
        name="density",
        group="structure",
        label="Speech density and silence",
        fn=structure.density,
        note="Speech-to-silence ratio, gaps, leading and trailing silence.",
        requires=("vad",),
    ),
    Analysis(
        name="truncation",
        group="structure",
        label="Utterance length and truncation",
        fn=structure.truncation,
        note="Flags short files and speech touching either end of the file.",
        requires=("vad",),
        prefers=("transcript",),
    ),
    Analysis(
        name="tempo",
        group="structure",
        label="Speech tempo (words per minute)",
        fn=structure.tempo,
        note="Words per minute of speech, not per minute of file.",
        requires=("transcript", "vad"),
    ),
    Analysis(
        name="overlap",
        group="structure",
        label="Overlapping speech",
        fn=structure.overlap,
        note="Cross-talk between diarised speakers. Does not work on mono "
        "audio, and says so rather than reporting zero.",
        requires=("transcript",),
        prefers=("vad",),
        version=2,
    ),
    Analysis(
        name="speakers",
        group="speaker",
        label="Speaker count",
        fn=speaker.speakers,
        note="Distinct speakers found by diarisation.",
        requires=("transcript",),
    ),
    Analysis(
        name="f0",
        group="speaker",
        label="Pitch (F0) and range",
        fn=speaker.f0,
        note="Median and spread of fundamental frequency over speech.",
        requires=("vad",),
        prefers=("transcript",),
        cost="slow",
    ),
    Analysis(
        name="codeswitch",
        group="speaker",
        label="Code-switching and dialect",
        fn=speaker.codeswitch,
        note="Not implemented yet. Deferred to a later version.",
    ),
]

BY_NAME: dict[str, Analysis] = {a.name: a for a in ANALYSES}


def resolve(selected: Iterable[str]) -> list[str]:
    """Order a selection so every analysis runs after its requirements.

    Requirements are pulled in whether or not the caller asked for them — an
    analysis cannot run without them. The UI is expected to show which extras
    a tick implies rather than let them appear from nowhere.
    """
    chosen = set(selected)
    ordered: list[str] = []
    placed: set[str] = set()
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in placed:
            return
        if name in visiting:
            raise ValueError(f"circular requirement involving {name!r}")
        if name not in BY_NAME:
            raise KeyError(f"unknown analysis: {name!r}")
        visiting.add(name)
        analysis = BY_NAME[name]
        for req in analysis.requires:
            visit(req)
        # A preference only orders the run; it never adds to it.
        for pref in analysis.prefers:
            if pref in chosen:
                visit(pref)
        visiting.discard(name)
        ordered.append(name)
        placed.add(name)

    # Sorted so the resolved order is stable regardless of tick order, which
    # keeps the progress display predictable.
    for name in sorted(chosen):
        visit(name)
    return ordered


def implied(selected: Iterable[str]) -> list[str]:
    """The extra analyses a selection drags in. For telling the user."""
    chosen = set(selected)
    return [name for name in resolve(chosen) if name not in chosen]


def jsonable(value: Any) -> Any:
    """Coerce numpy scalars, arrays and non-finite floats for json.dump.

    Non-finite values become None rather than raising or writing ``NaN``,
    which is not valid JSON. A missing metric reads as missing in the UI,
    which is the honest rendering of "this could not be measured here".
    """
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def run(
    bundle: AudioBundle,
    digest: str,
    selected: Sequence[str],
    *,
    on_start: Callable[[str, Analysis], None] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run a selection in dependency order, using the cache where possible.

    Returns one entry per analysis, keyed by name. A failing extractor records
    its error under that key instead of aborting the run: one broken metric
    should not cost you the other twelve, and the UI can show which failed.
    """
    results: dict[str, Any] = {}

    for name in resolve(selected):
        analysis = BY_NAME[name]
        if on_start is not None:
            on_start(name, analysis)

        # Which optional inputs this run actually has. It belongs in the cache
        # key: the same analysis at the same version gives a different — and
        # better — answer when a preferred input is present, and serving the
        # earlier answer would silently throw that away.
        satisfied = tuple(p for p in analysis.prefers if p in results)

        if analysis.cached and not force:
            hit = cache.read(digest, name, analysis.version, satisfied)
            if hit is not None:
                results[name] = hit
                continue

        try:
            result = jsonable(analysis.fn(bundle, results))
        except Exception as exc:  # noqa: BLE001 - surfaced in the UI
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue

        if analysis.cached:
            cache.write(digest, name, analysis.version, result, satisfied)
        results[name] = result

    return results
