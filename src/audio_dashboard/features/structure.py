"""Section 3 of notes.md: speech structure and temporal dynamics.

How the content is spoken, rather than how it was recorded. These are the
metrics behind the hallucination failure modes: long silences make
autoregressive models loop, short clips make them invent, and fast speech makes
them skip.

Voice activity detection underpins most of this and is the slowest step in the
whole dashboard, so it is its own analysis rather than something each metric
does privately.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import asr
from ..audio_io import VAD_SR, AudioBundle

# Speech starting or ending within this of a file boundary suggests the
# recording was cut mid-utterance rather than around it. notes.md:25.
EDGE_TOLERANCE_S = 0.1
SHORT_FILE_S = 2.0
SHORT_WORD_COUNT = 6

_MODEL = None


def _model():
    """Load Silero once per process.

    The package ships the weights, so this is a local file read rather than a
    download, but it still costs enough that reloading it per file would show.
    """
    global _MODEL
    if _MODEL is None:
        from silero_vad import load_silero_vad

        _MODEL = load_silero_vad()
    return _MODEL


def transcript(bundle: AudioBundle, results: dict[str, Any]) -> dict[str, Any]:
    """Diarised transcript with per-word confidence.

    Thin wrapper over the vendored ``asr`` module, which keeps its own cache
    keyed on file content — so this bills the API once per file, ever.
    """
    return asr.transcribe(bundle.path)


def vad(bundle: AudioBundle, results: dict[str, Any]) -> dict[str, Any]:
    """Speech spans in seconds, from Silero.

    Runs on the 16 kHz copy because that is what the model was trained on.
    Spans come back in seconds rather than sample indices so that consumers
    working at the native rate do not have to know that.
    """
    import torch
    from silero_vad import get_speech_timestamps

    audio = torch.from_numpy(np.ascontiguousarray(bundle.samples_16k))
    stamps = get_speech_timestamps(audio, _model(), sampling_rate=VAD_SR)

    spans = [[s["start"] / VAD_SR, s["end"] / VAD_SR] for s in stamps]
    speech_s = float(sum(end - start for start, end in spans))

    return {
        "speech_spans": [[round(a, 3), round(b, 3)] for a, b in spans],
        "segment_count": len(spans),
        "speech_seconds": round(speech_s, 3),
        "duration_s": round(bundle.duration, 3),
        "sampling_rate": VAD_SR,
    }


def density(bundle: AudioBundle, results: dict[str, Any]) -> dict[str, Any]:
    """How much of the file is speech, and where the silence sits."""
    v = results.get("vad") or {}
    spans = v.get("speech_spans") or []
    duration = bundle.duration

    if not duration:
        return {"usable": False, "reason": "zero-length audio"}

    speech_s = float(v.get("speech_seconds", 0.0))
    silence_s = max(0.0, duration - speech_s)

    if spans:
        leading = spans[0][0]
        trailing = duration - spans[-1][1]
        # Gaps between segments, excluding the head and tail, which are
        # reported separately because they mean something different.
        gaps = [spans[i + 1][0] - spans[i][1] for i in range(len(spans) - 1)]
        longest_gap = max(gaps) if gaps else 0.0
        segment_lengths = [b - a for a, b in spans]
    else:
        leading = trailing = duration
        longest_gap = duration
        segment_lengths = []

    silence_ratio = silence_s / duration

    return {
        "usable": True,
        "speech_seconds": round(speech_s, 3),
        "silence_seconds": round(silence_s, 3),
        "speech_ratio": round(speech_s / duration, 4),
        "silence_ratio": round(silence_ratio, 4),
        "segment_count": len(spans),
        "leading_silence_s": round(leading, 3),
        "trailing_silence_s": round(trailing, 3),
        "longest_internal_gap_s": round(longest_gap, 3),
        "median_segment_s": round(float(np.median(segment_lengths)), 3)
        if segment_lengths
        else None,
        "sparse_speech": bool(silence_ratio > 0.7),
    }


def truncation(bundle: AudioBundle, results: dict[str, Any]) -> dict[str, Any]:
    """Flags the short and clipped-at-the-edges cases.

    notes.md:26 puts this class at the highest hallucination rate of anything
    in the list, so it is worth flagging loudly even though the check is
    trivial: speech that touches either end of the file was probably cut.
    """
    v = results.get("vad") or {}
    spans = v.get("speech_spans") or []
    tr = results.get("transcript") or {}
    words = tr.get("words") or []

    starts_at_edge = bool(spans) and spans[0][0] <= EDGE_TOLERANCE_S
    ends_at_edge = bool(spans) and (bundle.duration - spans[-1][1]) <= EDGE_TOLERANCE_S

    word_count = len(words)
    few_words = bool(tr) and word_count <= SHORT_WORD_COUNT

    return {
        "duration_s": round(bundle.duration, 3),
        "short_file": bool(bundle.duration < SHORT_FILE_S),
        "speech_at_start": starts_at_edge,
        "speech_at_end": ends_at_edge,
        "word_count": word_count if tr else None,
        "few_words": few_words if tr else None,
        "hallucination_risk": bool(
            bundle.duration < SHORT_FILE_S or starts_at_edge or ends_at_edge or few_words
        ),
    }


def tempo(bundle: AudioBundle, results: dict[str, Any]) -> dict[str, Any]:
    """Words per minute, measured against speech time rather than file time.

    A file that is half silence is not being spoken at half speed, so the
    denominator is VAD speech seconds. The per-utterance spread matters more
    than the average — one fast passage is what causes deletions, and it
    disappears into a mean.
    """
    tr = results.get("transcript") or {}
    v = results.get("vad") or {}
    words = tr.get("words") or []
    speech_s = float(v.get("speech_seconds", 0.0))

    if not words or speech_s <= 0:
        return {"usable": False, "reason": "needs both a transcript and speech time"}

    per_utterance = []
    for utt in tr.get("utterances") or []:
        span = float(utt.get("end", 0.0)) - float(utt.get("start", 0.0))
        n = len(utt.get("words") or [])
        if span > 0.5 and n >= 3:
            per_utterance.append(60.0 * n / span)

    result = {
        "usable": True,
        "wpm": round(60.0 * len(words) / speech_s, 1),
        "word_count": len(words),
        "speech_seconds": round(speech_s, 3),
        "wpm_per_utterance": [round(x, 1) for x in per_utterance],
    }
    if per_utterance:
        arr = np.array(per_utterance)
        result["wpm_median"] = round(float(np.median(arr)), 1)
        result["wpm_p90"] = round(float(np.percentile(arr, 90)), 1)
        result["fast_utterance_count"] = int((arr > 200).sum())
    return result


def _merge(spans: list[list[float]]) -> list[list[float]]:
    """Union of possibly-overlapping spans."""
    merged: list[list[float]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def overlap(bundle: AudioBundle, results: dict[str, Any]) -> dict[str, Any]:
    """Time where two diarised speakers are talking at once.

    On mono audio this does not work, and that is measured rather than
    suspected. Scored against the synthetic corpus, where the true overlap is
    known from the manifest, it returned 0.000 s on all eleven files that had
    real overlap in them — up to 8.9 s of it. Mono diarisation assigns every
    moment to exactly one speaker, so there is never an overlapping utterance
    to find. A mono zero is therefore reported as unusable, not as a pass: a
    call full of cross-talk would otherwise clear the routing rule.

    Per-channel audio is the fix. Where each speaker has their own channel,
    diarisation is trivial and the overlaps are real.
    """
    tr = results.get("transcript") or {}
    utterances = tr.get("utterances") or []
    duration = bundle.duration

    if not utterances or not duration:
        return {"usable": False, "reason": "needs a diarised transcript"}

    spans = [
        [float(u["start"]), float(u["end"])]
        for u in utterances
        if u.get("end") is not None and float(u["end"]) > float(u["start"])
    ]

    # Sweep the endpoints and count how many utterances are live at each point.
    events: list[tuple[float, int]] = []
    for start, end in spans:
        events.append((start, 1))
        events.append((end, -1))
    events.sort()

    live = 0
    prev = 0.0
    overlap_s = 0.0
    overlap_spans: list[list[float]] = []
    for time, delta in events:
        if live >= 2 and time > prev:
            overlap_s += time - prev
            overlap_spans.append([round(prev, 3), round(time, 3)])
        live += delta
        prev = time

    mono = bundle.channels is None

    result = {
        # Measured against the synthetic corpus, where the true overlap is
        # known from where each turn was placed: on mono audio this comes back
        # as exactly 0.000 s on every file, including one with 8.9 s of genuine
        # overlap. Not merely an underestimate — the diariser never emits an
        # overlapping utterance at all, so there is nothing here to sweep. A
        # zero on mono is therefore "not measured", and saying otherwise would
        # turn the routing rule into a false pass on every call with cross-talk.
        "usable": not (mono and overlap_s == 0.0),
        "overlap_seconds": round(overlap_s, 3),
        "overlap_ratio": round(overlap_s / duration, 4),
        "overlap_spans": _merge(overlap_spans),
        "high_overlap": bool(overlap_s / duration > 0.15),
        "underreports_on_mono": True,
        "mono": mono,
    }
    if not result["usable"]:
        result["reason"] = (
            "mono diarisation assigns every moment to exactly one speaker, so "
            "it reports no overlap whether or not there is any. Measured 0.000 s "
            "on every file in the synthetic corpus, including ones built with "
            "real overlap. Needs per-channel audio or a diariser that models "
            "overlapping speech."
        )

    # Cross-check: VAD speech that no utterance covers. Kept because it is the
    # only independent number available here, but it did not discriminate on
    # the synthetic corpus either — a median of 2.27% on files built with real
    # overlap against 2.54% on files built with none, which is the wrong way
    # round. Read it as "speech the diariser dropped", not as evidence of
    # cross-talk.
    vad_spans = (results.get("vad") or {}).get("speech_spans")
    if vad_spans:
        covered = _merge([list(s) for s in spans])
        uncovered = 0.0
        for start, end in vad_spans:
            remaining = end - start
            for c_start, c_end in covered:
                if c_end <= start or c_start >= end:
                    continue
                remaining -= min(end, c_end) - max(start, c_start)
            uncovered += max(0.0, remaining)
        result["unattributed_speech_s"] = round(uncovered, 3)
        result["unattributed_speech_ratio"] = round(uncovered / duration, 4)

    return result
