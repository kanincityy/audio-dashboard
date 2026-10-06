"""Section 4 of notes.md: speaker and demographic profiling.

Acoustic properties of the voice rather than the recording. Speaker count is
free from diarisation; pitch is cheap DSP; dialect and code-switch detection
would need another model and is deferred.
"""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np

from ..audio_io import VAD_SR, AudioBundle
from ..thresholds import HIGH_PITCH_HZ

# Wide enough to cover a low voice through a child's, which is the range
# notes.md:36 cares about — pitch far above the adult-read-speech distribution
# is where word error rates spike.
F0_MIN_HZ = 50.0
F0_MAX_HZ = 500.0


def speakers(bundle: AudioBundle, results: dict[str, Any]) -> dict[str, Any]:
    tr = results.get("transcript") or {}
    if not tr:
        return {"usable": False, "reason": "needs a transcript"}

    utterances = tr.get("utterances") or []
    labels = sorted({u.get("speaker") for u in utterances if u.get("speaker")})
    count = int(tr.get("speaker_count") or len(labels))

    # Turns per speaker says something the count does not: one speaker holding
    # the floor with a second interjecting twice is a different routing case
    # from a balanced two-way call.
    turns: dict[str, int] = {}
    talk_time: dict[str, float] = {}
    for u in utterances:
        label = u.get("speaker")
        if not label:
            continue
        turns[label] = turns.get(label, 0) + 1
        talk_time[label] = talk_time.get(label, 0.0) + (
            float(u.get("end", 0.0)) - float(u.get("start", 0.0))
        )

    return {
        "usable": True,
        "speaker_count": count,
        "labels": labels,
        "turns_per_speaker": turns,
        "talk_time_s": {k: round(v, 2) for k, v in talk_time.items()},
        "single_speaker": count <= 1,
        "needs_diarisation": count > 1,
    }


def f0(bundle: AudioBundle, results: dict[str, Any]) -> dict[str, Any]:
    """Fundamental frequency over speech, via probabilistic YIN.

    Runs on the 16 kHz copy: pyin is the slowest thing in the dashboard and
    nothing in this pitch range needs more bandwidth than that. The track is
    computed over the whole file and then masked down to VAD speech, which
    keeps frame times aligned to the original timeline so per-speaker
    attribution stays possible.
    """
    v = results.get("vad") or {}
    spans = v.get("speech_spans") or []
    if not spans:
        return {"usable": False, "reason": "no speech spans; run VAD first"}

    hop = 256
    track, voiced, _ = librosa.pyin(
        bundle.samples_16k,
        fmin=F0_MIN_HZ,
        fmax=F0_MAX_HZ,
        sr=VAD_SR,
        hop_length=hop,
    )
    times = librosa.times_like(track, sr=VAD_SR, hop_length=hop)

    in_speech = np.zeros(len(times), dtype=bool)
    for start, end in spans:
        in_speech |= (times >= start) & (times <= end)

    usable = in_speech & voiced & np.isfinite(track)
    values = track[usable]

    if values.size < 5:
        return {
            "usable": False,
            "reason": "too few voiced frames inside speech to characterise pitch",
            "voiced_frames": int(values.size),
        }

    median = float(np.median(values))
    p25, p75 = (float(x) for x in np.percentile(values, [25, 75]))

    result = {
        "usable": True,
        "f0_median_hz": round(median, 1),
        "f0_p25_hz": round(p25, 1),
        "f0_p75_hz": round(p75, 1),
        "f0_iqr_hz": round(p75 - p25, 1),
        "f0_min_hz": round(float(values.min()), 1),
        "f0_max_hz": round(float(values.max()), 1),
        "voiced_fraction": round(float(usable.sum()) / max(1, int(in_speech.sum())), 3),
        "high_pitch": bool(median > HIGH_PITCH_HZ),
        # Thinned for plotting. Unvoiced frames stay as None so the UI draws
        # gaps rather than interpolating through them.
        "track_times_s": times[::4].round(3).tolist(),
        "track_hz": [
            round(float(x), 1) if m else None
            for x, m in zip(track[::4], usable[::4], strict=False)
        ],
    }

    # Per-speaker medians, which is the part that speaks to notes.md:36 — one
    # high-pitched speaker in a two-way call is a routing signal that a
    # file-wide median averages away.
    utterances = (results.get("transcript") or {}).get("utterances") or []
    if utterances:
        per_speaker: dict[str, list[float]] = {}
        for u in utterances:
            label = u.get("speaker")
            if not label:
                continue
            window = (times >= float(u.get("start", 0.0))) & (
                times <= float(u.get("end", 0.0))
            )
            picked = track[window & usable]
            if picked.size:
                per_speaker.setdefault(label, []).extend(picked.tolist())
        result["f0_by_speaker_hz"] = {
            label: round(float(np.median(vals)), 1)
            for label, vals in per_speaker.items()
            if len(vals) >= 5
        }

    return result


def codeswitch(bundle: AudioBundle, results: dict[str, Any]) -> dict[str, Any]:
    """Deferred.

    Doing this honestly needs a windowed language-ID model, which is another
    large download and real runtime per file. Registered rather than omitted so
    the UI shows the gap instead of implying section 4 is complete.
    """
    return {
        "usable": False,
        "status": "not_implemented",
        "reason": "needs a windowed language-ID model; deferred to a later version",
    }
