"""Reference-free summaries of a transcript.

These are deliberately cheap signals that need no ground truth. If the
preprocessing experiment in notes.txt goes ahead, they become its proxy metrics.
"""

from __future__ import annotations

import statistics
from typing import Any

from . import salience

# Turns shorter than this are counted as "flapping" — a run of very short
# alternating turns is the usual signature of diarisation breaking down.
SHORT_TURN_SECONDS = 1.0

# A "% of words below threshold" figure computed over fewer words than this is
# noise, not signal — one bad word alone in a bin would read as 100%.
MIN_BIN_WORDS = 3


def percentile_threshold(
    words: list[dict[str, Any]],
    pct: float,
    *,
    suppress_filler: bool = True,
) -> float:
    """The cutoff that flags ``pct`` percent of *this* file's words.

    Used by the percentile colouring mode, which is robust to per-file shifts in
    the confidence distribution that fixed bands miss.

    ``pct`` means percent of the *whole transcript*, which is the only reading
    that keeps the number comparable to the ~4% the literature found readers
    prefer — that figure is a share of words highlighted on screen, not of some
    eligible subset.

    This searches the real predicate rather than computing a percentile of the
    confidence distribution, because suppressed filler breaks the correspondence
    between "Nth percentile of confidences" and "N percent of words flagged".
    Searching also cannot drift from :func:`salience.is_flagged` the way a
    parallel formula could.

    :func:`flagged_rate` is monotone non-decreasing in the cutoff, so the
    smallest cutoff hitting the target is a bisection over the values where the
    rate can change at all — each word's confidence.
    """
    if not words:
        return 0.0

    candidates = sorted({w["confidence"] for w in words} | {0.0})

    def rate_at(base: float) -> float:
        return flagged_rate(words, base, suppress_filler=suppress_filler)

    # Unreachable targets clamp to the strictest cutoff on offer rather than
    # running off the end of the scale — a file can simply not have enough
    # flaggable words to fill a 25% request.
    if rate_at(candidates[-1]) < pct:
        return candidates[-1]

    lo, hi = 0, len(candidates) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if rate_at(candidates[mid]) >= pct:
            hi = mid
        else:
            lo = mid + 1
    return candidates[lo]


def resolve_threshold(
    words: list[dict[str, Any]],
    mode: str,
    *,
    value: float = 0.6,
    pct: float = 4.0,
    floor: float = 0.9,
    suppress_filler: bool = True,
) -> float:
    """The single threshold, however the user chose to specify it.

    One resolver so the marks, the timeline, the rug and the readout are all
    driven by the same number — previously the app and the renderer each derived
    it separately and could drift.

    There is no calibrated "correct" value to look up: published optimal
    thresholds range 0.41-0.94 across transcripts (arXiv:2503.15124), so a fixed
    value gives an unpredictable highlight rate file to file. Hence the
    rate-targeting modes.
    """
    if mode == "band":
        return value
    pct_threshold = percentile_threshold(words, pct, suppress_filler=suppress_filler)
    if mode == "percentile_floor":
        # Stops a genuinely clean call having its least-confident words flagged
        # as if they were problems.
        return min(pct_threshold, floor)
    return pct_threshold


def flagged_rate(
    words: list[dict[str, Any]],
    threshold: float,
    *,
    suppress_filler: bool = True,
) -> float:
    """Percentage of *all* words the threshold currently flags.

    The denominator stays the full word count even when filler is suppressed, so
    the number keeps meaning "how much of this transcript is marked" rather than
    silently switching to a share of a subset.
    """
    if not words:
        return 0.0
    flagged = sum(
        1
        for w in words
        if salience.is_flagged(w, threshold, suppress_filler=suppress_filler)
    )
    return 100.0 * flagged / len(words)


def boundary_confidence(
    transcript: dict[str, Any], window: int = 3, threshold: float = 0.7
) -> dict[str, Any]:
    """Confidence of words either side of a speaker change, vs everywhere else.

    Word confidence measures *transcription* uncertainty; AssemblyAI exposes no
    per-speaker-label confidence. This is the closest available proxy for
    diarisation uncertainty: if the words framing a speaker change are markedly
    less confident than the rest, the boundary itself was likely hard to place.

    Measured effect on the reference file is real but modest (0.910 vs 0.946),
    so treat it as directional evidence rather than proof.
    """
    utterances = transcript.get("utterances", [])
    all_words = transcript.get("words", [])
    if len(utterances) < 2 or not all_words:
        return {"count": 0, "boundaries": []}

    # Global word indices, relying on `words` being the utterances flattened in
    # order (see asr._parse).
    offsets: list[int] = []
    running = 0
    for utt in utterances:
        offsets.append(running)
        running += len(utt["words"])

    boundary_idx: set[int] = set()
    boundaries: list[dict[str, Any]] = []

    for i in range(len(utterances) - 1):
        before, after = utterances[i], utterances[i + 1]
        if before["speaker"] == after["speaker"]:
            continue

        left = [offsets[i] + j for j in range(len(before["words"]))][-window:]
        right = [offsets[i + 1] + j for j in range(min(window, len(after["words"])))]
        idx = left + right
        if not idx:
            continue
        boundary_idx.update(idx)

        confs = [all_words[k]["confidence"] for k in idx]
        boundaries.append(
            {
                "time": before["end"],
                "from": before["speaker"],
                "to": after["speaker"],
                "mean": statistics.fmean(confs),
                "n_below": sum(1 for c in confs if c < threshold),
                "words": " ".join(all_words[k]["text"] for k in idx),
            }
        )

    if not boundaries:
        return {"count": 0, "boundaries": []}

    at = [all_words[k]["confidence"] for k in sorted(boundary_idx)]
    away = [
        w["confidence"] for k, w in enumerate(all_words) if k not in boundary_idx
    ]

    def pct_below(values: list[float]) -> float:
        return 100.0 * sum(1 for c in values if c < threshold) / len(values) if values else 0.0

    boundary_mean = statistics.fmean(at)
    other_mean = statistics.fmean(away) if away else boundary_mean

    return {
        "count": len(boundaries),
        "window": window,
        "boundary_mean": boundary_mean,
        "other_mean": other_mean,
        "deficit": boundary_mean - other_mean,
        "boundary_pct_below": pct_below(at),
        "other_pct_below": pct_below(away),
        # Worst first — these are the ones worth listening to.
        "boundaries": sorted(boundaries, key=lambda b: b["mean"]),
    }


def confidence_timeline(
    transcript: dict[str, Any],
    bin_seconds: float,
    threshold: float,
    min_words: int = MIN_BIN_WORDS,
    *,
    suppress_filler: bool = True,
) -> dict[str, Any]:
    """Percentage of words below ``threshold`` per time bin, overall and per speaker.

    Percentage-below rather than mean confidence, deliberately: the median word
    sits at ~0.998, so a bin mean only moves when dragged by outliers and goes
    flat as bins coarsen. The share below threshold stays legible at any bin size.

    Bins in which a speaker said nothing are ``None``, never ``0.0`` — zero means
    "spoke, all words confident" and absent means "did not speak". Callers must
    break the line across ``None`` rather than dropping it to the floor, or every
    silence reads as perfect transcription.

    Bins with fewer than ``min_words`` words are ``None`` for the same reason: a
    percentage over one or two words is not a measurement. Without this floor a
    single low-confidence word alone in a bin reads as 100%, which blows out the
    auto-scaled y axis and buries the real single-digit signal. At 2s bins that
    would affect roughly a fifth of populated bins.
    """
    words = transcript.get("words", [])
    duration = transcript.get("audio_duration", 0.0)
    if not words or duration <= 0 or bin_seconds <= 0:
        return {"bin_seconds": bin_seconds, "duration": duration, "bins": [], "series": {}}

    n_bins = max(1, int(duration // bin_seconds) + 1)
    speakers = sorted({w["speaker"] for w in words})

    # (below, total) counts per bin, for the overall series and one per speaker.
    tallies: dict[str, list[list[int]]] = {
        key: [[0, 0] for _ in range(n_bins)] for key in ["__all__", *speakers]
    }

    for w in words:
        idx = min(int(w["start"] // bin_seconds), n_bins - 1)
        low = (
            1
            if salience.is_flagged(w, threshold, suppress_filler=suppress_filler)
            else 0
        )
        for key in ("__all__", w["speaker"]):
            tallies[key][idx][0] += low
            tallies[key][idx][1] += 1

    def to_series(counts: list[list[int]]) -> list[float | None]:
        return [
            (100.0 * below / total) if total >= min_words else None
            for below, total in counts
        ]

    bins = [
        {
            "start": i * bin_seconds,
            "end": min((i + 1) * bin_seconds, duration),
            "count": tallies["__all__"][i][1],
        }
        for i in range(n_bins)
    ]

    return {
        "bin_seconds": bin_seconds,
        "duration": duration,
        "threshold": threshold,
        "bins": bins,
        "overall": to_series(tallies["__all__"]),
        "series": {s: to_series(tallies[s]) for s in speakers},
        # Individual low-confidence words, for the rug strip — independent of
        # binning, so bad patches stay visible however coarse the bins are.
        "low_words": [
            w["start"]
            for w in words
            if salience.is_flagged(w, threshold, suppress_filler=suppress_filler)
        ],
    }


def summarise(
    transcript: dict[str, Any],
    threshold: float,
    *,
    suppress_filler: bool = True,
) -> dict[str, Any]:
    words = transcript.get("words", [])
    utterances = transcript.get("utterances", [])
    confidences = [w["confidence"] for w in words]

    per_speaker: dict[str, list[float]] = {}
    for w in words:
        per_speaker.setdefault(w["speaker"], []).append(w["confidence"])

    speaker_means = {
        speaker: statistics.fmean(vals) for speaker, vals in sorted(per_speaker.items())
    }
    # A wide gap between speakers means one of them is being captured badly —
    # far more diagnostic than the overall mean, which hides it.
    speaker_gap = (
        max(speaker_means.values()) - min(speaker_means.values())
        if len(speaker_means) > 1
        else 0.0
    )

    short_turns = sum(
        1 for u in utterances if (u["end"] - u["start"]) < SHORT_TURN_SECONDS
    )

    return {
        "word_count": len(words),
        "mean_confidence": statistics.fmean(confidences) if confidences else 0.0,
        "median_confidence": statistics.median(confidences) if confidences else 0.0,
        "min_confidence": min(confidences) if confidences else 0.0,
        "pct_below_threshold": flagged_rate(
            words, threshold, suppress_filler=suppress_filler
        ),
        "speaker_count": transcript.get("speaker_count", len(speaker_means)),
        "speaker_means": speaker_means,
        "speaker_gap": speaker_gap,
        "turn_count": len(utterances),
        "short_turns": short_turns,
        "pct_short_turns": (100.0 * short_turns / len(utterances)) if utterances else 0.0,
    }


def lowest_confidence_words(
    transcript: dict[str, Any], limit: int = 50, *, suppress_filler: bool = True
) -> list[dict[str, Any]]:
    """The worst words, filler excluded — otherwise the table is mostly `and`."""
    words = salience.eligible(
        transcript.get("words", []), suppress_filler=suppress_filler
    )
    return sorted(words, key=lambda w: w["confidence"])[:limit]


def histogram(
    transcript: dict[str, Any], bins: int = 20
) -> list[tuple[float, float, int]]:
    """Bucket confidences into ``(low, high, count)`` triples over [0, 1].

    Fixed 0-1 range rather than data-driven bounds, so histograms stay
    comparable across files and conditions.
    """
    words = transcript.get("words", [])
    counts = [0] * bins
    for w in words:
        idx = min(int(w["confidence"] * bins), bins - 1)
        counts[idx] += 1
    return [(i / bins, (i + 1) / bins, counts[i]) for i in range(bins)]
