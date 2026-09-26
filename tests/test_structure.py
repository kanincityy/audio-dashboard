"""Speech-structure extractors, and one rule they must not break.

The overlap tests are here because of what the synthetic corpus showed: on
mono audio the estimator returned 0.000 s on every file that had real overlap
in it. A zero from a method that cannot produce anything else is not a
measurement, and must not reach the routing layer as a pass.

``scripts/validate_corpus.py`` is where that claim is scored against ground
truth. This is where it is nailed down so it cannot regress.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_dashboard.audio_io import AudioBundle  # noqa: E402
from audio_dashboard.features import structure  # noqa: E402


def bundle(duration: float = 10.0, *, stereo: bool = False) -> AudioBundle:
    sr = 16_000
    samples = np.zeros(int(sr * duration), dtype=np.float32)
    return AudioBundle(
        path=Path("nowhere.wav"),
        samples=samples,
        sr=sr,
        channels=np.stack([samples, samples]) if stereo else None,
        samples_16k=samples,
        duration=duration,
    )


def transcript(spans: list[tuple[float, float, str]]) -> dict:
    return {
        "transcript": {
            "utterances": [
                {"start": start, "end": end, "speaker": who, "words": []}
                for start, end, who in spans
            ]
        }
    }


def test_real_overlap_on_mono_is_still_reported():
    """A diariser that does emit an overlap is believed."""
    results = transcript([(0.0, 4.0, "A"), (3.0, 6.0, "B")])
    out = structure.overlap(bundle(), results)
    assert out["usable"] is True
    assert out["overlap_seconds"] == 1.0


def test_a_mono_zero_is_reported_as_unmeasured_not_as_clear():
    """The finding from the corpus, pinned down.

    Mono diarisation never emits overlapping utterances, so a zero here says
    nothing about whether people talked over each other.
    """
    results = transcript([(0.0, 4.0, "A"), (4.5, 8.0, "B")])
    out = structure.overlap(bundle(), results)
    assert out["usable"] is False
    assert out["overlap_seconds"] == 0.0
    assert "mono" in out["reason"]


def test_a_zero_from_multichannel_audio_is_a_real_answer():
    """Per-channel audio can genuinely have no overlap, and that is a pass."""
    results = transcript([(0.0, 4.0, "A"), (4.5, 8.0, "B")])
    out = structure.overlap(bundle(stereo=True), results)
    assert out["usable"] is True
    assert out["overlap_ratio"] == 0.0


def test_the_routing_rule_cannot_read_a_mono_zero_as_a_pass():
    from audio_dashboard import routing

    results = {"overlap": structure.overlap(bundle(), transcript([(0.0, 4.0, "A")]))}
    verdict = next(v for v in routing.evaluate(results) if v.rule == "Overlapping speech")
    assert verdict.status == routing.UNKNOWN


def test_unattributed_speech_counts_only_what_no_utterance_covers():
    results = transcript([(0.0, 4.0, "A")])
    results["vad"] = {"speech_spans": [[0.0, 4.0], [6.0, 8.0]]}
    out = structure.overlap(bundle(), results)
    assert out["unattributed_speech_s"] == 2.0
