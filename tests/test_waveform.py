"""The waveform outline against signals whose extremes are known."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_dashboard import waveform


def test_a_short_clipped_peak_survives_the_downsampling():
    """Plain decimation would skip the spike; min/max per bucket keeps it."""
    samples = np.zeros(100_000, dtype=np.float32)
    samples[51_234] = 1.0

    outline = waveform.envelope(samples, buckets=100)

    assert outline["points"] == 100
    assert max(outline["max"]) == 1.0
    assert min(outline["min"]) == 0.0


def test_negative_peaks_are_kept_in_the_min():
    samples = np.zeros(10_000, dtype=np.float32)
    samples[5_000] = -0.8
    assert min(waveform.envelope(samples, buckets=50)["min"]) == pytest.approx(-0.8)


def test_a_signal_shorter_than_the_bucket_count_gives_one_point_per_sample():
    outline = waveform.envelope(
        np.array([0.1, -0.2, 0.3], dtype=np.float32), buckets=4000
    )
    assert outline["points"] == 3
    assert outline["max"] == pytest.approx([0.1, -0.2, 0.3])


def test_empty_audio_gives_an_empty_outline():
    assert waveform.envelope(np.zeros(0, dtype=np.float32)) == {
        "points": 0,
        "min": [],
        "max": [],
    }


def test_zero_buckets_is_refused():
    with pytest.raises(ValueError):
        waveform.envelope(np.zeros(10, dtype=np.float32), buckets=0)
