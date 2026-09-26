"""Quality extractors against signals whose properties we constructed.

The value of a synthesised signal is that the right answer is known in advance,
so these are checks that the estimator recovers a number we put there, not that
it returns the number it happens to return today.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_dashboard.audio_io import AudioBundle  # noqa: E402
from audio_dashboard.features import quality  # noqa: E402

SR = 16_000


def bundle(samples: np.ndarray, sr: int = SR) -> AudioBundle:
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    return AudioBundle(
        path=Path("synthetic.wav"),
        samples=samples,
        sr=sr,
        channels=None,
        samples_16k=samples,
        duration=len(samples) / sr,
    )


def sine(freq: float, seconds: float, amplitude: float = 1.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return amplitude * np.sin(2 * np.pi * freq * t)


# ------------------------------------------------------------------- levels


def test_half_amplitude_sine_peaks_at_minus_six_dbfs():
    result = quality.levels(bundle(sine(440, 1.0, amplitude=0.5)), {})
    assert result["peak_dbfs"] == pytest.approx(-6.02, abs=0.1)


def test_sine_rms_is_three_db_below_its_peak():
    """A sine's RMS is its amplitude over root two, which is 3.01 dB down."""
    result = quality.levels(bundle(sine(440, 1.0, amplitude=1.0)), {})
    assert result["rms_dbfs"] == pytest.approx(-3.01, abs=0.1)
    assert result["crest_factor_db"] == pytest.approx(3.01, abs=0.1)


def test_silence_does_not_produce_negative_infinity():
    result = quality.levels(bundle(np.zeros(SR)), {})
    assert np.isfinite(result["peak_dbfs"])
    assert np.isfinite(result["rms_dbfs"])


# ----------------------------------------------------------------- clipping


def test_clean_sine_reports_no_clipping():
    result = quality.clipping(bundle(sine(440, 1.0, amplitude=0.5)), {})
    assert result["clipped_run_count"] == 0
    assert result["clipped_sample_pct"] == 0.0


def test_hard_clipped_sine_is_detected():
    # Amplify past full scale then clamp, which is what a hot input does.
    clipped = np.clip(sine(200, 1.0, amplitude=2.0), -1.0, 1.0)
    result = quality.clipping(bundle(clipped), {})
    assert result["clipped_run_count"] > 0
    # Each cycle spends a good share of its time flattened at both rails.
    assert result["clipped_run_pct"] > 20.0
    assert result["longest_run_samples"] >= quality.CLIP_RUN_SAMPLES


def test_isolated_full_scale_samples_are_not_counted_as_runs():
    """A single sample at the ceiling is a peak, not a flattened waveform."""
    samples = sine(440, 1.0, amplitude=0.5)
    samples[1000] = 1.0
    samples[5000] = -1.0
    result = quality.clipping(bundle(samples), {})
    assert result["clipped_sample_pct"] > 0.0
    assert result["clipped_run_count"] == 0


def test_clipping_run_positions_are_in_seconds():
    clipped = np.clip(sine(200, 2.0, amplitude=2.0), -1.0, 1.0)
    result = quality.clipping(bundle(clipped), {})
    assert all(0.0 <= p <= 2.0 for p in result["run_positions_s"])


# ---------------------------------------------------------------- bandwidth


def test_full_band_noise_uses_most_of_the_available_band():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.1, SR * 2)
    result = quality.bandwidth(bundle(noise), {})
    assert result["usable"]
    assert not result["cliff_found"]
    assert result["cutoff_to_nyquist"] == 1.0
    assert not result["likely_upsampled"]


def test_a_sloping_speech_spectrum_is_not_mistaken_for_band_limiting():
    """Speech energy falls away with frequency; that is not a cutoff.

    This is the failure the cliff test exists to prevent — thresholding a
    fixed distance below the spectral peak flags every speech recording as
    narrowband, because the peak sits a few hundred Hz up and everything
    above it is quieter by construction.
    """
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 1.0, SR * 2)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(len(noise), 1 / SR)
    # About 12 dB per octave of tilt, steeper than real speech, and still
    # gradual rather than a cliff.
    tilt = 1.0 / (1.0 + (freqs / 300.0) ** 2)
    sloped = np.fft.irfft(spectrum * tilt)
    sloped = 0.3 * sloped / np.max(np.abs(sloped))

    result = quality.bandwidth(bundle(sloped), {})
    assert not result["cliff_found"]
    assert not result["likely_upsampled"]


def test_upsampled_narrowband_audio_is_flagged():
    """8 kHz content padded to 16 kHz leaves the top half of the band empty."""
    import librosa

    rng = np.random.default_rng(1)
    narrow = rng.normal(0, 0.1, 8_000 * 2).astype(np.float32)
    # A real resampler, which is the operation actually being detected: it
    # cannot invent content above the old Nyquist, so 4 kHz upwards is empty.
    upsampled = librosa.resample(narrow, orig_sr=8_000, target_sr=SR)

    result = quality.bandwidth(bundle(upsampled, sr=SR), {})
    assert result["usable"]
    assert result["cliff_found"]
    assert result["likely_upsampled"]
    assert result["effective_cutoff_khz"] == pytest.approx(4.0, abs=0.7)


def test_bandwidth_declines_on_a_file_too_short_to_analyse():
    result = quality.bandwidth(bundle(np.zeros(100)), {})
    assert result["usable"] is False


# ---------------------------------------------------------------------- snr


def speech_like(seconds: float, sr: int = SR) -> np.ndarray:
    """A crude voiced sound: a fundamental plus a few harmonics."""
    return sum(sine(140 * k, seconds, amplitude=0.4 / k, sr=sr) for k in (1, 2, 3, 4))


def build_snr_case(snr_db: float, seed: int = 0):
    """Speech in the middle of a file, over noise at a constructed level.

    Returns the samples and the true speech span, so the VAD-based path can be
    tested without running Silero.
    """
    rng = np.random.default_rng(seed)
    total, speech_s = 6.0, 2.0
    noise = rng.normal(0, 1.0, int(total * SR))
    noise /= np.sqrt(np.mean(noise**2))

    speech = speech_like(speech_s)
    speech /= np.sqrt(np.mean(speech**2))

    # Scale speech so speech-plus-noise sits the requested distance above noise.
    target = 10 ** (snr_db / 20.0)
    start = int(2.0 * SR)
    samples = noise * 0.05
    samples[start : start + len(speech)] += speech * 0.05 * target
    return samples, [[2.0, 2.0 + speech_s]]


@pytest.mark.parametrize("snr_db", [5.0, 15.0, 25.0])
def test_vad_snr_recovers_the_constructed_ratio(snr_db):
    samples, spans = build_snr_case(snr_db)
    result = quality.snr(bundle(samples), {"vad": {"speech_spans": spans}})
    assert result["method"] == "vad"
    # The speech frames contain speech *plus* noise, so the measured ratio sits
    # slightly above the constructed one; a couple of dB is the honest tolerance.
    assert result["snr_db"] == pytest.approx(snr_db, abs=2.5)


def test_snr_without_vad_falls_back_and_says_so():
    samples, _ = build_snr_case(15.0)
    result = quality.snr(bundle(samples), {})
    assert result["method"] == "percentile"
    assert result["snr_db"] is not None


def test_snr_declines_when_there_are_no_noise_frames():
    """Wall-to-wall speech gives nothing to measure the noise floor against."""
    samples = speech_like(3.0)
    spans = [[0.0, 3.0]]
    result = quality.snr(bundle(samples), {"vad": {"speech_spans": spans}})
    assert result["usable"] is False
    assert result["snr_db"] is None


# -------------------------------------------------------------------- rt60


def test_rt60_needs_vad():
    result = quality.rt60(bundle(sine(440, 1.0)), {})
    assert result["usable"] is False


def test_rt60_recovers_a_synthetic_decay():
    """Speech followed by an exponential tail with a known decay constant."""
    decay_s = 0.5
    speech = speech_like(1.0)
    tail_len = int(1.0 * SR)
    rng = np.random.default_rng(2)
    tail = rng.normal(0, 0.3, tail_len)
    # Amplitude decaying so that energy falls 60 dB in decay_s.
    t = np.arange(tail_len) / SR
    tail *= 10 ** (-3.0 * t / decay_s)

    samples = np.concatenate([speech, tail])
    spans = [[0.0, 1.0]]
    result = quality.rt60(bundle(samples), {"vad": {"speech_spans": spans}})

    assert result["usable"]
    assert result["windows"] >= 1
    assert result["rt60_s"] == pytest.approx(decay_s, rel=0.35)


def test_rt60_reports_no_usable_windows_when_speech_never_stops():
    samples = speech_like(2.0)
    spans = [[0.0, 2.0]]
    result = quality.rt60(bundle(samples), {"vad": {"speech_spans": spans}})
    assert result["usable"] is False
    assert result["windows"] == 0
