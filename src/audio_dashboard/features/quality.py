"""Section 2 of notes.md: signal quality and acoustic metrics.

These are the metrics that explain *why* an engine produced the transcript it
did. None of them need a reference signal, which is the whole point — they have
to work on a file that arrived with no ground truth attached.

Several are estimates rather than measurements, and each one says so in its
own result under a key the UI renders. A confidently wrong RT60 is worse than
an RT60 labelled as a guess.
"""

from __future__ import annotations

import librosa
import numpy as np

from ..audio_io import AudioBundle

FRAME_MS = 20.0

# A sample this close to full scale is at the ceiling. Anything short of 1.0
# leaves room for a decoder that rounds slightly under.
CLIP_CEILING = 0.999
# Isolated peaks at full scale are normal. A flat run is what destroys the
# waveform shape and with it the phonetic boundaries. notes.md:14.
CLIP_RUN_SAMPLES = 3

# Band-limiting is found by looking for a cliff, not by thresholding against
# the spectral peak. Speech energy falls away steadily with frequency, so any
# fixed distance below the peak just measures that tilt and calls every speech
# recording narrowband. A resampler or a codec cutoff instead leaves a near
# vertical drop, and that is what these constants describe.
CLIFF_DROP_DB = 20.0
# Width either side of a candidate cutoff, in Hz, over which the levels above
# and below are compared. Wide enough to ignore individual bins, narrow enough
# that a gradual roll-off does not accumulate into a false cliff.
CLIFF_WINDOW_HZ = 700.0
# Ignore the bottom of the spectrum: there is no such thing as a 200 Hz cutoff
# in speech audio, and the peak sits down there.
CLIFF_SEARCH_START_HZ = 1500.0

# Schroeder fit region. Starting at -5 dB skips the direct sound, and stopping
# at -25 dB stays above the noise floor on real recordings; the result is a T20
# multiplied up to a full 60 dB of decay.
DECAY_START_DB = -5.0
DECAY_END_DB = -25.0
MIN_DECAY_WINDOW_S = 0.15
MAX_DECAY_WINDOW_S = 1.5

_EPS = 1e-12


def _db(value: np.ndarray | float) -> np.ndarray | float:
    """Amplitude ratio to dB, with a floor so silence does not become -inf."""
    return 20.0 * np.log10(np.maximum(np.abs(value), _EPS))


def _frame_rms(samples: np.ndarray, sr: int, frame_ms: float = FRAME_MS):
    """RMS per non-overlapping frame, plus each frame's start time."""
    n = max(1, int(sr * frame_ms / 1000.0))
    usable = (len(samples) // n) * n
    if usable == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    frames = samples[:usable].reshape(-1, n).astype(np.float64)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    times = np.arange(len(rms)) * n / sr
    return rms, times


def _speech_mask(spans, times: np.ndarray, frame_s: float) -> np.ndarray:
    """Which frames fall inside a speech span."""
    mask = np.zeros(len(times), dtype=bool)
    for start, end in spans:
        mask |= (times + frame_s > start) & (times < end)
    return mask


def levels(bundle: AudioBundle, results) -> dict:
    samples = bundle.samples
    rms, _ = _frame_rms(samples, bundle.sr)

    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    overall_rms = (
        float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        if len(samples)
        else 0.0
    )

    # Dynamic range as the spread of frame loudness rather than peak-to-noise:
    # the 95th percentile is a loud moment of speech, the 10th is the quiet
    # background, and the gap between them is what a normaliser would close.
    loud = float(np.percentile(rms, 95)) if len(rms) else 0.0
    quiet = float(np.percentile(rms, 10)) if len(rms) else 0.0

    return {
        "peak_dbfs": float(_db(peak)),
        "rms_dbfs": float(_db(overall_rms)),
        "crest_factor_db": float(_db(peak) - _db(overall_rms)),
        "dynamic_range_db": float(_db(loud) - _db(quiet)),
        "loud_frame_dbfs": float(_db(loud)),
        "quiet_frame_dbfs": float(_db(quiet)),
    }


def clipping(bundle: AudioBundle, results) -> dict:
    samples = bundle.samples
    if not len(samples):
        return {
            "clipped_sample_pct": 0.0,
            "clipped_run_count": 0,
            "clipped_run_pct": 0.0,
        }

    at_ceiling = np.abs(samples) >= CLIP_CEILING
    total = len(samples)

    # Run-length encode the ceiling mask, then keep only runs long enough to
    # be a flattened waveform rather than a single loud sample.
    padded = np.concatenate(([False], at_ceiling, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    lengths = ends - starts
    long_runs = lengths >= CLIP_RUN_SAMPLES

    run_starts = starts[long_runs]
    run_lengths = lengths[long_runs]

    # Positions in seconds, for marking on the waveform plot. Capped so a
    # badly clipped file does not write a megabyte of cache.
    positions = (run_starts[:2000] / bundle.sr).tolist()

    return {
        "clipped_sample_pct": round(100.0 * float(at_ceiling.sum()) / total, 4),
        "clipped_run_count": int(long_runs.sum()),
        "clipped_run_pct": round(100.0 * float(run_lengths.sum()) / total, 4),
        "longest_run_samples": int(run_lengths.max()) if len(run_lengths) else 0,
        "run_positions_s": positions,
        "run_positions_truncated": int(long_runs.sum()) > 2000,
    }


def snr(bundle: AudioBundle, results) -> dict:
    """Speech level against noise floor, in dB.

    With VAD available this is a real measurement: the RMS of frames the model
    called speech against the RMS of frames it did not. Without it, this falls
    back to a percentile split of frame energy, which assumes the quietest
    tenth of the file is noise and the loudest tenth is speech. That assumption
    fails on continuous speech with no pauses, and fails differently on a file
    that is mostly silence — hence ``method`` in the result, which the UI shows.
    """
    rms, times = _frame_rms(bundle.samples, bundle.sr)
    if not len(rms):
        return {"snr_db": None, "method": "none", "usable": False}

    frame_s = FRAME_MS / 1000.0
    spans = (results.get("vad") or {}).get("speech_spans")

    if spans:
        mask = _speech_mask(spans, times, frame_s)
        speech_frames = rms[mask]
        noise_frames = rms[~mask]
        method = "vad"
    else:
        speech_frames = noise_frames = np.zeros(0)
        method = "percentile"

    # Even with VAD, a file that is wall-to-wall speech gives no noise frames
    # to measure against. Say so rather than dividing by the epsilon floor and
    # reporting a spectacular SNR that means nothing.
    if method == "vad" and (len(speech_frames) < 3 or len(noise_frames) < 3):
        return {
            "snr_db": None,
            "method": "vad",
            "usable": False,
            "reason": "not enough speech or non-speech frames to compare",
            "speech_frames": int(len(speech_frames)),
            "noise_frames": int(len(noise_frames)),
        }

    if method == "percentile":
        speech_level = float(np.percentile(rms, 90))
        noise_level = float(np.percentile(rms, 10))
    else:
        speech_level = float(np.sqrt(np.mean(speech_frames.astype(np.float64) ** 2)))
        noise_level = float(np.sqrt(np.mean(noise_frames.astype(np.float64) ** 2)))

    return {
        "snr_db": round(float(_db(speech_level) - _db(noise_level)), 2),
        "method": method,
        "usable": True,
        "speech_level_dbfs": round(float(_db(speech_level)), 2),
        "noise_floor_dbfs": round(float(_db(noise_level)), 2),
        "speech_frames": int(len(speech_frames)) if method == "vad" else None,
        "noise_frames": int(len(noise_frames)) if method == "vad" else None,
    }


def bandwidth(bundle: AudioBundle, results) -> dict:
    """Where the spectrum actually stops, against where it could stop.

    An 8 kHz recording padded up to 16 kHz still reports a 16 kHz sample rate,
    so the format metadata alone cannot catch it. The average spectrum can: the
    top half is empty, and a model expecting fricative energy up there is being
    misled. notes.md:16.

    The thing being looked for is a cliff — a sharp drop with near-nothing
    above it — because that is what a resampler or a codec cutoff leaves
    behind. Speech spectra slope downwards on their own, so a cutoff defined as
    a fixed distance below the spectral peak would flag every speech recording
    ever made as band-limited. When there is no cliff, the file is using its
    whole band and the cutoff is the Nyquist frequency.
    """
    samples = bundle.samples
    if len(samples) < 2048:
        return {"usable": False, "reason": "file too short to estimate a spectrum"}

    spec = np.abs(np.asarray(librosa.stft(samples, n_fft=2048, hop_length=512)))
    power = np.mean(spec.astype(np.float64) ** 2, axis=1)
    power_db = 10.0 * np.log10(np.maximum(power, _EPS))
    peak_db = float(power_db.max())

    nyquist = bundle.sr / 2.0
    freqs = np.linspace(0.0, nyquist, len(power_db))
    bin_hz = nyquist / max(1, len(power_db) - 1)

    window = max(2, int(CLIFF_WINDOW_HZ / bin_hz))
    start = max(window, int(CLIFF_SEARCH_START_HZ / bin_hz))

    best_drop, best_bin = 0.0, len(power_db) - 1
    for i in range(start, len(power_db) - window):
        below = float(np.median(power_db[i - window : i]))
        above = float(np.median(power_db[i : i + window]))
        drop = below - above
        if drop > best_drop:
            best_drop, best_bin = drop, i

    if best_drop >= CLIFF_DROP_DB:
        cutoff_hz = float(freqs[best_bin])
    else:
        cutoff_hz = nyquist

    ratio = cutoff_hz / nyquist if nyquist else 0.0

    # Store a thinned spectrum so the UI can draw the cliff. The scalar alone
    # is unconvincing; seeing where the energy stops is not.
    step = max(1, len(power_db) // 512)

    return {
        "usable": True,
        "effective_cutoff_hz": round(cutoff_hz, 1),
        "effective_cutoff_khz": round(cutoff_hz / 1000.0, 2),
        "nyquist_hz": round(nyquist, 1),
        "cutoff_to_nyquist": round(float(ratio), 3),
        "cliff_drop_db": round(best_drop, 1),
        "cliff_found": bool(best_drop >= CLIFF_DROP_DB),
        "likely_upsampled": bool(ratio < 0.6),
        "spectrum_freqs_hz": freqs[::step].round(1).tolist(),
        "spectrum_db": power_db[::step].round(2).tolist(),
        "spectrum_peak_db": round(peak_db, 2),
        "cliff_threshold_db": CLIFF_DROP_DB,
    }


def rt60(bundle: AudioBundle, results) -> dict:
    """Blind reverberation estimate from the decay after each speech offset.

    There is no impulse response here, so this takes what the room gives us:
    the tail after a talker stops. Each gap long enough to show a decay gets a
    Schroeder backward integration and a T20 fit, extrapolated to RT60.

    Treat the number as indicative. With few usable windows it says very
    little, which is why the window count is returned alongside it and shown.
    """
    spans = (results.get("vad") or {}).get("speech_spans") or []
    if not spans:
        return {"usable": False, "reason": "no speech spans; run VAD first"}

    samples = bundle.samples.astype(np.float64)
    sr = bundle.sr
    estimates: list[float] = []

    for i, (_, offset) in enumerate(spans):
        # The decay window runs from this offset to the next onset, so no
        # window ever contains speech.
        next_onset = spans[i + 1][0] if i + 1 < len(spans) else bundle.duration
        window_s = min(next_onset - offset, MAX_DECAY_WINDOW_S)
        if window_s < MIN_DECAY_WINDOW_S:
            continue

        start = int(offset * sr)
        stop = int((offset + window_s) * sr)
        segment = samples[start:stop]
        if len(segment) < 64:
            continue

        # Schroeder: energy remaining from each point to the end of the window.
        energy = segment**2
        remaining = np.cumsum(energy[::-1])[::-1]
        curve_db = 10.0 * np.log10(
            np.maximum(remaining / max(remaining[0], _EPS), _EPS)
        )

        below_start = np.flatnonzero(curve_db <= DECAY_START_DB)
        below_end = np.flatnonzero(curve_db <= DECAY_END_DB)
        if not len(below_start) or not len(below_end):
            continue

        i0, i1 = int(below_start[0]), int(below_end[0])
        if i1 <= i0:
            continue

        # Least squares over the fit region rather than the two crossing points
        # alone, so a single noisy sample cannot set the slope.
        idx = np.arange(i0, i1 + 1)
        slope, _ = np.polyfit(idx / sr, curve_db[i0 : i1 + 1], 1)
        if slope >= 0:
            continue
        estimates.append(float(-60.0 / slope))

    if not estimates:
        return {
            "usable": False,
            "reason": "no gap after speech was long or clean enough to fit a decay",
            "windows": 0,
        }

    values = np.array(estimates)
    return {
        "usable": True,
        "estimate": True,
        "rt60_s": round(float(np.median(values)), 3),
        "rt60_iqr_s": round(
            float(np.percentile(values, 75) - np.percentile(values, 25)), 3
        ),
        "windows": int(len(values)),
        "far_field": bool(np.median(values) > 0.8),
        "fit_region_db": [DECAY_START_DB, DECAY_END_DB],
    }
