"""The routing matrix from notes.md, as code.

Each rule reads the merged analysis results and either fires or does not. A
rule that cannot decide — because the analysis it needs was not run, or ran and
could not produce a usable number — says so rather than quietly not firing.
That distinction is the point: "no reverberation problem" and "we did not
measure reverberation" must not look the same in the UI.

Thresholds live here as named constants because tuning them is the likely
outcome of using this tool, and they should be tunable in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# notes.md:41 and notes.md:12. The matrix says 12 dB, the prose says 15 dB;
# 12 is the harder threshold and the one the matrix routes on.
SNR_DB = 12.0
NARROWBAND_HZ = 8_000
SHORT_DURATION_S = 3.0
SILENCE_RATIO = 0.70
OVERLAP_RATIO = 0.15
QUIET_RMS_DBFS = -30.0
# notes.md:18. Above this the recording is far-field rather than close-mic.
RT60_S = 0.8
# notes.md:16. Below this share of the available band, the top of the spectrum
# is empty and the file was probably upsampled from something narrower.
BANDWIDTH_RATIO = 0.60
# notes.md:36. Roughly a soprano speaking voice — above what most models saw
# much of in training. features/speaker.py imports this so the flag it sets and
# the threshold shown here can never drift apart.
HIGH_PITCH_HZ = 255.0

FIRED = "fired"
CLEAR = "clear"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Verdict:
    rule: str
    signal: str
    status: str  # FIRED | CLEAR | UNKNOWN
    observed: Any
    threshold: str
    action: str
    detail: str = ""


def metric_value(results: dict[str, Any], analysis: str, key: str) -> Any:
    """Read a metric, treating a failed or unusable analysis as absent.

    Public because ``interpret`` reads results the same way, and the two must
    agree about what counts as absent. A rule that cannot see a number says so;
    a card that cannot see a number must not colour itself green.
    """
    block = results.get(analysis)
    if not isinstance(block, dict) or "error" in block:
        return None
    if block.get("usable") is False:
        return None
    return block.get(key)


def _rule(
    *,
    rule: str,
    signal: str,
    observed: Any,
    fired: bool | None,
    threshold: str,
    action: str,
    missing: str,
    detail: str = "",
) -> Verdict:
    if observed is None or fired is None:
        return Verdict(rule, signal, UNKNOWN, None, threshold, action, missing)
    return Verdict(
        rule, signal, FIRED if fired else CLEAR, observed, threshold, action, detail
    )


def evaluate(results: dict[str, Any]) -> list[Verdict]:
    """Every rule, in the order notes.md lists them."""
    verdicts: list[Verdict] = []

    snr_db = metric_value(results, "snr", "snr_db")
    snr_method = metric_value(results, "snr", "method")
    verdicts.append(
        _rule(
            rule="Low signal-to-noise ratio",
            signal="snr_db",
            observed=snr_db,
            fired=None if snr_db is None else snr_db < SNR_DB,
            threshold=f"< {SNR_DB:g} dB",
            action="Apply noise suppression (DeepFilterNet or RNNoise), "
            "then route to a noise-robust engine.",
            missing="Run the signal-to-noise analysis to decide this.",
            detail="Estimated from a percentile split, not from VAD — treat as rough."
            if snr_method == "percentile"
            else "",
        )
    )

    sample_rate = metric_value(results, "format", "sample_rate_hz")
    verdicts.append(
        _rule(
            rule="Telephony-band audio",
            signal="sample_rate_hz",
            observed=sample_rate,
            fired=None if sample_rate is None else sample_rate <= NARROWBAND_HZ,
            threshold=f"<= {NARROWBAND_HZ} Hz",
            action="Skip upsampling to 16 kHz. Route directly to a "
            "telephony-tuned engine such as Deepgram Nova-2 Telephony or Speechmatics.",
            missing="Run the format analysis to decide this.",
        )
    )

    channels = metric_value(results, "format", "channels")
    verdicts.append(
        _rule(
            rule="Multi-channel audio",
            signal="channels",
            observed=channels,
            fired=None if channels is None else channels > 1,
            threshold="> 1 channel",
            action="Split per channel before transcription to remove cross-talk.",
            missing="Run the format analysis to decide this.",
        )
    )

    heavy = metric_value(results, "format", "heavy_compression")
    codec = metric_value(results, "format", "codec")
    verdicts.append(
        _rule(
            rule="Heavy compression",
            signal="codec / bitrate",
            observed=codec,
            fired=heavy,
            threshold="lossy speech codec, or under 32 kbps",
            action="Expect spectral distortion and hallucinated text. Prefer a "
            "model tolerant of codec artefacts and treat low-confidence spans "
            "with suspicion.",
            missing="Run the format analysis to decide this.",
        )
    )

    duration = metric_value(results, "format", "duration_s") or metric_value(
        results, "truncation", "duration_s"
    )
    verdicts.append(
        _rule(
            rule="Very short utterance",
            signal="duration_s",
            observed=duration,
            fired=None if duration is None else duration < SHORT_DURATION_S,
            threshold=f"< {SHORT_DURATION_S:g} s",
            action="Trim leading and trailing silence, then use a "
            "non-autoregressive or strict-prompt engine to avoid hallucination.",
            missing="Run the format analysis to decide this.",
        )
    )

    silence_ratio = metric_value(results, "density", "silence_ratio")
    verdicts.append(
        _rule(
            rule="Sparse speech",
            signal="silence_ratio",
            observed=silence_ratio,
            fired=None if silence_ratio is None else silence_ratio > SILENCE_RATIO,
            threshold=f"> {SILENCE_RATIO:.0%} silence",
            action="Trim silence before transcription. Long silent stretches "
            "make autoregressive models loop or invent repeated text.",
            missing="Run voice activity detection and speech density to decide this.",
        )
    )

    truncated = metric_value(results, "truncation", "hallucination_risk")
    verdicts.append(
        _rule(
            rule="Truncated or clipped utterance",
            signal="truncation flags",
            observed=truncated,
            fired=truncated,
            threshold="speech touching either file boundary, or under 2 s",
            action="Highest hallucination rate of any class here. Check the "
            "source segmentation before trusting the transcript.",
            missing="Run the truncation analysis to decide this.",
        )
    )

    overlap_ratio = metric_value(results, "overlap", "overlap_ratio")
    verdicts.append(
        _rule(
            rule="Overlapping speech",
            signal="overlap_ratio",
            observed=overlap_ratio,
            fired=None if overlap_ratio is None else overlap_ratio > OVERLAP_RATIO,
            threshold=f"> {OVERLAP_RATIO:.0%} of duration",
            action="Run source separation or multi-channel diarisation before "
            "transcription.",
            missing="Transcribe and run the overlap analysis to decide this.",
            detail="Mono diarisation underreports overlap, so a clear verdict "
            "here is weak evidence. Check unattributed speech as well.",
        )
    )

    rms = metric_value(results, "levels", "rms_dbfs")
    verdicts.append(
        _rule(
            rule="Audio too quiet",
            signal="rms_dbfs",
            observed=rms,
            fired=None if rms is None else rms < QUIET_RMS_DBFS,
            threshold=f"< {QUIET_RMS_DBFS:g} dBFS",
            action="Apply peak or loudness normalisation before transcription.",
            missing="Run the levels analysis to decide this.",
        )
    )

    clip_runs = metric_value(results, "clipping", "clipped_run_count")
    verdicts.append(
        _rule(
            rule="Clipping and distortion",
            signal="clipped_run_count",
            observed=clip_runs,
            fired=None if clip_runs is None else clip_runs > 0,
            threshold="any run of 3+ samples at full scale",
            action="Clipping flattens the waveform and ruins phonetic boundary "
            "detection. Re-capture at a lower input gain where possible; there "
            "is no clean recovery after the fact.",
            missing="Run the clipping analysis to decide this.",
        )
    )

    ratio = metric_value(results, "bandwidth", "cutoff_to_nyquist")
    cutoff_khz = metric_value(results, "bandwidth", "effective_cutoff_khz")
    verdicts.append(
        _rule(
            rule="High-frequency loss or artificial upsampling",
            signal="cutoff_to_nyquist",
            observed=ratio,
            fired=None if ratio is None else ratio < BANDWIDTH_RATIO,
            threshold=f"< {BANDWIDTH_RATIO:.0%} of available band",
            action="Do not treat the declared sample rate as the real "
            "bandwidth. Route as narrowband audio.",
            missing="Run the bandwidth analysis to decide this.",
            detail=f"Energy stops at about {cutoff_khz} kHz."
            if cutoff_khz is not None
            else "",
        )
    )

    rt = metric_value(results, "rt60", "rt60_s")
    windows = metric_value(results, "rt60", "windows")
    verdicts.append(
        _rule(
            rule="High reverberation",
            signal="rt60_s",
            observed=rt,
            fired=None if rt is None else rt > RT60_S,
            threshold=f"> {RT60_S:g} s",
            action="Far-field recording. Route to a dereverberation pipeline or "
            "a Conformer-based engine.",
            missing="Run voice activity detection and the RT60 analysis to decide this.",
            detail=f"Blind estimate from {windows} decay window(s); "
            "few windows means little confidence."
            if windows
            else "",
        )
    )

    speaker_count = metric_value(results, "speakers", "speaker_count")
    verdicts.append(
        _rule(
            rule="Multiple speakers",
            signal="speaker_count",
            observed=speaker_count,
            fired=None if speaker_count is None else speaker_count > 1,
            threshold="> 1 speaker",
            action="Use a diarisation-enabled endpoint. Single-speaker "
            "dictation can take a cheaper, faster model.",
            missing="Transcribe and run the speaker-count analysis to decide this.",
        )
    )

    high_pitch = metric_value(results, "f0", "high_pitch")
    median_f0 = metric_value(results, "f0", "f0_median_hz")
    verdicts.append(
        _rule(
            rule="High fundamental pitch",
            signal="f0_median_hz",
            observed=median_f0,
            fired=high_pitch,
            threshold=f"median above {HIGH_PITCH_HZ:g} Hz",
            action="Children's and high-pitched voices spike word error rates "
            "in models trained mostly on adult read speech. Prefer a model with "
            "broader pitch coverage.",
            missing="Run voice activity detection and the pitch analysis to decide this.",
        )
    )

    return verdicts
