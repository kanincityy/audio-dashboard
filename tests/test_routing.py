"""Every routing rule, at and either side of its threshold.

These are pure functions over a dict, so there is no excuse for not covering
them exhaustively. The rule that matters most is the three-way one: a rule
whose input is missing must come back UNKNOWN, never CLEAR. Silently reading
"we did not measure reverberation" as "reverberation is fine" is exactly the
failure this dashboard exists to avoid.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_dashboard import routing  # noqa: E402


def verdict(results: dict, rule: str) -> routing.Verdict:
    for v in routing.evaluate(results):
        if v.rule == rule:
            return v
    raise AssertionError(f"no rule named {rule!r}")


def test_every_rule_is_unknown_when_nothing_has_been_run():
    for v in routing.evaluate({}):
        assert v.status == routing.UNKNOWN, v.rule
        assert v.detail, f"{v.rule} must say what it needs"


def test_low_snr_fires_below_threshold():
    below = {"snr": {"snr_db": 11.9, "method": "vad"}}
    assert verdict(below, "Low signal-to-noise ratio").status == routing.FIRED


def test_snr_at_threshold_is_clear():
    at = {"snr": {"snr_db": routing.SNR_DB, "method": "vad"}}
    assert verdict(at, "Low signal-to-noise ratio").status == routing.CLEAR


def test_percentile_snr_is_labelled_as_rough():
    rough = {"snr": {"snr_db": 5.0, "method": "percentile"}}
    v = verdict(rough, "Low signal-to-noise ratio")
    assert v.status == routing.FIRED
    assert "percentile" in v.detail


def test_unusable_snr_is_unknown_not_clear():
    unusable = {"snr": {"snr_db": None, "method": "vad", "usable": False}}
    assert verdict(unusable, "Low signal-to-noise ratio").status == routing.UNKNOWN


def test_failed_analysis_is_unknown_not_clear():
    failed = {"snr": {"error": "ValueError: boom"}}
    assert verdict(failed, "Low signal-to-noise ratio").status == routing.UNKNOWN


def test_telephony_fires_at_8k_and_is_clear_at_16k():
    assert (
        verdict({"format": {"sample_rate_hz": 8000}}, "Telephony-band audio").status
        == routing.FIRED
    )
    assert (
        verdict({"format": {"sample_rate_hz": 16000}}, "Telephony-band audio").status
        == routing.CLEAR
    )


def test_multichannel_fires_on_stereo():
    assert verdict({"format": {"channels": 2}}, "Multi-channel audio").status == routing.FIRED
    assert verdict({"format": {"channels": 1}}, "Multi-channel audio").status == routing.CLEAR


def test_heavy_compression_follows_the_format_flag():
    fired = {"format": {"heavy_compression": True, "codec": "amr_nb"}}
    assert verdict(fired, "Heavy compression").status == routing.FIRED
    clear = {"format": {"heavy_compression": False, "codec": "flac"}}
    assert verdict(clear, "Heavy compression").status == routing.CLEAR


def test_short_utterance_threshold():
    assert (
        verdict({"format": {"duration_s": 2.9}}, "Very short utterance").status
        == routing.FIRED
    )
    assert (
        verdict({"format": {"duration_s": 3.0}}, "Very short utterance").status
        == routing.CLEAR
    )


def test_short_utterance_falls_back_to_truncation_duration():
    """Duration is available from truncation when format was not run."""
    results = {"truncation": {"duration_s": 1.0, "hallucination_risk": True}}
    assert verdict(results, "Very short utterance").status == routing.FIRED


def test_sparse_speech_threshold():
    assert (
        verdict({"density": {"silence_ratio": 0.71}}, "Sparse speech").status
        == routing.FIRED
    )
    assert (
        verdict({"density": {"silence_ratio": 0.70}}, "Sparse speech").status
        == routing.CLEAR
    )


def test_truncation_rule_follows_its_flag():
    fired = {"truncation": {"hallucination_risk": True, "duration_s": 10.0}}
    assert verdict(fired, "Truncated or clipped utterance").status == routing.FIRED
    clear = {"truncation": {"hallucination_risk": False, "duration_s": 10.0}}
    assert verdict(clear, "Truncated or clipped utterance").status == routing.CLEAR


def test_overlap_threshold_and_its_caveat():
    fired = verdict({"overlap": {"overlap_ratio": 0.16}}, "Overlapping speech")
    assert fired.status == routing.FIRED
    clear = verdict({"overlap": {"overlap_ratio": 0.15}}, "Overlapping speech")
    assert clear.status == routing.CLEAR
    # The mono caveat must survive onto a clear verdict, because that is the
    # verdict it undermines.
    assert "underreports" in clear.detail


def test_quiet_audio_threshold():
    assert verdict({"levels": {"rms_dbfs": -30.1}}, "Audio too quiet").status == routing.FIRED
    assert verdict({"levels": {"rms_dbfs": -30.0}}, "Audio too quiet").status == routing.CLEAR


def test_clipping_fires_on_any_run():
    assert (
        verdict({"clipping": {"clipped_run_count": 1}}, "Clipping and distortion").status
        == routing.FIRED
    )
    assert (
        verdict({"clipping": {"clipped_run_count": 0}}, "Clipping and distortion").status
        == routing.CLEAR
    )


def test_bandwidth_threshold():
    rule = "High-frequency loss or artificial upsampling"
    assert verdict({"bandwidth": {"cutoff_to_nyquist": 0.59}}, rule).status == routing.FIRED
    assert verdict({"bandwidth": {"cutoff_to_nyquist": 0.60}}, rule).status == routing.CLEAR


def test_rt60_threshold_and_window_count_in_detail():
    fired = verdict({"rt60": {"rt60_s": 0.9, "windows": 7}}, "High reverberation")
    assert fired.status == routing.FIRED
    assert "7 decay window" in fired.detail
    assert verdict({"rt60": {"rt60_s": 0.8, "windows": 7}}, "High reverberation").status == routing.CLEAR


def test_multiple_speakers_threshold():
    assert verdict({"speakers": {"speaker_count": 2}}, "Multiple speakers").status == routing.FIRED
    assert verdict({"speakers": {"speaker_count": 1}}, "Multiple speakers").status == routing.CLEAR


def test_high_pitch_follows_the_f0_flag():
    fired = {"f0": {"high_pitch": True, "f0_median_hz": 280.0}}
    assert verdict(fired, "High fundamental pitch").status == routing.FIRED
    clear = {"f0": {"high_pitch": False, "f0_median_hz": 120.0}}
    assert verdict(clear, "High fundamental pitch").status == routing.CLEAR


def test_every_fired_rule_recommends_something():
    results = {
        "snr": {"snr_db": 3.0, "method": "vad"},
        "format": {
            "sample_rate_hz": 8000,
            "channels": 2,
            "duration_s": 1.0,
            "heavy_compression": True,
            "codec": "gsm",
        },
        "density": {"silence_ratio": 0.9},
        "truncation": {"hallucination_risk": True, "duration_s": 1.0},
        "overlap": {"overlap_ratio": 0.5},
        "levels": {"rms_dbfs": -40.0},
        "clipping": {"clipped_run_count": 12},
        "bandwidth": {"cutoff_to_nyquist": 0.2, "effective_cutoff_khz": 3.4},
        "rt60": {"rt60_s": 1.4, "windows": 5},
        "speakers": {"speaker_count": 3},
        "f0": {"high_pitch": True, "f0_median_hz": 300.0},
    }
    verdicts = routing.evaluate(results)
    assert all(v.status == routing.FIRED for v in verdicts)
    assert all(v.action.strip() for v in verdicts)
