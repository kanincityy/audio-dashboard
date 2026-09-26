"""The interpretation layer: bands, statuses, and the promise not to guess.

The load-bearing test here is the last one. A card that colours itself green
because nobody measured anything is worse than no card at all, and it is the
exact mistake this whole module exists to avoid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_dashboard import interpret, registry, routing  # noqa: E402
from audio_dashboard.features import quality  # noqa: E402


# ------------------------------------------------------------- the catalogue


def test_every_metric_names_a_registered_analysis():
    for metric in interpret.METRICS:
        assert metric.analysis in registry.BY_NAME, metric.key


def test_every_metric_explains_itself():
    for metric in interpret.METRICS:
        assert metric.what.strip(), metric.key
        assert metric.why.strip(), metric.key
        assert metric.source.strip(), metric.key
        # The name is for someone who does not know the jargon term, so it
        # must not simply be the jargon term.
        assert metric.name != metric.key


def test_no_metric_is_registered_twice():
    seen = [(m.analysis, m.key) for m in interpret.METRICS]
    assert len(seen) == len(set(seen))


def test_numeric_metrics_have_a_closed_scale_and_an_open_last_band():
    for metric in interpret.METRICS:
        if metric.kind != "number":
            continue
        assert metric.bands, metric.key
        assert metric.scale is not None, metric.key
        uppers = [b.upper for b in metric.bands]
        # Only the last band is open-ended, so no value falls through.
        assert uppers[-1] is None, metric.key
        assert all(u is not None for u in uppers[:-1]), metric.key
        # Ascending, so the first matching band is the right one.
        closed = [u for u in uppers[:-1]]
        assert closed == sorted(closed), metric.key


def test_every_band_status_is_one_the_ui_can_colour():
    allowed = {interpret.GOOD, interpret.CAUTION, interpret.PROBLEM}
    for metric in interpret.METRICS:
        for band in metric.bands:
            assert band.status in allowed, f"{metric.key}: {band.status}"
            assert band.label.strip(), metric.key


# ---------------------------------------------------------------- classifying


def metric(analysis: str, key: str) -> interpret.Metric:
    for m in interpret.BY_ANALYSIS[analysis]:
        if m.key == key:
            return m
    raise KeyError(key)


def test_a_value_on_a_band_edge_belongs_to_that_band():
    snr = metric("snr", "snr_db")
    # The routing rule fires below 12, so 12 itself is the top of the noisy
    # band and must not read as clean.
    assert interpret.classify(snr, routing.SNR_DB)[0] == interpret.PROBLEM
    assert interpret.classify(snr, routing.SNR_DB - 0.01)[0] == interpret.PROBLEM
    assert interpret.classify(snr, routing.SNR_DB + 0.01)[0] == interpret.CAUTION


def test_the_open_ended_band_catches_anything():
    snr = metric("snr", "snr_db")
    assert interpret.classify(snr, 10_000.0)[0] == interpret.GOOD


def test_a_value_below_the_scale_still_classifies():
    snr = metric("snr", "snr_db")
    assert interpret.classify(snr, -100.0)[0] == interpret.PROBLEM


@pytest.mark.parametrize(
    ("analysis", "key", "value", "expected"),
    [
        ("format", "sample_rate_khz", 8.0, interpret.PROBLEM),
        ("format", "sample_rate_khz", 16.0, interpret.GOOD),
        ("levels", "rms_dbfs", -35.0, interpret.PROBLEM),
        ("levels", "rms_dbfs", -18.0, interpret.GOOD),
        ("density", "silence_ratio", 0.9, interpret.PROBLEM),
        ("density", "silence_ratio", 0.2, interpret.GOOD),
        ("overlap", "overlap_ratio", 0.4, interpret.PROBLEM),
        ("rt60", "rt60_s", 1.5, interpret.PROBLEM),
        ("rt60", "rt60_s", 0.2, interpret.GOOD),
        ("f0", "f0_median_hz", 300.0, interpret.PROBLEM),
        ("f0", "f0_median_hz", 120.0, interpret.GOOD),
    ],
)
def test_known_values_land_where_notes_md_says_they_should(
    analysis, key, value, expected
):
    assert interpret.classify(metric(analysis, key), value)[0] == expected


def test_flags_carry_a_status_each_way_round():
    narrowband = metric("format", "narrowband")
    assert interpret.classify(narrowband, True)[0] == interpret.PROBLEM
    assert interpret.classify(narrowband, False)[0] == interpret.GOOD

    # Not every flag is bad when true.
    single = metric("speakers", "single_speaker")
    assert interpret.classify(single, True)[0] == interpret.GOOD
    assert interpret.classify(single, False)[0] == interpret.CAUTION


def test_text_metrics_are_neither_good_nor_bad():
    codec = metric("format", "codec")
    status, label = interpret.classify(codec, "gsm")
    assert status == interpret.INFO
    assert label == "gsm"


# ------------------------------------------------ not measured is not measured


def test_a_missing_value_is_unknown_not_good():
    assert interpret.classify(metric("snr", "snr_db"), None)[0] == interpret.UNKNOWN


@pytest.mark.parametrize(
    "results",
    [
        {},
        {"snr": {"error": "RuntimeError: boom"}},
        {"snr": {"usable": False, "reason": "not enough frames"}},
        {"snr": {"usable": True}},
    ],
    ids=["not run", "failed", "declined", "ran but no key"],
)
def test_an_unmeasurable_metric_never_reads_as_fine(results):
    value, status, label = interpret.read(results, metric("snr", "snr_db"))
    assert value is None
    assert status == interpret.UNKNOWN
    assert label == "not measured"


def test_reading_agrees_with_the_routing_rules_about_what_counts_as_absent():
    """One definition of absent, or a card and a rule could disagree."""
    results = {"snr": {"usable": False, "snr_db": 3.0}}
    assert routing.metric_value(results, "snr", "snr_db") is None
    assert interpret.read(results, metric("snr", "snr_db"))[0] is None


# ------------------------------------------------------------ shared constants


def test_thresholds_are_shared_with_routing_rather_than_restated():
    snr = metric("snr", "snr_db")
    assert snr.bands[0].upper == routing.SNR_DB

    silence = metric("density", "silence_ratio")
    assert silence.bands[1].upper == routing.SILENCE_RATIO

    pitch = metric("f0", "f0_median_hz")
    assert pitch.bands[1].upper == routing.HIGH_PITCH_HZ

    band = metric("bandwidth", "cutoff_to_nyquist")
    assert band.bands[0].upper == routing.BANDWIDTH_RATIO

    rt = metric("rt60", "rt60_s")
    assert rt.bands[1].upper == routing.RT60_S


def test_the_cliff_band_matches_the_detector_that_produces_it():
    """interpret stays pure, so this is where the two constants are compared."""
    cliff = metric("bandwidth", "cliff_drop_db")
    assert cliff.bands[0].upper == quality.CLIFF_DROP_DB


# ---------------------------------------------------------------- rule mapping


def test_every_routing_rule_has_an_explanation_to_show():
    for verdict in routing.evaluate({}):
        assert verdict.rule in interpret.RULE_METRICS, verdict.rule


def test_every_mapped_rule_resolves_to_a_real_metric():
    for rule in interpret.RULE_METRICS:
        assert interpret.metric_for_rule(rule) is not None, rule


def test_an_unmapped_rule_returns_nothing_rather_than_raising():
    assert interpret.metric_for_rule("Not a rule") is None


# -------------------------------------------------------------------- display


def test_values_are_formatted_to_the_precision_the_metric_asked_for():
    assert interpret.format_value(metric("snr", "snr_db"), 12.3456) == "12.35"
    assert interpret.format_value(metric("speakers", "speaker_count"), 4) == "4"
    assert interpret.format_value(metric("snr", "snr_db"), None) == "—"


def test_flags_format_as_words_rather_than_true_and_false():
    narrowband = metric("format", "narrowband")
    assert interpret.format_value(narrowband, True) == "telephony"
    assert interpret.format_value(narrowband, False) == "wideband"


# ---------------------------------------------------------------- plot guides


def test_every_plot_the_signal_tab_draws_has_a_guide():
    assert set(interpret.PLOTS) == {"waveform", "spectrum", "pitch"}
    for key, guide in interpret.PLOTS.items():
        assert guide.caption.strip(), key
        assert guide.normal.strip(), key
        assert guide.abnormal.strip(), key
