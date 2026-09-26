"""Drive the real app headlessly, so a broken tab fails here and not in a demo.

These use Streamlit's own test harness, which runs ``app.py`` as a script and
exposes the resulting widget tree. They deliberately do not assert on layout,
only that every tab renders without raising and that the numbers reaching the
screen are the ones the analyses produced.

A short generated wav keeps them fast and keeps them from depending on any
audio file being present on this machine.
"""

from __future__ import annotations

import os
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_dashboard import routing  # noqa: E402

APP = ROOT / "app.py"


@pytest.fixture(scope="module")
def wav_path(tmp_path_factory) -> Path:
    """Two seconds of tone-plus-noise with a silent gap in the middle.

    The gap matters: it gives voice activity detection something to find, so
    the analyses that depend on it have a real answer rather than a decline.
    """
    sr = 16_000
    t = np.arange(sr * 3) / sr
    tone = sum(0.3 / k * np.sin(2 * np.pi * 150 * k * t) for k in (1, 2, 3))
    rng = np.random.default_rng(0)
    signal = tone + rng.normal(0, 0.01, len(t))
    signal[sr : 2 * sr] *= 0.01  # a quiet second in the middle

    path = tmp_path_factory.mktemp("audio") / "sample.wav"
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes((np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def run_app(wav_path: Path, picks: list[str]) -> AppTest:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()

    at.text_input[0].set_value(str(wav_path)).run()

    for checkbox in at.checkbox:
        if checkbox.key and checkbox.key.startswith("pick_"):
            checkbox.set_value(checkbox.key.removeprefix("pick_") in picks)
    at.run()

    at.button[0].click().run()
    return at


def rendered_metrics(at: AppTest) -> dict[str, str]:
    """Every metric row that actually reached a table, by label."""
    metrics: dict[str, str] = {}
    for element in at.dataframe:
        frame = element.value
        if "Metric" not in frame.columns:
            continue
        for row in frame.to_dict("records"):
            metrics[row["Metric"]] = row["Value"]
    return metrics


def test_app_loads_with_no_file_selected():
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
    assert any("Upload an audio file" in m.value for m in at.info)


def test_running_the_fast_analyses_renders_every_tab(wav_path):
    at = run_app(wav_path, ["format", "levels", "clipping", "bandwidth"])
    assert not at.exception

    # The format table reached the screen with the rate we wrote into the file.
    metrics = rendered_metrics(at)
    assert metrics["Sample rate"] == "16"
    assert metrics["Channels"] == "1"


def all_markdown(at: AppTest) -> str:
    """Everything the page wrote as markdown, including the card HTML."""
    return "\n".join(
        element.value for element in at.markdown if isinstance(element.value, str)
    )


def test_metrics_are_shown_as_interpreted_cards(wav_path):
    """A number on its own is not the deliverable; the judgement about it is."""
    at = run_app(wav_path, ["format", "levels"])
    assert not at.exception

    body = all_markdown(at)
    # Plain-English names rather than the result keys.
    assert "Recording bandwidth" in body
    assert "Average loudness" in body
    # Each card carries a status chip, which is what makes the tab readable
    # without knowing what dBFS means.
    assert "border-radius:999px" in body
    assert "wideband" in body  # the 16 kHz fixture, classified


def test_the_raw_numbers_remain_available_behind_the_cards(wav_path):
    at = run_app(wav_path, ["format"])
    metrics = rendered_metrics(at)
    assert metrics["Sample rate"] == "16"


def test_the_headline_never_lets_unrun_checks_read_as_passes(wav_path):
    """Nothing fires on the clean fixture, but most rules were not evaluated.

    The success line is only allowed to speak for the rules that ran, and the
    count of the rest has to appear next to it.
    """
    at = run_app(wav_path, ["format", "levels", "clipping"])
    assert not at.exception

    assert any("Nothing stands out" in m.value for m in at.success)
    assert any(
        "could not be evaluated" in c.value and "not passes" in c.value
        for c in at.caption
    )


def test_routing_shows_an_explanation_panel(wav_path):
    at = run_app(wav_path, ["format"])
    assert not at.exception

    body = all_markdown(at)
    # The panel opens on whatever most deserves attention without being asked.
    assert "**What this is**" in body
    assert "**What the router would do**" in body
    assert any("simulation" in m.value for m in at.info)


def test_the_explanation_panel_follows_the_rule_you_ask_about(wav_path):
    at = run_app(wav_path, ["format", "levels"])
    assert not at.exception

    # The panel is the last subheader on the page, in the right-hand column.
    opened_on = at.subheader[-1].value

    explains = [b for b in at.button if b.label == "Explain"]
    assert len(explains) == len(routing.evaluate({})), "every rule needs a button"
    explains[-1].click().run()

    assert at.subheader[-1].value != opened_on
    assert at.subheader[-1].value == at.session_state["routing_focus"]


def test_signal_plots_come_with_a_how_to_read_it_panel(wav_path):
    at = run_app(wav_path, ["format"])
    assert not at.exception
    body = all_markdown(at)
    assert "A normal one looks like this" in body
    assert "Something is wrong when" in body


def test_transcript_tab_prompts_rather_than_billing(wav_path):
    """With Transcribe unticked, the transcript tab must explain itself."""
    at = run_app(wav_path, ["format"])
    assert not at.exception
    assert any("Tick Transcribe" in m.value for m in at.info)


def test_routing_reports_unevaluated_rules_separately(wav_path):
    """Rules whose input was not run must not be presented as passes."""
    at = run_app(wav_path, ["format"])
    assert not at.exception
    headers = [h.value for h in at.subheader]
    assert any("could not be evaluated" in h for h in headers)


def test_vad_dependent_analyses_are_wired_up(wav_path):
    """The dependency chain reaches the screen.

    The fixture is a synthetic tone, which Silero rightly declines to call
    speech, so this asserts on plumbing rather than on detection quality:
    ticking SNR alongside VAD must take the VAD path rather than the
    percentile fallback, and the density table must render.
    """
    at = run_app(wav_path, ["vad", "density", "snr", "truncation"])
    assert not at.exception

    metrics = rendered_metrics(at)
    assert metrics["Method"] == "vad"
    assert "Longest internal gap" in metrics


# Real speech, when this machine happens to have a corpus to point at.
# Detection quality cannot be checked against a synthesised tone, and
# generating speech in a test is not worth the dependency. Set AUDIO_CORPUS to
# a directory holding the two files named below; without it these tests skip.
CORPUS = Path(os.environ.get("AUDIO_CORPUS", "")) if os.environ.get("AUDIO_CORPUS") else None
SPEECH = CORPUS / os.environ.get("AUDIO_CORPUS_SPEECH", "speech.mp3") if CORPUS else None

# The transcript tests need a transcript. They run only against a file that is
# already in the cache, so no test can ever call the API or bill for one.
CALL = CORPUS / os.environ.get("AUDIO_CORPUS_CALL", "call.mp3") if CORPUS else None


def call_is_cached() -> bool:
    if CALL is None or not CALL.exists():
        return False
    from audio_dashboard import asr

    return asr.is_cached(CALL, asr.TranscribeConfig())


@pytest.mark.skipif(not call_is_cached(), reason="no cached transcript to read")
def test_transcript_offers_all_three_ways_of_choosing_the_cutoff():
    at = run_app(CALL, ["transcript"])
    assert not at.exception

    modes = [r for r in at.radio if "cutoff" in r.label]
    assert modes, "the threshold-mode control is missing"
    # Options come back formatted, which is what the user actually reads.
    assert set(modes[0].options) == {
        "Fixed confidence value",
        "Target a highlight rate",
        "Target a rate, with a floor",
    }


@pytest.mark.skipif(not call_is_cached(), reason="no cached transcript to read")
def test_the_fixed_cutoff_defaults_high_enough_to_mark_anything():
    """At 0.6 the panel looks reassuring on a file that is not.

    The median word comes back around 0.998, so a low cutoff marks almost
    nothing and the transcript reads as clean whatever the audio was like.
    """
    at = run_app(CALL, ["transcript"])
    sliders = [s for s in at.slider if "Flag words below" in s.label]
    assert sliders and sliders[0].value == 0.9


@pytest.mark.skipif(not call_is_cached(), reason="no cached transcript to read")
def test_the_transcript_panel_reports_what_the_cutoff_is_doing():
    at = run_app(CALL, ["transcript"])
    assert any(
        "Marking" in c.value and "of words" in c.value and "below" in c.value
        for c in at.caption
    )


@pytest.mark.skipif(
    SPEECH is None or not SPEECH.exists(), reason="AUDIO_CORPUS not set"
)
def test_real_speech_produces_a_plausible_analysis():
    at = run_app(SPEECH, ["format", "vad", "density", "snr", "levels"])
    assert not at.exception

    metrics = rendered_metrics(at)
    assert metrics["Method"] == "vad"
    # A recording that is mostly someone talking.
    assert float(metrics["Speech share of file"]) > 0.5
    assert int(metrics["Speech segments"]) >= 1
    # Noisy by construction, but not so noisy the estimator falls apart.
    assert -5.0 < float(metrics["Signal-to-noise ratio"]) < 30.0
