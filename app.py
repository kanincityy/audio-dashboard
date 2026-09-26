"""Audio analysis dashboard.

Upload a file, pick which analyses to run, and look at the signal, the metrics,
the transcript and the routing verdicts together. The point is to find out
whether the signals in notes.md are extractable and interpretable on real
audio, before any of this becomes an automatic routing layer.

Interpretable is half the job, so every number on screen is shown against the
range it is normally expected to sit in, and says in plain words what it means
and why an ASR model cares. That text lives in ``interpret``, not here, so the
metric card and the routing panel cannot drift apart.

Run with: uv run streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from audio_dashboard import (  # noqa: E402
    asr,
    audio_io,
    cache,
    interpret,
    registry,
    render,
    routing,
    stats,
)

load_dotenv()

UPLOAD_DIR = ROOT / "uploads"
# Above this, inlining the audio as a data URI in the transcript iframe stops
# being reasonable, and click-to-seek is dropped.
MAX_INLINE_BYTES = 30 * 1024 * 1024

# Chip and band colours. Fixed hex rather than theme variables because they
# have to mean the same thing in both themes; these are the 600 shades, which
# carry white text and stay legible on a light or a dark background.
STATUS_COLOUR = {
    interpret.GOOD: "#16a34a",
    interpret.CAUTION: "#d97706",
    interpret.PROBLEM: "#dc2626",
    interpret.UNKNOWN: "#94a3b8",
    interpret.INFO: "#64748b",
}

st.set_page_config(page_title="Audio analysis dashboard", layout="wide")


# ---------------------------------------------------------------- data access


@st.cache_resource(show_spinner=False)
def load_bundle(path_str: str, digest: str) -> audio_io.AudioBundle:
    """Decode once per file per session.

    ``digest`` is unused in the body but part of the cache key, so replacing a
    file at the same path is not served from a stale entry. cache_resource
    rather than cache_data because the bundle holds numpy arrays that there is
    no reason to pickle.
    """
    return audio_io.load(Path(path_str))


def store_upload(uploaded) -> Path:
    """Write an upload to a stable path keyed by its content.

    A stable path is what lets the feature cache, the transcript cache and the
    audio player all agree about what file this is across reruns.
    """
    UPLOAD_DIR.mkdir(exist_ok=True)
    data = uploaded.getvalue()
    import hashlib

    digest = hashlib.sha256(data).hexdigest()[:16]
    dest = UPLOAD_DIR / f"{digest}{Path(uploaded.name).suffix.lower()}"
    if not dest.exists():
        dest.write_bytes(data)
    return dest


# -------------------------------------------------------------------- helpers


def flag_icon(value: bool | None) -> str:
    if value is None:
        return "—"
    return "flagged" if value else "ok"


def metric_rows(result: dict, keys: list[tuple[str, str, str]]) -> list[dict]:
    """Turn a result dict into display rows: label, value, unit.

    Values are rendered as text rather than left as mixed types. A column
    holding both 12.5 and "flagged" has no single Arrow type, and Streamlit
    only survives it by silently coercing the whole column — which turns the
    flags into nulls on the screen.
    """
    rows = []
    for key, label, unit in keys:
        if key not in result:
            continue
        value = result[key]
        if isinstance(value, bool):
            text = flag_icon(value)
        elif value is None:
            text = "—"
        elif isinstance(value, float):
            text = f"{value:.3f}".rstrip("0").rstrip(".")
        else:
            text = str(value)
        rows.append({"Metric": label, "Value": text, "Unit": unit})
    return rows


def show_block(name: str, results: dict, keys: list[tuple[str, str, str]]) -> bool:
    """Render one analysis block as a raw table, or say why there is nothing.

    Returns whether real values were rendered, so callers can decide whether to
    draw the extras that go with them.
    """
    analysis = registry.BY_NAME[name]
    block = results.get(name)

    if block is None:
        st.caption(f"**{analysis.label}** — not run.")
        return False
    if "error" in block:
        st.error(f"**{analysis.label}** failed: {block['error']}")
        return False
    if block.get("usable") is False:
        reason = block.get("reason", "no usable result")
        st.caption(f"**{analysis.label}** — {reason}")
        return False

    st.markdown(f"**{analysis.label}**")
    rows = metric_rows(block, keys)
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    return True


def block_note(name: str, results: dict) -> str | None:
    """Why an analysis has nothing to show, or None when it has.

    Same three cases show_block handles, phrased for somebody who did not
    choose to run it and needs to know that is why the space is empty.
    """
    analysis = registry.BY_NAME[name]
    block = results.get(name)
    if block is None:
        return f"**{analysis.label}** — not run. Tick it in the sidebar."
    if "error" in block:
        return f"**{analysis.label}** failed: {block['error']}"
    if block.get("usable") is False:
        return f"**{analysis.label}** — {block.get('reason', 'no usable result')}"
    return None


# ------------------------------------------------------------- card rendering


def chip(status: str, label: str) -> str:
    colour = STATUS_COLOUR[status]
    return (
        f'<span style="background:{colour};color:#fff;border-radius:999px;'
        f'padding:0.1rem 0.55rem;font-size:0.72rem;font-weight:600;'
        f'white-space:nowrap;">{label}</span>'
    )


def band_bar(metric: interpret.Metric, value) -> str:
    """A strip of the metric's scale, coloured by band, marked at the value.

    The marker uses currentColor for its halo so it stays visible whichever
    theme the page is in — the band fills are translucent and the halo picks up
    the surrounding text colour, which is dark on light and light on dark.
    """
    if metric.scale is None or metric.kind != "number":
        return ""
    lo, hi = metric.scale
    span = (hi - lo) or 1.0

    parts: list[str] = []
    left = lo
    for band in metric.bands:
        right = hi if band.upper is None else min(max(float(band.upper), lo), hi)
        if right > left:
            width = 100.0 * (right - left) / span
            parts.append(
                f'<div style="width:{width:.2f}%;'
                f"background:{STATUS_COLOUR[band.status]};opacity:0.42;\"></div>"
            )
        left = max(left, right)

    if value is not None:
        pos = 100.0 * (min(max(float(value), lo), hi) - lo) / span
        parts.append(
            f'<div style="position:absolute;left:{pos:.2f}%;top:0;bottom:0;'
            'width:2px;background:currentColor;opacity:0.85;"></div>'
        )

    return (
        '<div style="position:relative;display:flex;height:10px;'
        'border-radius:3px;overflow:hidden;margin:0.15rem 0 0.1rem 0;">'
        + "".join(parts)
        + "</div>"
        '<div style="display:flex;justify-content:space-between;opacity:0.55;'
        'font-size:0.7rem;">'
        f"<span>{lo:g}</span><span>{hi:g} {metric.unit}</span></div>"
    )


def render_card(metric: interpret.Metric, value, status: str, label: str) -> None:
    """One metric, as a card: what it is, what it says, and whether that is bad."""
    with st.container(border=True):
        st.markdown(
            f"**{metric.name}**",
            help=f"{metric.what}\n\n*Where the band comes from: {metric.source}.*",
        )
        st.markdown(
            '<div style="display:flex;align-items:baseline;gap:0.45rem;'
            'flex-wrap:wrap;margin-bottom:0.15rem;">'
            f'<span style="font-size:1.55rem;font-weight:600;line-height:1.1;">'
            f"{interpret.format_value(metric, value)}</span>"
            f'<span style="opacity:0.6;font-size:0.85rem;">{metric.unit}</span>'
            f'<span style="margin-left:auto;">{chip(status, label)}</span></div>'
            + band_bar(metric, value),
            unsafe_allow_html=True,
        )
        st.caption(metric.why)
        st.caption(f"`{metric.analysis}.{metric.key}`")


def render_cards(name: str, results: dict, per_row: int = 3) -> None:
    """Every card for one analysis, or a line saying why there are none."""
    note = block_note(name, results)
    if note is not None:
        st.caption(note)
        return

    analysis = registry.BY_NAME[name]
    block = results[name]
    st.markdown(f"**{analysis.label}**", help=analysis.note)

    present = [
        m
        for m in interpret.BY_ANALYSIS.get(name, ())
        if block.get(m.key) is not None
        # ffprobe reports an empty string or a zero for fields a container
        # does not carry. A card saying "Bit depth: 0" is worse than no card.
        and not (m.kind == "text" and str(block[m.key]).strip() in ("", "0"))
    ]
    cards = [m for m in present if not interpret.is_compact(m)]
    chips = [m for m in present if interpret.is_compact(m)]

    if not present:
        st.caption("Ran, but produced nothing worth showing.")
        return

    for start in range(0, len(cards), per_row):
        columns = st.columns(per_row)
        for column, metric in zip(columns, cards[start : start + per_row], strict=False):
            status, label = interpret.classify(metric, block.get(metric.key))
            with column:
                render_card(metric, block.get(metric.key), status, label)

    if chips:
        # The threshold applied to a number shown above, which is what the
        # routing layer actually reads. Worth showing, not worth a card each.
        pieces = []
        for metric in chips:
            status, label = interpret.classify(metric, block.get(metric.key))
            pieces.append(f"{metric.name} {chip(status, label)}")
        st.markdown(
            '<div style="display:flex;flex-wrap:wrap;gap:0.5rem 1.1rem;'
            'align-items:center;font-size:0.8rem;opacity:0.9;margin:0.2rem 0 '
            '0.6rem 0;">'
            + "".join(f'<span style="white-space:nowrap;">{p}</span>' for p in pieces)
            + "</div>",
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------- main


def main() -> None:
    st.title("Audio analysis dashboard")

    with st.sidebar:
        st.header("Audio")
        uploaded = st.file_uploader(
            "Upload an audio file",
            type=[s.lstrip(".") for s in sorted(audio_io.AUDIO_SUFFIXES)],
        )

        st.caption("Or point at a file already on disk:")
        path_input = st.text_input("Path", value="", label_visibility="collapsed")

    path: Path | None = None
    if uploaded is not None:
        path = store_upload(uploaded)
    elif path_input.strip():
        candidate = Path(path_input.strip()).expanduser()
        if candidate.is_file():
            path = candidate
        else:
            st.sidebar.error("No file at that path.")

    if path is None:
        st.info("Upload an audio file, or give a path, to begin.")
        st.markdown(
            "Analyses come from `notes.md`. Ticking one automatically pulls in "
            "whatever it depends on — voice activity detection in particular is "
            "shared by signal-to-noise, reverberation, pitch and speech density, "
            "and only ever runs once per file."
        )
        return

    digest = cache.file_digest(path)

    # ------------------------------------------------------------ the picker
    with st.sidebar:
        st.header("Analyses")
        selected: list[str] = []

        for group, group_label in registry.GROUPS:
            entries = [a for a in registry.ANALYSES if a.group == group]
            if not entries:
                continue
            st.markdown(f"**{group_label}**")
            for analysis in entries:
                label = analysis.label
                if analysis.cost == "slow":
                    label += "  ·  slow"
                elif analysis.cost == "paid":
                    label += "  ·  billed"

                default = analysis.cost == "fast" and analysis.name != "codeswitch"
                if analysis.name == "transcript":
                    cached = asr.is_cached(path, asr.TranscribeConfig())
                    default = cached

                ticked = st.checkbox(
                    label,
                    value=default,
                    key=f"pick_{analysis.name}",
                    help=analysis.note,
                )
                if ticked:
                    selected.append(analysis.name)

                if analysis.name == "transcript":
                    st.caption(
                        "Cached — no API call."
                        if asr.is_cached(path, asr.TranscribeConfig())
                        else "Not cached — this will call the API and bill for it."
                    )

        extras = registry.implied(selected) if selected else []
        if extras:
            st.info(
                "Also running, because the selection needs it: "
                + ", ".join(registry.BY_NAME[n].label for n in extras)
            )

        force = st.checkbox("Ignore cached results", value=False)
        run = st.button("Run", type="primary", disabled=not selected)

    st.caption(f"`{path.name}` · digest `{digest}`")

    if run:
        st.session_state.pop("results", None)
        bundle = load_bundle(str(path), digest)
        order = registry.resolve(selected)
        progress = st.progress(0.0, text="Starting")
        done = {"n": 0}

        def on_start(name: str, analysis: registry.Analysis) -> None:
            progress.progress(
                done["n"] / len(order), text=f"Running {analysis.label}…"
            )
            done["n"] += 1

        results = registry.run(
            bundle, digest, selected, on_start=on_start, force=force
        )
        progress.empty()
        st.session_state["results"] = results
        st.session_state["results_digest"] = digest

    if st.session_state.get("results_digest") != digest:
        st.info("Pick your analyses in the sidebar and press Run.")
        return

    results = st.session_state["results"]
    bundle = load_bundle(str(path), digest)

    failed = [n for n, r in results.items() if isinstance(r, dict) and "error" in r]
    if failed:
        st.warning(
            "These analyses failed and are shown as errors below: "
            + ", ".join(registry.BY_NAME[n].label for n in failed)
        )

    tab_signal, tab_metrics, tab_transcript, tab_routing = st.tabs(
        ["Signal", "Metrics", "Transcript", "Routing"]
    )

    with tab_signal:
        render_signal(bundle, results)
    with tab_metrics:
        render_metrics(results)
    with tab_transcript:
        render_transcript(path, results)
    with tab_routing:
        render_routing(results)


# -------------------------------------------------------------------- signal


def plot_guide(key: str) -> None:
    """The caption and the how-to-read-it panel for one plot."""
    guide = interpret.PLOTS[key]
    st.caption(guide.caption)
    with st.expander("How to read this"):
        st.markdown(f"**A normal one looks like this.** {guide.normal}")
        st.markdown(f"**Something is wrong when.** {guide.abnormal}")


def render_signal(bundle: audio_io.AudioBundle, results: dict) -> None:
    st.audio(str(bundle.path))

    samples = bundle.samples
    # Plotting every sample of a ten-minute file helps nobody and is slow.
    # Min/max per bucket keeps the envelope honest where plain decimation
    # would hide short transients — including the clipped peaks we care about.
    target = 4000
    step = max(1, len(samples) // target)
    usable = (len(samples) // step) * step
    buckets = samples[:usable].reshape(-1, step)
    lows, highs = buckets.min(axis=1), buckets.max(axis=1)
    times = np.arange(len(lows)) * step / bundle.sr

    fig, ax = plt.subplots(figsize=(12, 2.6))
    ax.fill_between(times, lows, highs, linewidth=0, color="#2563eb")

    vad = results.get("vad") or {}
    for start, end in (vad.get("speech_spans") or []):
        ax.axvspan(start, end, color="#16a34a", alpha=0.14, linewidth=0)

    clip = results.get("clipping") or {}
    for pos in (clip.get("run_positions_s") or [])[:500]:
        ax.axvline(pos, color="#dc2626", alpha=0.6, linewidth=0.8)

    ax.set_xlim(0, bundle.duration)
    ax.set_ylim(-1.02, 1.02)
    ax.set_xlabel("seconds")
    ax.set_ylabel("amplitude")
    ax.set_title("Waveform — green is detected speech, red marks clipping runs")
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    plot_guide("waveform")
    if not vad:
        st.caption(
            "No green shading because voice activity detection was not run, "
            "not because no speech was found."
        )

    band = results.get("bandwidth") or {}
    if band.get("usable"):
        st.divider()
        fig, ax = plt.subplots(figsize=(12, 2.8))
        freqs = np.array(band["spectrum_freqs_hz"]) / 1000.0
        ax.plot(freqs, band["spectrum_db"], color="#2563eb", linewidth=1.0)
        ax.axvline(
            band["effective_cutoff_khz"],
            color="#dc2626",
            linestyle="--",
            label=f"cutoff {band['effective_cutoff_khz']} kHz"
            + (
                f" (drop {band['cliff_drop_db']} dB)"
                if band["cliff_found"]
                else " (no cliff)"
            ),
        )
        ax.set_xlabel("kHz")
        ax.set_ylabel("dB")
        ax.set_title("Average power spectrum — where the energy actually stops")
        ax.legend(loc="upper right", fontsize=8)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        plot_guide("spectrum")

    pitch = results.get("f0") or {}
    if pitch.get("usable"):
        st.divider()
        fig, ax = plt.subplots(figsize=(12, 2.2))
        track = [np.nan if v is None else v for v in pitch["track_hz"]]
        ax.plot(pitch["track_times_s"], track, color="#7c3aed", linewidth=1.0)
        ax.axhline(
            pitch["f0_median_hz"],
            color="#94a3b8",
            linestyle=":",
            label=f"median {pitch['f0_median_hz']} Hz",
        )
        ax.set_xlim(0, bundle.duration)
        ax.set_xlabel("seconds")
        ax.set_ylabel("Hz")
        ax.set_title("Fundamental frequency over speech")
        ax.legend(loc="upper right", fontsize=8)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        plot_guide("pitch")

        by_speaker = pitch.get("f0_by_speaker_hz")
        if by_speaker:
            st.caption("Median pitch per speaker — the figure that actually routes:")
            st.dataframe(
                [{"Speaker": k, "Median pitch (Hz)": v} for k, v in by_speaker.items()],
                hide_index=True,
                width="stretch",
            )


# ------------------------------------------------------------------- metrics


def headline(results: dict) -> None:
    """One plain sentence about the file, before any of the numbers."""
    verdicts = routing.evaluate(results)
    fired = [v for v in verdicts if v.status == routing.FIRED]
    unknown = [v for v in verdicts if v.status == routing.UNKNOWN]

    if fired:
        phrases = []
        for verdict in fired:
            metric = interpret.metric_for_rule(verdict.rule)
            if metric is None:
                phrases.append(verdict.rule.lower())
                continue
            _, label = interpret.classify(metric, verdict.observed)
            phrases.append(f"{metric.name.lower()}: {label}")
        st.warning(
            "**What stands out about this file:** "
            + "; ".join(phrases)
            + ". These are the things most likely to explain a transcription "
            "error here — the Routing tab explains each one."
        )
    elif len(unknown) < len(verdicts):
        st.success(
            "Nothing stands out among the checks that could be evaluated. "
            "Transcription errors on this file are less likely to be the "
            "recording's fault."
        )

    if unknown:
        st.caption(
            f"{len(unknown)} of {len(verdicts)} checks could not be evaluated, "
            "because the analysis they need was not run. Those are not passes."
        )


def render_metrics(results: dict) -> None:
    headline(results)
    st.caption(
        "Each card shows the measurement, where it sits on the range it is "
        "normally expected to fall in, and what it means for transcription. "
        "Hover the name for what is actually being measured."
    )

    st.subheader("1. Format and technical metadata")
    st.caption("What the file is, read from the container without decoding it.")
    render_cards("format", results)

    st.subheader("2. Signal quality and acoustic metrics")
    st.caption(
        "How well the speech was captured. These explain why an engine "
        "dropped, substituted or invented words."
    )
    render_cards("levels", results)
    render_cards("clipping", results)
    render_cards("snr", results)
    snr_db = (results.get("snr") or {}).get("snr_db")
    if snr_db is not None and snr_db > 20:
        # Measured bias, from scripts/validate_corpus.py: at a constructed
        # 5 dB the estimator lands within 0.7 dB, at 10 dB within 1.8, and on
        # 30 dB files it reads 3 to 13 dB low. Breaths and room tone inside
        # the gaps are counted as noise, which is fine at the threshold that
        # matters and pessimistic on clean audio.
        st.caption(
            "Above about 20 dB this reads low — against the synthetic corpus "
            "it under-read a constructed 30 dB by 3 to 13 dB, because breaths "
            "and room tone in the gaps count as noise. Clean audio is at least "
            "as clean as this says. Near the 12 dB routing threshold it is "
            "accurate to under a decibel."
        )
    if (results.get("snr") or {}).get("method") == "percentile":
        st.caption(
            "Estimated by splitting frame energy at the 10th and 90th "
            "percentiles, because voice activity detection was not run. That "
            "assumes the quietest tenth of the file is noise and the loudest "
            "tenth is speech, which is wrong on continuous speech and wrong "
            "differently on mostly-silent audio. Tick voice activity "
            "detection for a real measurement."
        )
    render_cards("bandwidth", results)
    if not (results.get("bandwidth") or {}).get("cliff_found", True):
        st.caption(
            "No sharp cutoff found, so the file is treated as using its whole "
            "band. Speech spectra slope downwards on their own; only a near "
            "vertical drop indicates a resampler or codec cutoff."
        )
    render_cards("rt60", results)
    if (results.get("rt60") or {}).get("usable"):
        st.caption(
            "Blind estimate. There is no impulse response here, so this fits a "
            "decay to the tail after each speech offset and extrapolates from "
            "20 dB to 60 dB. With few usable windows the number means very "
            "little — check the window count before trusting it."
        )

    st.subheader("3. Speech structure and temporal dynamics")
    st.caption(
        "How the content is spoken. These are the metrics behind looping and "
        "invented text."
    )
    render_cards("density", results)
    render_cards("truncation", results)
    render_cards("tempo", results)
    render_cards("overlap", results)
    if (results.get("overlap") or {}).get("usable"):
        st.caption(
            "Read the overlap figure as a floor. Scored against the synthetic "
            "corpus it found none of the overlap that was built into mono "
            "files, so anything it does report is real but incomplete."
        )

    st.subheader("4. Speaker and demographic profiling")
    st.caption("Properties of the voice rather than the recording.")
    render_cards("speakers", results)
    speakers = results.get("speakers") or {}
    if speakers.get("talk_time_s"):
        st.dataframe(
            [
                {
                    "Speaker": label,
                    "Turns": speakers["turns_per_speaker"].get(label, 0),
                    "Talk time (s)": seconds,
                }
                for label, seconds in speakers["talk_time_s"].items()
            ],
            hide_index=True,
            width="stretch",
        )

    render_cards("f0", results)
    by_speaker = (results.get("f0") or {}).get("f0_by_speaker_hz")
    if by_speaker:
        st.caption(
            "Per speaker, which is the figure that matters — one high-pitched "
            "voice in a call is a routing signal that a file-wide median "
            "averages away."
        )
        st.dataframe(
            [{"Speaker": k, "Median F0 (Hz)": v} for k, v in by_speaker.items()],
            hide_index=True,
            width="stretch",
        )

    note = block_note("codeswitch", results)
    if note:
        st.caption(note)

    with st.expander("Show every raw value"):
        st.caption(
            "The same measurements untouched, including the ones with no "
            "interpretation attached."
        )
        render_raw_tables(results)


def render_raw_tables(results: dict) -> None:
    """The original per-analysis tables, kept for auditing the numbers."""
    show_block(
        "format",
        results,
        [
            ("sample_rate_khz", "Sample rate", "kHz"),
            ("channels", "Channels", ""),
            ("channel_layout", "Layout", ""),
            ("codec", "Codec", ""),
            ("bit_rate_kbps", "Bitrate", "kbps"),
            ("bit_depth", "Bit depth", "bits"),
            ("duration_s", "Duration", "s"),
            ("narrowband", "Narrowband (8 kHz or below)", ""),
            ("multichannel", "Multi-channel", ""),
            ("heavy_compression", "Heavy compression", ""),
        ],
    )
    show_block(
        "levels",
        results,
        [
            ("peak_dbfs", "Peak", "dBFS"),
            ("rms_dbfs", "RMS", "dBFS"),
            ("crest_factor_db", "Crest factor", "dB"),
            ("dynamic_range_db", "Dynamic range", "dB"),
        ],
    )
    show_block(
        "clipping",
        results,
        [
            ("clipped_sample_pct", "Samples at full scale", "%"),
            ("clipped_run_count", "Clipping runs (3+ samples)", ""),
            ("clipped_run_pct", "Audio inside clipping runs", "%"),
            ("longest_run_samples", "Longest run", "samples"),
        ],
    )
    show_block(
        "snr",
        results,
        [
            ("snr_db", "Signal-to-noise ratio", "dB"),
            ("speech_level_dbfs", "Speech level", "dBFS"),
            ("noise_floor_dbfs", "Noise floor", "dBFS"),
            ("method", "Method", ""),
        ],
    )
    show_block(
        "bandwidth",
        results,
        [
            ("effective_cutoff_khz", "Effective cutoff", "kHz"),
            ("nyquist_hz", "Nyquist", "Hz"),
            ("cutoff_to_nyquist", "Share of available band used", "ratio"),
            ("cliff_drop_db", "Drop across the cutoff", "dB"),
            ("cliff_found", "Sharp cutoff found", ""),
            ("likely_upsampled", "Likely upsampled", ""),
        ],
    )
    show_block(
        "rt60",
        results,
        [
            ("rt60_s", "RT60", "s"),
            ("rt60_iqr_s", "Spread across windows", "s"),
            ("windows", "Usable decay windows", ""),
            ("far_field", "Far-field recording", ""),
        ],
    )
    show_block(
        "density",
        results,
        [
            ("speech_ratio", "Speech share of file", "ratio"),
            ("silence_ratio", "Silence share of file", "ratio"),
            ("segment_count", "Speech segments", ""),
            ("leading_silence_s", "Leading silence", "s"),
            ("trailing_silence_s", "Trailing silence", "s"),
            ("longest_internal_gap_s", "Longest internal gap", "s"),
            ("sparse_speech", "Sparse speech (over 70% silence)", ""),
        ],
    )
    show_block(
        "truncation",
        results,
        [
            ("duration_s", "Duration", "s"),
            ("short_file", "Under 2 seconds", ""),
            ("speech_at_start", "Speech touches file start", ""),
            ("speech_at_end", "Speech touches file end", ""),
            ("word_count", "Words", ""),
            ("hallucination_risk", "Elevated hallucination risk", ""),
        ],
    )
    show_block(
        "tempo",
        results,
        [
            ("wpm", "Words per minute of speech", "wpm"),
            ("wpm_median", "Median utterance tempo", "wpm"),
            ("wpm_p90", "90th percentile tempo", "wpm"),
            ("fast_utterance_count", "Utterances above 200 wpm", ""),
        ],
    )
    show_block(
        "overlap",
        results,
        [
            ("overlap_seconds", "Overlapping speech", "s"),
            ("overlap_ratio", "Share of duration", "ratio"),
            ("high_overlap", "Above the 15% threshold", ""),
            ("unattributed_speech_s", "Speech no speaker was assigned", "s"),
        ],
    )
    show_block(
        "speakers",
        results,
        [
            ("speaker_count", "Speakers", ""),
            ("single_speaker", "Single speaker", ""),
            ("needs_diarisation", "Needs diarisation", ""),
        ],
    )
    show_block(
        "f0",
        results,
        [
            ("f0_median_hz", "Median pitch", "Hz"),
            ("f0_iqr_hz", "Interquartile range", "Hz"),
            ("f0_min_hz", "Minimum", "Hz"),
            ("f0_max_hz", "Maximum", "Hz"),
            ("voiced_fraction", "Voiced share of speech", "ratio"),
            ("high_pitch", "High pitch", ""),
        ],
    )
    show_block("codeswitch", results, [])


# ----------------------------------------------------------------- transcript

THRESHOLD_MODES = {
    "band": "Fixed confidence value",
    "percentile": "Target a highlight rate",
    "percentile_floor": "Target a rate, with a floor",
}

# Published optimal thresholds range 0.41-0.94 across transcripts
# (arXiv:2503.15124). This sits at the top of that range deliberately: the
# median word comes back at about 0.998, so a lower cutoff marks almost
# nothing and the panel looks reassuring when it should not.
DEFAULT_BAND = 0.9


def render_transcript(path: Path, results: dict) -> None:
    transcript = results.get("transcript")
    if not transcript:
        st.info("Tick Transcribe in the sidebar to generate a transcript.")
        return
    if "error" in transcript:
        st.error(f"Transcription failed: {transcript['error']}")
        return

    st.caption(
        "Every word carries a confidence score from the engine. Marking the "
        "low ones is how a transcription error gets found in the first place — "
        "then the Metrics tab says what in the audio caused it."
    )

    # Three ways to set the same cutoff. A fixed value is easy to reason about
    # but gives a different highlight rate on every file; the rate-targeting
    # modes fix the rate instead and let the value move.
    controls = st.columns([2, 2, 2])
    with controls[0]:
        mode = st.radio(
            "How to set the cutoff",
            list(THRESHOLD_MODES),
            format_func=THRESHOLD_MODES.get,
            index=0,
            help="Published optimal thresholds range 0.41-0.94 across "
            "transcripts, so one fixed value gives an unpredictable highlight "
            "rate file to file. Targeting a rate keeps it steady.",
        )
    value, pct, floor = DEFAULT_BAND, 4.0, 0.9
    with controls[1]:
        if mode == "band":
            value = st.slider(
                "Flag words below",
                0.5,
                0.99,
                DEFAULT_BAND,
                0.01,
                help="Most of this range is dead — nearly every word comes "
                "back above 0.99.",
            )
        else:
            pct = st.slider("Flag the lowest %", 1.0, 25.0, 4.0, 0.5)
    with controls[2]:
        if mode == "percentile_floor":
            floor = st.slider(
                "Never flag above",
                0.5,
                1.0,
                0.9,
                0.01,
                help="Stops a genuinely clean call having its least-confident "
                "words flagged as if they were problems.",
            )
        suppress_filler = st.checkbox(
            "Ignore filler words",
            value=True,
            help="Skips words whose misrecognition cannot change what the "
            "record says — the, a, is, um, there. Keeps yes/no, numbers, "
            "names and pronouns.",
        )

    threshold = stats.resolve_threshold(
        transcript["words"],
        mode,
        value=value,
        pct=pct,
        floor=floor,
        suppress_filler=suppress_filler,
    )

    # What the current setting actually does on this file. Without this the
    # controls are guesswork.
    rate = stats.flagged_rate(
        transcript["words"], threshold, suppress_filler=suppress_filler
    )
    unfiltered = stats.flagged_rate(transcript["words"], threshold, suppress_filler=False)
    # In floor mode the requested rate is unreachable whenever the file is noisy
    # enough that its Nth percentile sits above the floor. The floor then sets
    # the threshold and the rate slider stops doing anything — without this
    # notice that reads as a broken control rather than the floor working.
    capped = mode == "percentile_floor" and threshold < stats.percentile_threshold(
        transcript["words"], pct, suppress_filler=suppress_filler
    )
    st.caption(
        f"Marking **{rate:.1f}%** of words — everything below {threshold:.3f}."
        + (
            f"  \n:orange[The floor at {floor:.2f} is capping this; you asked "
            f"for {pct:.1f}%. Raise the floor or the rate slider will not move.]"
            if capped
            else ""
        )
        + (
            f"  \n{unfiltered:.1f}% before filler words were excluded."
            if suppress_filler and unfiltered > rate
            else ""
        )
        + ("  \n:red[Above about 13% the marks stop being scannable.]" if rate >= 13.0 else "")
    )

    with st.expander("Display"):
        bin_seconds = st.slider(
            "Timeline bin size (s)",
            2.0,
            30.0,
            10.0,
            1.0,
            help="Smaller bins show more detail but more noise.",
        )
        show_confidence = st.checkbox(
            "Show confidence under every word",
            value=False,
            help="Off by default — a number under each of a thousand words "
            "roughly doubles the visual density. Hover a word for the same "
            "value.",
        )

    summary = stats.summarise(transcript, threshold, suppress_filler=suppress_filler)
    boundary = stats.boundary_confidence(transcript, threshold=threshold)

    cols = st.columns(5)
    cols[0].metric("Words", f"{summary['word_count']:,}")
    cols[1].metric("Mean confidence", f"{summary['mean_confidence']:.3f}")
    cols[2].metric("Marked", f"{summary['pct_below_threshold']:.1f}%")
    cols[3].metric("Speakers", summary["speaker_count"])
    cols[4].metric(
        "Turns",
        f"{summary['turn_count']:,}",
        help="Turns under a second are counted below. A run of them is the "
        "usual signature of diarisation breaking down.",
    )
    if summary["turn_count"]:
        st.caption(
            f"{summary['short_turns']} turns under 1s "
            f"({summary['pct_short_turns']:.1f}%)"
        )

    size = path.stat().st_size
    audio_src = render.audio_data_uri(path) if size <= MAX_INLINE_BYTES else None
    if audio_src is None:
        st.warning(
            f"{path.name} is {size / 1e6:.0f} MB — too large to inline, so "
            "click-to-seek is off. Use the player below."
        )
        st.audio(str(path))

    # One threshold and one eligibility rule drive the word colours, the
    # timeline and the rug, so moving a control moves all three together.
    timeline = stats.confidence_timeline(
        transcript, bin_seconds, threshold, suppress_filler=suppress_filler
    )
    html = render.build_html(
        transcript,
        audio_src=audio_src,
        threshold=threshold,
        timeline=timeline,
        boundaries=[b["time"] for b in boundary.get("boundaries", [])],
        show_confidence=show_confidence,
        suppress_filler=suppress_filler,
    )
    st.iframe(html, height=860)

    left, right = st.columns(2)

    with left:
        st.markdown("**Lowest-confidence words**")
        st.caption(
            "Where to start listening. Filler words are excluded when the "
            "toggle above is on, or this list is mostly `and`."
        )
        worst = stats.lowest_confidence_words(
            transcript, limit=40, suppress_filler=suppress_filler
        )
        st.dataframe(
            [
                {
                    "Word": w["text"],
                    "Confidence": round(w["confidence"], 3),
                    "Start (s)": round(w["start"], 2),
                    "Speaker": w["speaker"],
                }
                for w in worst
            ],
            hide_index=True,
            width="stretch",
            height=320,
        )

    with right:
        st.markdown("**Confidence by speaker**")
        st.caption(
            "A wide gap means one speaker is being captured much worse than "
            "the other, which is a microphone or a routing problem rather "
            "than a transcription one."
        )
        for speaker, mean in summary["speaker_means"].items():
            st.write(f"**{speaker}** — {mean:.3f}")
        if summary["speaker_gap"]:
            st.caption(f"Gap between best and worst: {summary['speaker_gap']:.3f}")

        if boundary.get("count"):
            st.markdown("**At speaker changes**")
            st.caption(
                "Word confidence measures transcription, not diarisation. A "
                "deficit here is the closest available proxy for an uncertain "
                "speaker boundary, not proof of one."
            )
            st.write(
                f"**{boundary['boundary_mean']:.3f}** at boundaries against "
                f"**{boundary['other_mean']:.3f}** elsewhere "
                f"({boundary['deficit']:+.3f})"
            )
            st.caption(f"{boundary['count']} speaker changes")

        st.markdown("**Confidence distribution**")
        st.caption(
            "Almost everything piles up at the right-hand end. That is normal "
            "and is why the cutoff sits so high."
        )
        hist = stats.histogram(transcript)
        st.bar_chart(
            {"words": [c for _, _, c in hist]},
            x_label="confidence bin (0-1)",
            y_label="words",
        )


# -------------------------------------------------------------------- routing

STATUS_WORD = {
    routing.FIRED: (interpret.PROBLEM, "fired"),
    routing.CLEAR: (interpret.GOOD, "clear"),
    routing.UNKNOWN: (interpret.UNKNOWN, "not evaluated"),
}


def rule_row(verdict: routing.Verdict) -> None:
    """One rule in the list, with a button that focuses the panel on it."""
    status, word = STATUS_WORD[verdict.status]
    with st.container(border=True):
        head, button = st.columns([5, 1])
        with head:
            st.markdown(
                f"{chip(status, word)} &nbsp; **{verdict.rule}**",
                unsafe_allow_html=True,
            )
            if verdict.status == routing.UNKNOWN:
                st.caption(verdict.detail)
            else:
                st.caption(
                    f"`{verdict.signal}` = **{verdict.observed}** · "
                    f"rule is {verdict.threshold}"
                )
        with button:
            if st.button("Explain", key=f"explain_{verdict.rule}"):
                st.session_state["routing_focus"] = verdict.rule


def render_detail(verdict: routing.Verdict) -> None:
    """The right-hand panel: what this rule is, and what to do about it."""
    status, word = STATUS_WORD[verdict.status]
    st.markdown(f"{chip(status, word)}", unsafe_allow_html=True)
    st.subheader(verdict.rule)

    metric = interpret.metric_for_rule(verdict.rule)

    if verdict.status == routing.UNKNOWN:
        st.warning(
            "This rule could not be evaluated. " + verdict.detail + " Until "
            "then this is not a pass — it is a blank."
        )
    else:
        st.markdown(
            f"**Measured:** {verdict.observed}"
            + (f" {metric.unit}" if metric and metric.unit else "")
        )
        st.markdown(f"**Rule fires when:** {verdict.threshold}")
        if metric is not None:
            _, label = interpret.classify(metric, verdict.observed)
            st.markdown(
                band_bar(metric, verdict.observed) + chip(status, label),
                unsafe_allow_html=True,
            )

    if metric is not None:
        st.markdown("**What this is**")
        st.write(metric.what)
        st.markdown("**Why a model cares**")
        st.write(metric.why)

    st.markdown("**What the router would do**")
    st.write(verdict.action)
    if verdict.status == routing.CLEAR:
        st.caption("Not on this file — shown so you can see what the rule is for.")

    if verdict.detail and verdict.status != routing.UNKNOWN:
        st.markdown("**How much to trust this**")
        st.write(verdict.detail)

    if metric is not None:
        st.caption(f"Band from: {metric.source}. Signal: `{verdict.signal}`.")


def render_routing(results: dict) -> None:
    verdicts = routing.evaluate(results)
    fired = [v for v in verdicts if v.status == routing.FIRED]
    unknown = [v for v in verdicts if v.status == routing.UNKNOWN]
    clear = [v for v in verdicts if v.status == routing.CLEAR]

    st.info(
        "This is a simulation. The rules from `notes.md` are evaluated against "
        "what was measured and the recommended route is shown, but nothing is "
        "sent anywhere and no model is chosen. Press **Explain** on any rule "
        "to see what it means."
    )

    left, right = st.columns([2, 1], gap="medium")

    with left:
        st.subheader(f"{len(fired)} of {len(verdicts)} rules fired")
        if not fired:
            st.success("Nothing flagged among the rules that could be evaluated.")
        for verdict in fired:
            rule_row(verdict)

        if unknown:
            st.subheader(f"{len(unknown)} could not be evaluated")
            st.caption(
                "These are not passes. The analysis they need was not run, or "
                "ran and could not produce a usable number."
            )
            for verdict in unknown:
                rule_row(verdict)

        if clear:
            st.subheader(f"{len(clear)} clear")
            for verdict in clear:
                rule_row(verdict)

    with right:
        # Default to whatever most deserves attention, so the panel is never
        # empty and never opens on a rule that passed while others fired.
        order = fired + unknown + clear
        by_rule = {v.rule: v for v in verdicts}
        focus = st.session_state.get("routing_focus")
        verdict = by_rule.get(focus) or (order[0] if order else None)
        if verdict is not None:
            with st.container(border=True):
                render_detail(verdict)


if __name__ == "__main__":
    main()
