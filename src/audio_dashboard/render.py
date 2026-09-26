"""Render a transcript to HTML with confidence-coloured words.

Colour mapping matters here. AssemblyAI word confidence is heavily skewed
towards 1.0, so a linear 0-1 map paints nearly every word the same and the
visualisation shows nothing. Two modes are offered instead:

- ``band``: fixed thresholds, adjustable in the UI.
- ``percentile``: colour the lowest N% of words *in this file*, which is robust
  to per-file shifts in the distribution.
"""

from __future__ import annotations

import base64
import html
import mimetypes
from pathlib import Path
from typing import Any

from . import salience

# Distinct hues per speaker, used only as a left border + faint tint so they
# never compete with the confidence colouring of the text itself.
SPEAKER_COLOURS = [
    "#2563eb",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#db2777",
    "#0891b2",
]

LOW_COLOUR = "#dc2626"
NORMAL_COLOUR = "#111827"

# SVG viewBox geometry for the timeline. Width is arbitrary — the SVG scales to
# the component width — but the aspect ratio and margins are fixed here.
TL_W = 1000.0
TL_H = 150.0
TL_PAD = {"top": 12.0, "right": 10.0, "bottom": 34.0, "left": 34.0}
TL_RUG_H = 14.0  # rug strip sits between the plot floor and the time axis


def _word_class(
    word: dict[str, Any],
    threshold: float,
    *,
    suppress_filler: bool = True,
) -> str:
    """Flagged or not — deliberately one band, not two.

    A second (amber) band makes a viewer ask what amber means, which is exactly
    the confusion this view exists to remove. That is also why suppressed filler
    is drawn as ordinary text rather than given a style of its own.

    A class rather than an inline style so the hover and currently-playing
    backgrounds compose with the mark rather than fighting it — an inline style
    would beat every selector.
    """
    return (
        "w low"
        if salience.is_flagged(word, threshold, suppress_filler=suppress_filler)
        else "w"
    )


def _nice_ceil(value: float) -> float:
    """Round up to a readable axis maximum.

    Floored at 5% so a quiet file with one bad patch still gets a sensible axis
    rather than a maximum of 0.4% that magnifies noise into drama.
    """
    for step in (5, 10, 15, 20, 25, 30, 40, 50, 75, 100):
        if value <= step:
            return float(step)
    return 100.0


def build_timeline_svg(
    timeline: dict[str, Any],
    speaker_colour: dict[str, str],
    boundaries: list[float] | None = None,
) -> str:
    """SVG of "% words below threshold" over time, one polyline per speaker.

    Y is auto-scaled: the signal is typically single-digit percent, so a fixed
    0-100 axis would squash it flat.
    """
    duration = timeline.get("duration", 0.0)
    if not timeline.get("bins") or duration <= 0:
        return '<div class="empty">No timeline data.</div>'

    x0 = TL_PAD["left"]
    x1 = TL_W - TL_PAD["right"]
    y0 = TL_PAD["top"]
    y1 = TL_H - TL_PAD["bottom"] - TL_RUG_H
    plot_w = x1 - x0
    plot_h = y1 - y0

    values = [v for series in timeline["series"].values() for v in series if v is not None]
    y_max = _nice_ceil(max(values) if values else 0.0)

    def sx(t: float) -> float:
        return x0 + plot_w * (t / duration)

    def sy(pct: float) -> float:
        return y1 - plot_h * (pct / y_max)

    parts: list[str] = []

    # Horizontal gridlines with axis labels.
    for frac in (0.0, 0.5, 1.0):
        pct = y_max * frac
        y = sy(pct)
        parts.append(
            f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x1:.1f}" y2="{y:.1f}" '
            f'class="grid"/>'
            f'<text x="{x0 - 5:.1f}" y="{y + 3:.1f}" class="ylab">{pct:.0f}%</text>'
        )

    # Time axis: about six ticks, whatever the duration.
    n_ticks = 6
    for i in range(n_ticks + 1):
        t = duration * i / n_ticks
        parts.append(
            f'<text x="{sx(t):.1f}" y="{TL_H - 6:.1f}" class="xlab">'
            f"{_fmt_time(t)}</text>"
        )

    # Rug: one tick per low-confidence word. Independent of bin size, so bad
    # patches stay visible however coarse the bins get.
    rug_top = y1 + 4
    rug_bottom = rug_top + TL_RUG_H - 4
    for t in timeline.get("low_words", []):
        parts.append(
            f'<line x1="{sx(t):.2f}" y1="{rug_top:.1f}" '
            f'x2="{sx(t):.2f}" y2="{rug_bottom:.1f}" class="rug"/>'
        )

    # Speaker changes, as ticks on the time axis — lets a bad boundary and a bad
    # patch of audio be read together.
    for t in boundaries or []:
        parts.append(
            f'<line x1="{sx(t):.2f}" y1="{rug_bottom + 1:.1f}" '
            f'x2="{sx(t):.2f}" y2="{rug_bottom + 6:.1f}" class="bnd"/>'
        )

    # One polyline per speaker, split into runs so that bins where the speaker
    # said nothing become gaps rather than dropping the line to zero.
    for speaker, series in timeline["series"].items():
        colour = speaker_colour.get(speaker, "#666")
        run: list[str] = []
        for i, value in enumerate(series):
            if value is None:
                if len(run) > 1:
                    parts.append(
                        f'<polyline points="{" ".join(run)}" class="line" '
                        f'stroke="{colour}"/>'
                    )
                run = []
                continue
            centre = timeline["bins"][i]["start"] + timeline["bin_seconds"] / 2
            run.append(f"{sx(min(centre, duration)):.2f},{sy(value):.2f}")
        if len(run) > 1:
            parts.append(
                f'<polyline points="{" ".join(run)}" class="line" stroke="{colour}"/>'
            )
        elif len(run) == 1:
            # A speaker with a single populated bin would otherwise be invisible.
            cx, cy = run[0].split(",")
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="2.5" fill="{colour}"/>')

    legend = " ".join(
        f'<span class="key"><i style="background:{speaker_colour.get(s, "#666")}"></i>'
        f"{html.escape(s)}</span>"
        for s in timeline["series"]
    )

    # The sample carries the real .w.low class rather than a copy of its styling,
    # so the key cannot drift from the transcript it is explaining.
    mark_key = '<span class="key"><span class="w low">word</span> low confidence</span>'

    return (
        f'<div class="tl-head"><span class="tl-title">% words below '
        f'{max(0.0, timeline.get("threshold", 0)):.2f}</span>{mark_key}{legend}</div>'
        f'<svg id="tl" viewBox="0 0 {TL_W:.0f} {TL_H:.0f}" preserveAspectRatio="none" '
        f'data-x0="{x0}" data-x1="{x1}" data-dur="{duration}">'
        f"{''.join(parts)}"
        f'<line id="playhead" x1="{x0}" y1="{y0}" x2="{x0}" y2="{rug_bottom:.1f}" '
        f'style="display:none"/>'
        f'<rect id="tl-hit" x="{x0}" y="{y0}" width="{plot_w}" '
        f'height="{rug_bottom - y0:.1f}" fill="transparent"/>'
        f"</svg>"
    )


# mimetypes guesses containers in ways browsers won't play: .opus is unknown to
# older Python, and .m4a comes back as the RFC 3016 streaming type rather than
# the one <audio> expects.
_AUDIO_MIME = {
    ".opus": "audio/ogg",
    ".oga": "audio/ogg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
}


def audio_data_uri(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = _AUDIO_MIME.get(suffix) or mimetypes.guess_type(path.name)[0] or "audio/mpeg"
    payload = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{payload}"


def build_html(
    transcript: dict[str, Any],
    *,
    audio_src: str | None,
    threshold: float = 0.6,
    timeline: dict[str, Any] | None = None,
    boundaries: list[float] | None = None,
    show_confidence: bool = False,
    suppress_filler: bool = True,
) -> str:
    """Build a self-contained HTML document for the transcript.

    ``audio_src`` may be a data URI, or None to render without a player (used
    for files too large to inline, where Streamlit's own player is shown
    instead and click-to-seek is unavailable).
    """
    speaker_colour: dict[str, str] = {}
    blocks: list[str] = []

    # No `title` attribute: the native tooltip is slow and unstyled, and it would
    # compete with the custom one. Confidence rides on data-conf, read both by
    # the hover handler and by the always-on `.c` span.
    def word_span(word: dict[str, Any]) -> str:
        cls = _word_class(word, threshold, suppress_filler=suppress_filler)
        return (
            f'<span class="{cls}" data-start="{word["start"]:.3f}" '
            f'data-conf="{word["confidence"]:.3f}">'
            f'<span class="t">{html.escape(word["text"])}</span>'
            f'<span class="c">{word["confidence"]:.2f}</span></span>'
        )

    for utt in transcript["utterances"]:
        speaker = utt["speaker"]
        if speaker not in speaker_colour:
            speaker_colour[speaker] = SPEAKER_COLOURS[
                len(speaker_colour) % len(SPEAKER_COLOURS)
            ]
        colour = speaker_colour[speaker]

        words_html = " ".join(word_span(w) for w in utt["words"])
        blocks.append(
            f'<div class="turn" style="border-left-color:{colour}">'
            f'<div class="meta"><span class="spk" style="color:{colour}">'
            f"{html.escape(speaker)}</span>"
            f'<span class="ts">{_fmt_time(utt["start"])}</span></div>'
            f'<div class="text">{words_html}</div></div>'
        )

    player = (
        f'<audio id="player" controls preload="metadata" src="{audio_src}"></audio>'
        if audio_src
        else '<div class="nofile">Audio too large to inline &mdash; '
        "click-to-seek disabled. Use the player above.</div>"
    )

    timeline_html = (
        build_timeline_svg(timeline, speaker_colour, boundaries or [])
        if timeline
        else ""
    )
    body_class = "show-conf" if show_confidence else ""

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 0; padding: 0 4px; background: #fff; color: #111827; }}
  #bar {{ position: sticky; top: 0; background: #fff; padding: 8px 0;
          border-bottom: 1px solid #e5e7eb; z-index: 10; }}
  audio {{ width: 100%; }}
  .nofile {{ font-size: 13px; color: #6b7280; padding: 6px 0; }}
  .empty {{ font-size: 13px; color: #9ca3af; padding: 8px 0; }}

  /* Timeline */
  #tl {{ width: 100%; height: 150px; display: block; }}
  .tl-head {{ font-size: 11px; color: #6b7280; padding: 4px 0 0 2px; }}
  .tl-title {{ font-weight: 600; margin-right: 10px; }}
  .key {{ margin-right: 10px; }}
  .key i {{ display: inline-block; width: 9px; height: 9px; border-radius: 2px;
            margin-right: 3px; vertical-align: -1px; }}
  /* The sample word is a real .w.low, so switch off the behaviours that only
     make sense inside the transcript. */
  .key .w.low {{ display: inline; padding: 0; cursor: default; color: inherit;
                 /* The wave is sized for 15px transcript text; at 11px the same
                    stroke reads as a scribble rather than a sample of it. */
                 text-decoration-thickness: 1px; text-underline-offset: 2px; }}
  .key .w.low:hover {{ background: none; }}
  .grid {{ stroke: #e5e7eb; stroke-width: 1; vector-effect: non-scaling-stroke; }}
  .line {{ fill: none; stroke-width: 1.6; vector-effect: non-scaling-stroke;
           stroke-linejoin: round; }}
  .rug {{ stroke: {LOW_COLOUR}; stroke-width: 1; opacity: .35;
          vector-effect: non-scaling-stroke; }}
  .bnd {{ stroke: #6b7280; stroke-width: 1; opacity: .5;
          vector-effect: non-scaling-stroke; }}
  .ylab {{ font-size: 9px; fill: #9ca3af; text-anchor: end; }}
  .xlab {{ font-size: 9px; fill: #9ca3af; text-anchor: middle; }}
  #playhead {{ stroke: #111827; stroke-width: 1;
               vector-effect: non-scaling-stroke; }}
  #tl-hit {{ cursor: crosshair; }}

  .turn {{ border-left: 3px solid #ccc; padding: 6px 0 6px 10px; margin: 10px 0; }}
  .meta {{ font-size: 12px; margin-bottom: 2px; }}
  .spk {{ font-weight: 600; }}
  .ts {{ color: #9ca3af; margin-left: 8px; font-variant-numeric: tabular-nums; }}
  .text {{ line-height: 2.1; font-size: 15px; }}

  /* A wavy underline rather than a filled tint. The signal is the same and
     already universally read as "this word is suspect", but it costs a stroke
     instead of a block: at the ~13% flag rate the readout warns about, tinted
     words merge into a wall, where underlines stay countable.

     Marking with text-decoration also leaves background free for the two
     transient states, so :hover and .playing no longer have to override the
     mark — all three can show at once and the word stays flagged while you
     hover or play it.

     skip-ink: none keeps the line continuous under descenders. Left on, a "j"
     or "g" punches a gap through it that reads as two separate marks. */
  .w {{ cursor: pointer; padding: 1px 3px; border-radius: 4px;
        color: {NORMAL_COLOUR}; }}
  .w.low {{ text-decoration: underline wavy {LOW_COLOUR};
            text-decoration-thickness: 2px;
            text-underline-offset: 3px;
            text-decoration-skip-ink: none; }}
  .w:hover {{ background: #fde68a; }}
  .w.playing {{ background: #bfdbfe; }}

  /* Always-on numbers. Toggling flips this one class rather than changing the
     markup, so the hover and always-on paths cannot drift apart. */
  .c {{ display: none; }}
  .show-conf .text {{ line-height: 2.6; }}
  .show-conf .w {{ display: inline-flex; flex-direction: column;
                   align-items: center; vertical-align: bottom; }}
  .show-conf .c {{ display: block; font-size: 9px; color: #9ca3af;
                   font-variant-numeric: tabular-nums; line-height: 1.1; }}

  #tip {{ position: fixed; display: none; z-index: 50; pointer-events: none;
          background: #111827; color: #fff; font-size: 11px; padding: 3px 6px;
          border-radius: 3px; font-variant-numeric: tabular-nums; }}
</style></head><body class="{body_class}">
<div id="bar">{player}{timeline_html}</div>
<div id="transcript">{"".join(blocks)}</div>
<div id="tip"></div>
<script>
  const player = document.getElementById("player");
  // Scoped to the transcript: the legend sample is a real .w.low too, and it has
  // no timestamp to seek to.
  const words = Array.from(document.querySelectorAll("#transcript .w"));
  const tip = document.getElementById("tip");
  const tl = document.getElementById("tl");
  const playhead = document.getElementById("playhead");

  // One shared tooltip driven by delegation, rather than a pseudo-element per
  // span — there are well over a thousand words in a typical call.
  const transcript = document.getElementById("transcript");
  transcript.addEventListener("mouseover", e => {{
    const w = e.target.closest(".w");
    if (!w) return;
    tip.textContent = "conf " + w.dataset.conf + "  ·  " +
      (+w.dataset.start).toFixed(2) + "s";
    tip.style.display = "block";
  }});
  transcript.addEventListener("mousemove", e => {{
    if (tip.style.display !== "block") return;
    tip.style.left = Math.min(e.clientX + 12, innerWidth - 120) + "px";
    tip.style.top = (e.clientY + 16) + "px";
  }});
  transcript.addEventListener("mouseout", e => {{
    if (e.target.closest(".w")) tip.style.display = "none";
  }});

  if (player) {{
    words.forEach(w => w.addEventListener("click", () => {{
      player.currentTime = parseFloat(w.dataset.start);
      player.play();
    }}));

    // Click the timeline to seek. The SVG scales to the component width, so map
    // the click back through the rendered box rather than viewBox units.
    const hit = document.getElementById("tl-hit");
    if (hit && tl) {{
      const x0 = +tl.dataset.x0, x1 = +tl.dataset.x1, dur = +tl.dataset.dur;
      hit.addEventListener("click", e => {{
        const box = tl.getBoundingClientRect();
        const vx = (e.clientX - box.left) / box.width * {TL_W:.0f};
        player.currentTime = Math.max(0, Math.min(dur,
          (vx - x0) / (x1 - x0) * dur));
        player.play();
      }});
    }}

    // Single timer drives both the spoken-word highlight and the playhead.
    // Words are in document order, so the linear scan is cheap enough at 4Hz.
    let last = 0;
    setInterval(() => {{
      if (player.paused) return;
      const t = player.currentTime;
      let idx = words.findIndex(w => parseFloat(w.dataset.start) > t) - 1;
      if (idx < 0) idx = words.length - 1;
      if (idx !== last) {{
        words[last]?.classList.remove("playing");
        words[idx]?.classList.add("playing");
        last = idx;
      }}
      if (playhead && tl) {{
        const x0 = +tl.dataset.x0, x1 = +tl.dataset.x1, dur = +tl.dataset.dur;
        const x = x0 + (x1 - x0) * Math.min(1, t / dur);
        playhead.setAttribute("x1", x);
        playhead.setAttribute("x2", x);
        playhead.style.display = "block";
      }}
    }}, 250);
  }}
</script>
</body></html>"""


def _fmt_time(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:d}:{secs:02d}"
