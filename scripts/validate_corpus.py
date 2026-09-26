"""Measure the SNR and overlap estimators against constructed ground truth.

The synthetic corpus this scores against was built by adding a known amount of
noise, and by starting a known share of turns early into the previous one. That
makes it the only place where these two estimators can be scored rather than
eyeballed. Point ``AUDIO_CORPUS_SYNTHETIC`` at the directory holding it.

Ground truth:

- **SNR.** The corpus builder adds band-limited pink noise at a target
  ratio against the RMS of the speech stretches only, so ``snr_5.wav`` really
  is 5 dB speech-to-noise. Every variant also carries a 30 dB line-noise floor,
  which is what ``base.wav`` is being scored against.
- **Overlap.** Each manifest records where every turn was actually placed, so
  the true overlapping time is a sweep over those turns — not an estimate.

Only the mono downmixes are used. The stereo originals put one speaker per
channel, which makes diarisation trivial and the overlap question meaningless.

Runs against cached transcripts only, so it never calls the API. Files without
one are skipped and counted.

    uv run python scripts/validate_corpus.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_dashboard import asr, audio_io, cache, registry  # noqa: E402

CORPUS = Path(os.environ.get("AUDIO_CORPUS_SYNTHETIC", "")).expanduser()

# The line-noise floor every variant gets, set when the corpus was built.
FLOOR_SNR_DB = 30.0
EXPECTED_SNR = {"base": FLOOR_SNR_DB, "snr_10": 10.0, "snr_5": 5.0}
OVERLAP_VARIANTS = ("base", "overlap_10", "overlap_25")


def true_overlap_seconds(turns: list[dict]) -> float:
    """Time with two or more turns live at once, from where they were placed.

    Same sweep the estimator uses, over the manifest rather than over a
    diarised transcript — so a difference is the diariser's, not the method's.
    """
    events: list[tuple[float, int]] = []
    for turn in turns:
        events.append((float(turn["start"]), 1))
        events.append((float(turn["end"]), -1))
    events.sort()

    live = 0
    previous = 0.0
    total = 0.0
    for time, delta in events:
        if live >= 2 and time > previous:
            total += time - previous
        live += delta
        previous = time
    return total


def analyse(path: Path, picks: list[str]) -> dict:
    bundle = audio_io.load(path)
    return registry.run(bundle, cache.file_digest(path), picks)


def manifests() -> list[tuple[str, dict]]:
    found = []
    for manifest in sorted(CORPUS.glob("*/manifest.json")):
        found.append((manifest.parent.name, json.loads(manifest.read_text())))
    return found


def report(title: str, rows: list[dict], columns: list[tuple[str, str]]) -> None:
    print(f"\n{title}")
    widths = {key: max(len(head), 12) for key, head in columns}
    print("  " + "  ".join(f"{head:>{widths[key]}}" for key, head in columns))
    for row in rows:
        print(
            "  "
            + "  ".join(
                f"{row.get(key, ''):>{widths[key]}}" if isinstance(row.get(key), str)
                else f"{row.get(key, float('nan')):>{widths[key]}.2f}"
                for key, _ in columns
            )
        )


def validate_snr() -> None:
    rows: list[dict] = []
    errors: dict[str, list[float]] = {}
    skipped = 0

    for script, _ in manifests():
        for variant, expected in EXPECTED_SNR.items():
            path = CORPUS / script / "mono" / f"{variant}.wav"
            if not path.exists():
                skipped += 1
                continue
            result = analyse(path, ["vad", "snr"])["snr"]
            if not result.get("usable"):
                rows.append(
                    {
                        "file": f"{script}/{variant}",
                        "expected": expected,
                        "measured": float("nan"),
                        "error": float("nan"),
                    }
                )
                continue
            measured = result["snr_db"]
            rows.append(
                {
                    "file": f"{script}/{variant}",
                    "expected": expected,
                    "measured": measured,
                    "error": measured - expected,
                }
            )
            errors.setdefault(variant, []).append(measured - expected)

    report(
        "SNR — measured against the ratio the noise was added at",
        rows,
        [("file", "file"), ("expected", "true dB"), ("measured", "measured"), ("error", "error")],
    )
    print(f"\n  {skipped} file(s) missing.")
    for variant, values in sorted(errors.items()):
        print(
            f"  {variant:10} n={len(values):2}  "
            f"mean error {statistics.fmean(values):+.2f} dB  "
            f"spread {(max(values) - min(values)):.2f} dB"
        )


def validate_overlap() -> None:
    rows: list[dict] = []
    skipped = 0

    for script, manifest in manifests():
        for variant in OVERLAP_VARIANTS:
            if variant not in manifest:
                continue
            path = CORPUS / script / "mono" / f"{variant}.wav"
            if not path.exists() or not asr.is_cached(path, asr.TranscribeConfig()):
                skipped += 1
                continue

            truth_s = true_overlap_seconds(manifest[variant]["turns"])
            duration = float(manifest[variant]["duration_s"])
            results = analyse(path, ["transcript", "vad", "overlap", "speakers"])
            overlap = results["overlap"]
            speakers = results["speakers"]

            rows.append(
                {
                    "file": f"{script}/{variant}",
                    "true_s": truth_s,
                    "true_pct": 100.0 * truth_s / duration,
                    "meas_s": float(overlap.get("overlap_seconds", float("nan"))),
                    "meas_pct": 100.0 * float(overlap.get("overlap_ratio", 0.0)),
                    "unattr_pct": 100.0
                    * float(overlap.get("unattributed_speech_ratio", 0.0)),
                    "speakers": float(speakers.get("speaker_count", 0)),
                }
            )

    report(
        "Overlap — measured against where the turns were actually placed",
        rows,
        [
            ("file", "file"),
            ("true_s", "true s"),
            ("true_pct", "true %"),
            ("meas_s", "measured s"),
            ("meas_pct", "measured %"),
            ("unattr_pct", "unattrib %"),
            ("speakers", "speakers"),
        ],
    )
    print(f"\n  {skipped} file(s) without a cached transcript, skipped.")

    with_overlap = [r for r in rows if r["true_s"] > 0]
    without = [r for r in rows if r["true_s"] == 0]
    if with_overlap:
        recovered = [
            100.0 * r["meas_s"] / r["true_s"] for r in with_overlap if r["true_s"]
        ]
        print(
            f"  files with real overlap: n={len(with_overlap)}  "
            f"median share recovered {statistics.median(recovered):.1f}%  "
            f"unattributed speech {statistics.median(r['unattr_pct'] for r in with_overlap):.2f}%"
        )
    if without:
        print(
            f"  files with none:        n={len(without)}  "
            f"median reported {statistics.median(r['meas_pct'] for r in without):.2f}%  "
            f"unattributed speech {statistics.median(r['unattr_pct'] for r in without):.2f}%"
        )


def main() -> int:
    if not CORPUS.is_dir():
        print(f"No corpus at {CORPUS}")
        return 1
    validate_overlap()
    validate_snr()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
