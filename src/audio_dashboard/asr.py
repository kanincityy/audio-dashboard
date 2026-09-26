"""Transcription with an on-disk cache, over AssemblyAI or ElevenLabs.

A pipeline that only needs a readable transcript keeps ``utterances[].speaker``
and ``utterances[].text`` and throws the rest away. This keeps per-word
confidence and timestamps instead, because the per-word numbers are what the
diagnostics here are built on.

Both providers are flattened into the same shape, so nothing downstream can
tell them apart. Where that flattening had to invent something, it says so.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import assemblyai as aai

CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "transcripts"

# Both models are pinned rather than left to the SDK default. The model id is
# part of the cache key, so a default that moves under us would silently mix
# transcripts from two models in one corpus.
TRANSCRIPTION_MODEL = "universal-3-5-pro"
ELEVENLABS_MODEL = "scribe_v2"

# The SDK default is 240 s, and the corpus clips run to six minutes with the
# real calls longer still. A timeout mid-call is the expensive kind of failure:
# the audio is billed but no cache entry is written, so the retry pays twice.
_ELEVENLABS_TIMEOUT_S = 1800.0

# Cache-key composition. See TranscribeConfig.cache_key.
_ORIGINAL_FIELDS = (
    "speaker_labels",
    "language_code",
    "speech_model",
    "speakers_expected",
)
_LATER_FIELDS = ("multichannel", "provider", "model_id")


@dataclass(frozen=True)
class TranscribeConfig:
    """Transcription options.

    Kept as a frozen dataclass rather than passing ``aai.TranscriptionConfig``
    around so it can be hashed into the cache key.

    The AssemblyAI fields come first because they came first: their names,
    order and defaults are what the existing cache was keyed on and must not
    move. ``provider`` and everything after it are later additions.
    """

    speaker_labels: bool = True
    # ElevenLabs has no separate diarisation flag either — this is sent as
    # `diarize`, which means the same thing.
    language_code: str | None = "en_uk"
    speech_model: str = TRANSCRIPTION_MODEL
    speakers_expected: int | None = None
    # Off by default. Transcribes each channel separately and labels
    # utterances by channel instead of clustering — only meaningful on audio
    # with one speaker per channel. Billed per channel.
    multichannel: bool = False
    # "assemblyai" or "elevenlabs".
    provider: str = "assemblyai"
    # The ElevenLabs model. Unused by AssemblyAI, which has `speech_model`.
    model_id: str = ""

    def to_aai(self) -> aai.TranscriptionConfig:
        """Only called when ``provider == "assemblyai"``."""
        return aai.TranscriptionConfig(
            speaker_labels=self.speaker_labels,
            language_code=self.language_code,
            speech_models=[self.speech_model],
            speakers_expected=self.speakers_expected,
            multichannel=self.multichannel or None,
        )

    def cache_key(self) -> str:
        key = {f: getattr(self, f) for f in _ORIGINAL_FIELDS}
        # Options added after the cache was populated only enter the key when
        # set, so adding one does not invalidate every entry written before it
        # existed. Changing this list re-transcribes the whole corpus.
        for field in _LATER_FIELDS:
            value = getattr(self, field)
            if value != self.__dataclass_fields__[field].default:
                key[field] = value
        blob = json.dumps(key, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:8]


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()[:16]


def cache_path(path: Path, config: TranscribeConfig) -> Path:
    return CACHE_DIR / f"{_file_digest(path)}-{config.cache_key()}.json"


def is_cached(path: Path, config: TranscribeConfig) -> bool:
    return cache_path(path, config).exists()


def transcribe(
    path: Path,
    config: TranscribeConfig | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Transcribe a local audio file, caching the parsed response.

    Returns a dict with ``utterances`` (each holding its own ``words``),
    ``words`` (flattened), and file-level metadata.
    """
    config = config or TranscribeConfig()
    dest = cache_path(path, config)

    if dest.exists() and not force:
        return json.loads(dest.read_text())

    # The cache invariant lives here and only here: the provider functions call
    # their API and flatten the response, and this decides what gets written.
    if config.provider == "assemblyai":
        parsed, latency_s = _transcribe_assemblyai(path, config)
    elif config.provider == "elevenlabs":
        parsed, latency_s = _transcribe_elevenlabs(path, config)
    else:
        raise ValueError(f"unknown provider: {config.provider!r}")

    parsed["latency_s"] = latency_s
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(parsed, indent=2))
    return parsed


def _transcribe_assemblyai(
    path: Path, config: TranscribeConfig
) -> tuple[dict[str, Any], float]:
    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ASSEMBLYAI_API_KEY is not set. Put it in .env at the repo root."
        )
    aai.settings.api_key = api_key

    # Timed around the live call only, then stored, so a cache hit reports the
    # latency that was actually measured rather than the time to read a file.
    started = time.monotonic()
    result = aai.Transcriber(config=config.to_aai()).transcribe(str(path))
    latency_s = time.monotonic() - started
    if result.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"AssemblyAI transcription error: {result.error or ''}")

    return _parse(result, path, config), latency_s


def _transcribe_elevenlabs(
    path: Path, config: TranscribeConfig
) -> tuple[dict[str, Any], float]:
    """Transcribe with ElevenLabs Scribe.

    The files here are local, so they are uploaded rather than handed over as a
    URL. No ``language_code`` is sent: Scribe then detects the language itself,
    and what it detected is recorded in the parsed result. Pinning a language
    would hide a file that came back as something other than English.
    """
    # Imported inside the function so the AssemblyAI path never pays to load
    # the ElevenLabs SDK.
    from elevenlabs.client import ElevenLabs

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ELEVENLABS_API_KEY is not set. Put it in .env at the repo root."
        )
    client = ElevenLabs(api_key=api_key, timeout=_ELEVENLABS_TIMEOUT_S)

    started = time.monotonic()
    with path.open("rb") as fh:
        result = client.speech_to_text.convert(
            model_id=config.model_id or ELEVENLABS_MODEL,
            file=fh,
            diarize=config.speaker_labels,
            timestamps_granularity="word",
            language_code=config.language_code,
        )
    latency_s = time.monotonic() - started

    # convert() returns a union, and the multichannel member carries
    # `transcripts` instead of `words`. Failing loudly beats scoring an empty
    # transcript that looks like a diarisation failure.
    if getattr(result, "words", None) is None:
        raise RuntimeError(
            f"ElevenLabs returned no words for {path.name}: got {type(result).__name__}"
        )

    return _parse_elevenlabs(result, path, config), latency_s


def _parse(
    result: aai.Transcript, path: Path, config: TranscribeConfig
) -> dict[str, Any]:
    """Flatten the SDK response, keeping word confidence and timestamps.

    Speakers are remapped to ``Speaker 1..N`` in first-appearance order, so a
    label means the same thing across files and providers. Under
    ``multichannel`` there may be no speaker at all, in which case the channel
    is the label — that is the whole point of the mode.
    """
    speaker_map: dict[str, str] = {}
    utterances: list[dict[str, Any]] = []
    flat_words: list[dict[str, Any]] = []

    for utt in result.utterances or []:
        channel = getattr(utt, "channel", None)
        # A None speaker would otherwise become a single shared map entry,
        # collapsing every channel onto one label.
        raw = utt.speaker if utt.speaker is not None else f"channel:{channel}"
        if raw not in speaker_map:
            speaker_map[raw] = f"Speaker {len(speaker_map) + 1}"
        speaker = speaker_map[raw]

        words = [
            {
                "text": w.text,
                # AssemblyAI returns milliseconds; seconds are friendlier for
                # both the audio element and the stats.
                "start": w.start / 1000.0,
                "end": w.end / 1000.0,
                "confidence": w.confidence,
                "speaker": speaker,
                "channel": getattr(w, "channel", None) or channel,
            }
            for w in (utt.words or [])
        ]
        flat_words.extend(words)

        utterances.append(
            {
                "speaker": speaker,
                "channel": channel,
                "text": utt.text,
                "start": utt.start / 1000.0,
                "end": utt.end / 1000.0,
                "confidence": utt.confidence,
                "words": words,
            }
        )

    return {
        "file": path.name,
        "model": config.speech_model,
        "language_code": config.language_code,
        "audio_duration": result.audio_duration or 0.0,
        "overall_confidence": result.confidence,
        "speaker_count": len(speaker_map),
        "utterances": utterances,
        "words": flat_words,
    }


# The key that a word with no speaker at all is filed under. A literal None
# would share one map entry with nothing else, but naming it keeps it from
# colliding with a real speaker_id.
_NO_SPEAKER = "\x00none"


def _confidence_from_logprob(logprob: float | None) -> float | None:
    """ElevenLabs reports log probability; AssemblyAI reports a probability.

    ``exp`` is the right transform — a logprob lives in [-inf, 0], so this
    lands in (0, 1] — but the result is *not* AssemblyAI's confidence in
    different units. Neither number is calibrated against the other, so a word
    at 0.55 from one provider is not less certain than a word at 0.65 from the
    other. Compare within a provider, never across.
    """
    if logprob is None:
        return None
    return min(1.0, max(0.0, math.exp(float(logprob))))


def utterances_from_words(
    words: Sequence[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group ElevenLabs' flat word list into utterances. Returns (utterances, words).

    Scribe returns words and no utterances, so unlike the AssemblyAI path these
    boundaries are ours rather than the provider's. That matters because DER's
    false-alarm term is measured against them directly, so this rule is part of
    the result and not an implementation detail.

    **A speaker change is the only thing that starts a new utterance.** There is
    no silence-gap split, deliberately: a gap threshold is a free parameter with
    no principled value, and DER moves with it, so choosing one would amount to
    choosing the number the experiment reports. The cost is that a long turn
    containing pauses stays a single utterance and is charged false alarm across
    the silence, where AssemblyAI's own segmenter would have broken it up. That
    asymmetry is real, is why cpWER rather than DER is the headline metric, and
    has to be stated wherever DER for the two providers appears together.

    Duck-typed so tests can pass plain objects instead of SDK models.
    """
    groups: list[dict[str, Any]] = []
    pending: list[Any] = []  # spacing tokens not yet claimed by a group

    for token in words or []:
        kind = getattr(token, "type", None) or "word"
        # Laughter, footsteps and the like. Dropped before anything else: they
        # carry no text and would otherwise open or close a speaker group.
        if kind == "audio_event":
            continue
        if kind == "spacing":
            # Inert: never opens or closes a group, never carries a speaker or
            # a timestamp. Held until the next real word says which side of a
            # boundary it belongs to.
            if groups:
                pending.append(token)
            continue

        raw = getattr(token, "speaker_id", None) or _NO_SPEAKER
        if groups and groups[-1]["raw"] == raw:
            groups[-1]["tokens"].extend(pending)
        else:
            # The space between two speakers belongs to neither, and giving it
            # to either would be arbitrary.
            groups.append({"raw": raw, "tokens": []})
        pending = []
        groups[-1]["tokens"].append(token)

    speaker_map: dict[str, str] = {}
    utterances: list[dict[str, Any]] = []
    flat_words: list[dict[str, Any]] = []
    # Words can arrive without timings. `diarisation.align` takes each word's
    # midpoint, so None would crash it; a zero-width word at the last known
    # instant is the least invented placement available.
    last_end = 0.0

    for group in groups:
        raw = group["raw"]
        if raw not in speaker_map:
            speaker_map[raw] = f"Speaker {len(speaker_map) + 1}"
        speaker = speaker_map[raw]

        text = "".join(getattr(t, "text", "") or "" for t in group["tokens"]).strip()
        spoken = [
            t
            for t in group["tokens"]
            if (getattr(t, "type", None) or "word") != "spacing"
        ]

        group_words: list[dict[str, Any]] = []
        for token in spoken:
            start = getattr(token, "start", None)
            end = getattr(token, "end", None)
            start = last_end if start is None else float(start)
            end = start if end is None else float(end)
            last_end = max(last_end, end)
            group_words.append(
                {
                    "text": getattr(token, "text", "") or "",
                    "start": start,
                    "end": end,
                    "confidence": _confidence_from_logprob(
                        getattr(token, "logprob", None)
                    ),
                    "speaker": speaker,
                    "channel": getattr(token, "channel_index", None),
                }
            )
        flat_words.extend(group_words)

        confidences = [
            w["confidence"] for w in group_words if w["confidence"] is not None
        ]
        utterances.append(
            {
                "speaker": speaker,
                "channel": group_words[0]["channel"] if group_words else None,
                "text": text,
                "start": group_words[0]["start"] if group_words else last_end,
                "end": group_words[-1]["end"] if group_words else last_end,
                # Derived. Scribe has no utterance-level confidence because it
                # has no utterances.
                "confidence": (
                    sum(confidences) / len(confidences) if confidences else None
                ),
                "words": group_words,
            }
        )

    return utterances, flat_words


def _parse_elevenlabs(
    result: Any, path: Path, config: TranscribeConfig
) -> dict[str, Any]:
    """Flatten a Scribe response into the same shape as ``_parse``.

    ``language_code`` is the language Scribe *detected*, not the one requested:
    no language is sent, so this is the only record of what it decided. A file
    that came back as something other than English has numbers that say nothing
    about diarisation.
    """
    utterances, flat_words = utterances_from_words(getattr(result, "words", None) or [])
    confidences = [w["confidence"] for w in flat_words if w["confidence"] is not None]

    return {
        "file": path.name,
        "model": config.model_id or ELEVENLABS_MODEL,
        "language_code": getattr(result, "language_code", None),
        "audio_duration": getattr(result, "audio_duration_secs", None) or 0.0,
        # Derived, unlike AssemblyAI's, which is reported by the API.
        "overall_confidence": (
            sum(confidences) / len(confidences) if confidences else None
        ),
        "speaker_count": len({u["speaker"] for u in utterances}),
        "utterances": utterances,
        "words": flat_words,
    }
