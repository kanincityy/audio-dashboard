"""What every number means, in words a non-specialist can act on.

The extractors produce facts. This module turns a fact into a judgement — is
-18.4 dBFS fine or a problem? — and into two sentences of plain English: what
was measured, and why an ASR model cares.

It is the single source for that. The metric cards and the plot captions both
read it, so a number and its explanation can never drift apart.

Pure and dependency-free, and tested for that reason.

On provenance: every metric names where its bands came from in ``source``.
Some are thresholds from ``notes.md``, imported from ``thresholds`` rather
than restated. The rest are conventional ranges with nothing
citable behind them, and they say so, because a tuned threshold and a guess
should not look alike on screen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .thresholds import (
    BANDWIDTH_RATIO,
    HIGH_PITCH_HZ,
    OVERLAP_RATIO,
    QUIET_RMS_DBFS,
    RT60_S,
    SILENCE_RATIO,
    SNR_DB,
)
from .thresholds import NARROWBAND_HZ as _NARROWBAND_HZ

# Verdict for a single number. A value nobody measured is not a value that
# passed, so UNKNOWN is its own status rather than a quiet GOOD.
GOOD = "good"
CAUTION = "caution"
PROBLEM = "problem"
UNKNOWN = "unknown"
# Facts with no good or bad about them — a codec name, a channel layout.
INFO = "info"

NARROWBAND_KHZ = _NARROWBAND_HZ / 1000.0

# Where a set of bands came from. Shown on the card.
FROM_NOTES = "threshold from notes.md"
CONVENTIONAL = "conventional range, not a tuned threshold"
FROM_CODE = "matches the detector's own constant"


@dataclass(frozen=True)
class Band:
    """One region of a metric's scale, up to and including ``upper``."""

    upper: float | None  # None means open-ended, and must close the list
    status: str
    label: str


@dataclass(frozen=True)
class Metric:
    """One number, with everything needed to render and explain it."""

    analysis: str  # must exist in registry.BY_NAME
    key: str  # key inside that analysis's result
    name: str  # plain English, not the jargon term
    unit: str
    what: str  # what is being measured
    why: str  # why it changes a transcript
    source: str
    bands: tuple[Band, ...] = ()
    scale: tuple[float, float] | None = None  # bar range; None draws no bar
    precision: int = 1
    kind: str = "number"  # number | flag | text
    # For flags: the status each way round. Most flags are bad when true, but
    # not all of them — "single speaker" is a flag that is good when true.
    true_status: str = PROBLEM
    false_status: str = GOOD
    true_label: str = "yes"
    false_label: str = "no"


def _n(
    analysis: str,
    key: str,
    name: str,
    unit: str,
    what: str,
    why: str,
    source: str,
    bands: tuple[Band, ...],
    scale: tuple[float, float],
    precision: int = 1,
) -> Metric:
    return Metric(
        analysis=analysis,
        key=key,
        name=name,
        unit=unit,
        what=what,
        why=why,
        source=source,
        bands=bands,
        scale=scale,
        precision=precision,
    )


def _flag(
    analysis: str,
    key: str,
    name: str,
    what: str,
    why: str,
    source: str,
    *,
    true_status: str = PROBLEM,
    false_status: str = GOOD,
    true_label: str = "yes",
    false_label: str = "no",
) -> Metric:
    return Metric(
        analysis=analysis,
        key=key,
        name=name,
        unit="",
        what=what,
        why=why,
        source=source,
        kind="flag",
        true_status=true_status,
        false_status=false_status,
        true_label=true_label,
        false_label=false_label,
    )


def _text(
    analysis: str, key: str, name: str, what: str, why: str, source: str = ""
) -> Metric:
    return Metric(
        analysis=analysis,
        key=key,
        name=name,
        unit="",
        what=what,
        why=why,
        source=source or CONVENTIONAL,
        kind="text",
    )


# --------------------------------------------------------------- the catalogue

METRICS: tuple[Metric, ...] = (
    # ------------------------------------------ 1. format and technical metadata
    _n(
        "format",
        "sample_rate_khz",
        "Recording bandwidth",
        "kHz",
        "How many times per second the audio was sampled, which fixes the "
        "highest sound frequency the file can contain.",
        "Phone audio at 8 kHz has no high frequencies at all, so consonants "
        "like s and f are guesswork. Models trained on studio audio expect "
        "detail that is simply not in the file.",
        FROM_NOTES,
        (
            Band(NARROWBAND_KHZ, PROBLEM, "telephony band"),
            Band(15.9, CAUTION, "below the usual model rate"),
            Band(None, GOOD, "wideband"),
        ),
        (0, 48),
        precision=3,
    ),
    _n(
        "format",
        "channels",
        "Channel count",
        "",
        "How many separate audio tracks the file holds — one for mono, two "
        "for stereo, more for a multitrack recording.",
        "When each speaker has their own channel, splitting them before "
        "transcription removes cross-talk entirely. Mixed down to one track, "
        "that information is gone.",
        FROM_NOTES,
        (Band(1, GOOD, "mono"), Band(None, CAUTION, "split these first")),
        (1, 8),
        precision=0,
    ),
    _text(
        "format",
        "channel_layout",
        "Channel layout",
        "How the channels are arranged, as the container declares them.",
        "Tells you whether a two-channel file is genuine stereo or the same "
        "mono signal copied twice.",
    ),
    _text(
        "format",
        "codec",
        "Codec",
        "The compression scheme the audio was stored with.",
        "Speech codecs such as GSM and AMR throw away detail to save "
        "bandwidth, and what they throw away is exactly what an ASR model "
        "listens to.",
    ),
    _n(
        "format",
        "bit_rate_kbps",
        "Bitrate",
        "kbps",
        "How many bits per second the file spends on the audio.",
        "Below about 32 kbps the compression starts inventing spectral "
        "detail, and models hallucinate text to match it.",
        FROM_CODE,
        (
            Band(32, PROBLEM, "heavily compressed"),
            Band(64, CAUTION, "low"),
            Band(None, GOOD, "plenty"),
        ),
        (0, 320),
    ),
    _text(
        "format",
        "bit_depth",
        "Bit depth",
        "How finely each sample is measured, in bits.",
        "16 bits is standard. Below that, quiet passages sit on audible "
        "quantisation noise.",
    ),
    _n(
        "format",
        "duration_s",
        "Length",
        "s",
        "How long the file is.",
        "Very short clips are the single worst case in this whole list: with "
        "almost no context, autoregressive models invent plausible text "
        "rather than admitting they heard nothing.",
        FROM_NOTES,
        (
            Band(3, PROBLEM, "very short"),
            Band(10, CAUTION, "short"),
            Band(None, GOOD, "enough context"),
        ),
        (0, 60),
        precision=2,
    ),
    _flag(
        "format",
        "narrowband",
        "Telephony band",
        "Whether the sample rate is 8 kHz or below.",
        "Route these to an engine trained on call-centre audio rather than "
        "upsampling them and hoping.",
        FROM_NOTES,
        true_label="telephony",
        false_label="wideband",
    ),
    _flag(
        "format",
        "multichannel",
        "More than one channel",
        "Whether the file carries more than a single track.",
        "Split per channel before transcription to remove cross-talk.",
        FROM_NOTES,
        true_status=CAUTION,
        true_label="multi-channel",
        false_label="mono",
    ),
    _flag(
        "format",
        "heavy_compression",
        "Heavy compression",
        "Whether the codec is a lossy speech codec, or the bitrate is under "
        "32 kbps.",
        "Expect spectral distortion and invented words. Treat low-confidence "
        "spans in the transcript with more suspicion than usual.",
        FROM_CODE,
        true_label="heavily compressed",
        false_label="not heavily compressed",
    ),
    # ------------------------------------ 2. signal quality and acoustic metrics
    _n(
        "levels",
        "rms_dbfs",
        "Average loudness",
        "dBFS",
        "The average level of the whole file, where 0 is the loudest a "
        "digital file can be and everything real is negative.",
        "Too quiet and the model is listening to the noise floor as much as "
        "to the speech. Normalising the level first is cheap and usually "
        "helps.",
        FROM_NOTES,
        (
            Band(QUIET_RMS_DBFS, PROBLEM, "too quiet"),
            Band(-12, GOOD, "healthy"),
            Band(None, CAUTION, "very hot"),
        ),
        (-60, 0),
        precision=2,
    ),
    _n(
        "levels",
        "peak_dbfs",
        "Loudest moment",
        "dBFS",
        "The single loudest sample in the file, on the same scale where 0 is "
        "the ceiling.",
        "Sitting right on 0 means the waveform was cut off at the top, which "
        "destroys the shape of the sound rather than merely making it loud.",
        CONVENTIONAL,
        (
            Band(-20, CAUTION, "quiet, room to normalise"),
            Band(-0.5, GOOD, "healthy headroom"),
            Band(None, CAUTION, "touching the ceiling"),
        ),
        (-60, 0),
        precision=2,
    ),
    _n(
        "levels",
        "crest_factor_db",
        "Peak-to-average ratio",
        "dB",
        "The gap between the loudest moment and the average level.",
        "Natural speech breathes, giving roughly 12 to 20 dB. A much smaller "
        "gap means the audio was compressed or limited, which flattens the "
        "cues a model uses to find word boundaries.",
        CONVENTIONAL,
        (
            Band(6, CAUTION, "squashed by a compressor"),
            Band(25, GOOD, "natural for speech"),
            Band(None, CAUTION, "very spiky"),
        ),
        (0, 30),
        precision=2,
    ),
    _n(
        "levels",
        "dynamic_range_db",
        "Loud-to-quiet spread",
        "dB",
        "The gap between a loud moment of speech and the quiet background, "
        "measured across the file.",
        "A wide spread is normal. A very narrow one means the background is "
        "nearly as loud as the speech.",
        CONVENTIONAL,
        (
            Band(6, CAUTION, "background nearly as loud as speech"),
            Band(40, GOOD, "normal"),
            Band(None, CAUTION, "unusually wide"),
        ),
        (0, 60),
        precision=2,
    ),
    _n(
        "clipping",
        "clipped_run_count",
        "Clipping runs",
        "",
        "How many times the waveform ran flat along the maximum level for "
        "three samples or more.",
        "Clipping flattens the peaks of the waveform, and phonetic boundary "
        "detection depends on those peaks. There is no clean way to undo it "
        "after the fact — the information is gone.",
        FROM_CODE,
        (
            Band(0, GOOD, "none"),
            Band(10, CAUTION, "occasional"),
            Band(None, PROBLEM, "widespread"),
        ),
        (0, 50),
        precision=0,
    ),
    _n(
        "clipping",
        "clipped_sample_pct",
        "Samples at maximum",
        "%",
        "The share of individual samples sitting at the top of the scale.",
        "On its own this is weak evidence — a loud recording touches the "
        "ceiling now and then. The run count is the number to trust.",
        CONVENTIONAL,
        (
            Band(0.01, GOOD, "clean"),
            Band(0.5, CAUTION, "some"),
            Band(None, PROBLEM, "a lot"),
        ),
        (0, 2),
        precision=4,
    ),
    _n(
        "clipping",
        "clipped_run_pct",
        "Audio inside clipping runs",
        "%",
        "How much of the file sits inside a flattened run.",
        "This is the share of the recording that is genuinely damaged, as "
        "opposed to merely loud.",
        CONVENTIONAL,
        (
            Band(0.01, GOOD, "negligible"),
            Band(0.5, CAUTION, "noticeable"),
            Band(None, PROBLEM, "severe"),
        ),
        (0, 2),
        precision=4,
    ),
    _n(
        "clipping",
        "longest_run_samples",
        "Longest flat run",
        "samples",
        "The longest single stretch stuck at the maximum level.",
        "A long run is a sustained shout or a badly set input gain, not a "
        "one-off transient.",
        CONVENTIONAL,
        (
            Band(0, GOOD, "none"),
            Band(50, CAUTION, "short bursts"),
            Band(None, PROBLEM, "sustained"),
        ),
        (0, 500),
        precision=0,
    ),
    _n(
        "snr",
        "snr_db",
        "Speech above background noise",
        "dB",
        "How much louder the speech is than the background, in decibels. "
        "Every 6 dB is roughly a doubling in level.",
        "This is the strongest single predictor of transcription errors in "
        "the whole set. Below about 12 dB, engines start dropping and "
        "substituting words, and noise suppression becomes worth its cost. "
        "Accurate to under a decibel near that threshold; reads low on very "
        "clean audio.",
        FROM_NOTES,
        (
            Band(SNR_DB, PROBLEM, "noisy"),
            Band(20, CAUTION, "audible background"),
            Band(None, GOOD, "clean"),
        ),
        (-5, 40),
        precision=2,
    ),
    _n(
        "snr",
        "noise_floor_dbfs",
        "Background noise level",
        "dBFS",
        "How loud the file is during the moments nobody is speaking.",
        "This is the hiss, hum or room tone the model has to hear through.",
        CONVENTIONAL,
        (
            Band(-50, GOOD, "quiet room"),
            Band(-35, CAUTION, "audible"),
            Band(None, PROBLEM, "loud"),
        ),
        (-90, 0),
        precision=2,
    ),
    _n(
        "snr",
        "speech_level_dbfs",
        "Speech level",
        "dBFS",
        "How loud the file is while somebody is speaking.",
        "The distance between this and the background level is the "
        "signal-to-noise ratio above.",
        CONVENTIONAL,
        (
            Band(QUIET_RMS_DBFS, PROBLEM, "very quiet speech"),
            Band(-12, GOOD, "healthy"),
            Band(None, CAUTION, "very hot"),
        ),
        (-60, 0),
        precision=2,
    ),
    _text(
        "snr",
        "method",
        "How this was measured",
        "Either 'vad', meaning speech and silence were told apart by a model, "
        "or 'percentile', meaning they were guessed from loudness alone.",
        "The percentile fallback is a rough guess and can be several decibels "
        "out. Tick voice activity detection for a real measurement.",
    ),
    _n(
        "bandwidth",
        "effective_cutoff_khz",
        "Where the sound actually stops",
        "kHz",
        "The frequency above which the recording holds essentially nothing, "
        "found by looking for a sharp drop in the spectrum.",
        "A file can claim 16 kHz and still contain nothing above 4 kHz — an "
        "8 kHz phone call padded out. The declared rate then tells you "
        "nothing useful and the routing decision has to come from here.",
        CONVENTIONAL,
        (
            Band(4.2, PROBLEM, "telephony"),
            Band(7, CAUTION, "restricted"),
            Band(None, GOOD, "full speech band"),
        ),
        (0, 24),
        precision=2,
    ),
    _n(
        "bandwidth",
        "cutoff_to_nyquist",
        "Share of the available band used",
        "",
        "How much of the frequency range the sample rate allows is actually "
        "carrying sound. 1.0 means all of it.",
        "Well under 1 means the top of the range is empty, so the file was "
        "upsampled from something narrower and a model will listen for "
        "high-frequency consonants that were never recorded.",
        FROM_NOTES,
        (
            Band(BANDWIDTH_RATIO, PROBLEM, "top of the band is empty"),
            Band(0.85, CAUTION, "partly empty"),
            Band(None, GOOD, "fully used"),
        ),
        (0, 1),
        precision=3,
    ),
    _n(
        "bandwidth",
        "cliff_drop_db",
        "Steepness of the cutoff",
        "dB",
        "How far the level falls across the sharpest edge found in the "
        "spectrum.",
        "Speech energy always tapers off gradually with frequency. Only a "
        "near-vertical drop means a resampler or a codec cut the audio short.",
        FROM_CODE,
        (
            Band(20, GOOD, "natural roll-off"),
            Band(None, CAUTION, "sharp cutoff, something cut it"),
        ),
        (0, 60),
        precision=1,
    ),
    _flag(
        "bandwidth",
        "cliff_found",
        "Sharp cutoff found",
        "Whether a near-vertical drop was found in the spectrum.",
        "No cliff means the file is using its whole band, which is the normal "
        "and healthy case.",
        FROM_CODE,
        true_status=CAUTION,
        true_label="found",
        false_label="none — using the whole band",
    ),
    _flag(
        "bandwidth",
        "likely_upsampled",
        "Probably upsampled",
        "Whether the file appears to have been stretched up from a lower "
        "sample rate.",
        "Do not trust the declared sample rate. Route this as narrowband "
        "audio regardless of what the header says.",
        FROM_NOTES,
        true_label="upsampled",
        false_label="genuine",
    ),
    _n(
        "rt60",
        "rt60_s",
        "Room echo",
        "s",
        "Roughly how long a sound takes to die away in the room, estimated "
        "from the tails after each person stops speaking.",
        "A long tail means the microphone was far from the speaker, and each "
        "sound smears into the next. That is a different failure mode from "
        "noise and needs a different fix.",
        FROM_NOTES,
        (
            Band(0.4, GOOD, "dry, close microphone"),
            Band(RT60_S, CAUTION, "some room"),
            Band(None, PROBLEM, "far-field, echoey"),
        ),
        (0, 2),
        precision=3,
    ),
    _n(
        "rt60",
        "windows",
        "Decay windows used",
        "",
        "How many pauses in the speech were long and clean enough to measure "
        "the echo from.",
        "This is the honest confidence figure for the echo estimate. Under "
        "about three windows the number above means very little.",
        CONVENTIONAL,
        (
            Band(2, PROBLEM, "too few to trust"),
            Band(9, CAUTION, "thin evidence"),
            Band(None, GOOD, "enough to trust"),
        ),
        (0, 40),
        precision=0,
    ),
    _n(
        "rt60",
        "rt60_iqr_s",
        "Spread across windows",
        "s",
        "How much the individual echo measurements disagreed with each other.",
        "A wide spread means the estimate is unstable and should not drive a "
        "routing decision on its own.",
        CONVENTIONAL,
        (
            Band(0.2, GOOD, "consistent"),
            Band(0.6, CAUTION, "variable"),
            Band(None, PROBLEM, "the windows disagree"),
        ),
        (0, 2),
        precision=3,
    ),
    _flag(
        "rt60",
        "far_field",
        "Far-field recording",
        "Whether the echo estimate is above the far-field threshold.",
        "Route to a dereverberation pipeline, or to an engine trained on "
        "distant microphones.",
        FROM_NOTES,
        true_status=CAUTION,
        true_label="far-field",
        false_label="close-mic",
    ),
    # ------------------------------ 3. speech structure and temporal dynamics
    _n(
        "density",
        "speech_ratio",
        "How much of the file is speech",
        "",
        "The share of the file where somebody is actually talking, as found "
        "by the voice activity model.",
        "Mostly-silent audio makes autoregressive models loop or invent "
        "repeated text to fill the gap. Trimming the silence first is the fix.",
        FROM_NOTES,
        (
            Band(1 - SILENCE_RATIO, PROBLEM, "mostly silence"),
            Band(0.5, CAUTION, "sparse"),
            Band(None, GOOD, "dense"),
        ),
        (0, 1),
        precision=3,
    ),
    _n(
        "density",
        "silence_ratio",
        "How much of the file is silence",
        "",
        "The other side of the same measurement.",
        "Above 70 percent silence, trim before transcribing.",
        FROM_NOTES,
        (
            Band(0.5, GOOD, "normal"),
            Band(SILENCE_RATIO, CAUTION, "quiet file"),
            Band(None, PROBLEM, "mostly silence"),
        ),
        (0, 1),
        precision=3,
    ),
    _n(
        "density",
        "segment_count",
        "Speech segments",
        "",
        "How many separate stretches of talking the model found.",
        "Zero means no speech was detected at all, which usually means the "
        "file is music, noise, or too quiet to work with.",
        CONVENTIONAL,
        (Band(0, PROBLEM, "no speech found"), Band(None, GOOD, "speech found")),
        (0, 50),
        precision=0,
    ),
    _n(
        "density",
        "longest_internal_gap_s",
        "Longest pause",
        "s",
        "The longest silence between two stretches of speech, not counting "
        "the start and end of the file.",
        "Long internal silences are where looping and repeated-text "
        "hallucinations start.",
        CONVENTIONAL,
        (
            Band(2, GOOD, "normal pause"),
            Band(10, CAUTION, "long"),
            Band(None, PROBLEM, "long enough to cause looping"),
        ),
        (0, 30),
        precision=2,
    ),
    _n(
        "density",
        "leading_silence_s",
        "Silence before speech",
        "s",
        "How long the file runs before anyone starts talking.",
        "Almost none means the recording probably started mid-word. A lot "
        "means there is dead air worth trimming.",
        FROM_CODE,
        (
            Band(0.1, CAUTION, "starts at the very edge"),
            Band(5, GOOD, "normal"),
            Band(None, CAUTION, "dead air to trim"),
        ),
        (0, 10),
        precision=2,
    ),
    _n(
        "density",
        "trailing_silence_s",
        "Silence after speech",
        "s",
        "How long the file runs on after the last word.",
        "Almost none means the recording was probably cut mid-word.",
        FROM_CODE,
        (
            Band(0.1, CAUTION, "ends at the very edge"),
            Band(5, GOOD, "normal"),
            Band(None, CAUTION, "dead air to trim"),
        ),
        (0, 10),
        precision=2,
    ),
    _n(
        "density",
        "median_segment_s",
        "Typical segment length",
        "s",
        "How long a typical unbroken stretch of speech lasts.",
        "Very short segments mean the speech is fragmented, which makes "
        "diarisation and punctuation unreliable.",
        CONVENTIONAL,
        (Band(0.5, CAUTION, "fragmented"), Band(None, GOOD, "continuous")),
        (0, 10),
        precision=2,
    ),
    _flag(
        "density",
        "sparse_speech",
        "Mostly silence",
        "Whether silence takes up more than 70 percent of the file.",
        "Trim the silence before transcribing.",
        FROM_NOTES,
        true_label="sparse",
        false_label="dense enough",
    ),
    _flag(
        "truncation",
        "short_file",
        "Under two seconds",
        "Whether the whole file is shorter than two seconds.",
        "Short clips have the highest hallucination rate of any category in "
        "this list.",
        FROM_NOTES,
        true_label="very short",
        false_label="long enough",
    ),
    _flag(
        "truncation",
        "speech_at_start",
        "Cut off at the start",
        "Whether speech is already running when the file begins.",
        "The recording was probably cut mid-utterance, so the first words are "
        "missing and the model has no run-up.",
        FROM_CODE,
        true_label="cut mid-word",
        false_label="clean start",
    ),
    _flag(
        "truncation",
        "speech_at_end",
        "Cut off at the end",
        "Whether speech is still running when the file ends.",
        "The last words are missing, and models tend to invent an ending "
        "rather than stop mid-sentence.",
        FROM_CODE,
        true_label="cut mid-word",
        false_label="clean end",
    ),
    _n(
        "truncation",
        "word_count",
        "Words transcribed",
        "",
        "How many words the transcript contains.",
        "Six words or fewer gives the model almost no context to work from, "
        "which is where invented text comes from.",
        FROM_NOTES,
        (
            Band(6, PROBLEM, "too few for context"),
            Band(20, CAUTION, "short"),
            Band(None, GOOD, "plenty of context"),
        ),
        (0, 200),
        precision=0,
    ),
    _flag(
        "truncation",
        "hallucination_risk",
        "Elevated hallucination risk",
        "Whether the file is short, or has speech touching either end.",
        "Check the source segmentation before trusting anything in this "
        "transcript.",
        FROM_NOTES,
        true_label="elevated",
        false_label="not elevated",
    ),
    _n(
        "tempo",
        "wpm",
        "Speaking speed",
        "wpm",
        "Words per minute, counted against time actually spent speaking "
        "rather than against the length of the file.",
        "Fast speech causes skipped words. A file that is half silence is not "
        "being spoken at half speed, which is why the silence is excluded.",
        CONVENTIONAL,
        (
            Band(100, CAUTION, "slow"),
            Band(180, GOOD, "typical conversation"),
            Band(220, CAUTION, "fast"),
            Band(None, PROBLEM, "very fast, expect skipped words"),
        ),
        (0, 300),
    ),
    _n(
        "tempo",
        "wpm_median",
        "Typical utterance speed",
        "wpm",
        "The speed of a middling utterance, rather than the average across "
        "the file.",
        "Less sensitive than the overall figure to one long pause or one "
        "rushed sentence.",
        CONVENTIONAL,
        (
            Band(100, CAUTION, "slow"),
            Band(180, GOOD, "typical"),
            Band(220, CAUTION, "fast"),
            Band(None, PROBLEM, "very fast"),
        ),
        (0, 300),
    ),
    _n(
        "tempo",
        "wpm_p90",
        "Speed of the fastest passages",
        "wpm",
        "The speed of the fastest tenth of utterances.",
        "One rushed passage is enough to lose words, and it disappears "
        "entirely into an average.",
        CONVENTIONAL,
        (
            Band(200, GOOD, "manageable"),
            Band(260, CAUTION, "fast passages"),
            Band(None, PROBLEM, "very fast passages"),
        ),
        (0, 350),
    ),
    _n(
        "tempo",
        "fast_utterance_count",
        "Utterances above 200 wpm",
        "",
        "How many individual utterances were spoken faster than 200 words a "
        "minute.",
        "These are the specific places to check the transcript for dropped "
        "words.",
        CONVENTIONAL,
        (
            Band(0, GOOD, "none"),
            Band(5, CAUTION, "a few"),
            Band(None, PROBLEM, "many"),
        ),
        (0, 30),
        precision=0,
    ),
    _n(
        "overlap",
        "overlap_ratio",
        "Time with two people talking",
        "",
        "The share of the file where the diariser placed two speakers at "
        "once.",
        "Cross-talk is the hardest case for a single-stream engine — it "
        "transcribes one voice and loses the other. On mono audio this cannot "
        "be measured at all, and reports itself as unmeasured rather than as "
        "zero.",
        FROM_NOTES,
        (
            Band(0.05, GOOD, "little"),
            Band(OVERLAP_RATIO, CAUTION, "some"),
            Band(None, PROBLEM, "heavy cross-talk"),
        ),
        (0, 1),
        precision=3,
    ),
    _n(
        "overlap",
        "unattributed_speech_ratio",
        "Speech nobody was assigned",
        "",
        "The share of the file where the voice activity model heard speech "
        "but the diariser gave it to no speaker at all.",
        "This is the independent cross-check on the overlap figure above. It "
        "does not depend on the diariser emitting overlapping spans, so it "
        "catches cross-talk that the overlap number misses entirely.",
        CONVENTIONAL,
        (
            Band(0.02, GOOD, "negligible"),
            Band(0.10, CAUTION, "some unaccounted speech"),
            Band(None, PROBLEM, "a lot unaccounted for"),
        ),
        (0, 1),
        precision=3,
    ),
    _flag(
        "overlap",
        "high_overlap",
        "Heavy cross-talk",
        "Whether overlapping speech passes 15 percent of the file.",
        "Run source separation or multi-channel diarisation before "
        "transcribing.",
        FROM_NOTES,
        true_label="heavy",
        false_label="within limits",
    ),
    # ---------------------------- 4. speaker and demographic profiling
    _n(
        "speakers",
        "speaker_count",
        "Speakers",
        "",
        "How many distinct voices the diariser found.",
        "One speaker can take a cheaper, faster model. More than one needs an "
        "endpoint that labels who said what.",
        FROM_NOTES,
        (
            Band(1, GOOD, "single speaker"),
            Band(4, CAUTION, "needs diarisation"),
            Band(None, CAUTION, "many speakers, harder to separate"),
        ),
        (0, 10),
        precision=0,
    ),
    _flag(
        "speakers",
        "single_speaker",
        "Single speaker",
        "Whether only one voice was found.",
        "Dictation from one speaker is the cheapest case to route.",
        FROM_NOTES,
        true_status=GOOD,
        false_status=CAUTION,
        true_label="one voice",
        false_label="several voices",
    ),
    _flag(
        "speakers",
        "needs_diarisation",
        "Needs diarisation",
        "Whether more than one voice was found.",
        "Use an endpoint that labels speakers, or the transcript runs the "
        "voices together.",
        FROM_NOTES,
        true_status=CAUTION,
        true_label="needed",
        false_label="not needed",
    ),
    _n(
        "f0",
        "f0_median_hz",
        "Voice pitch",
        "Hz",
        "The middle of the speaker's pitch range. Roughly 85 to 180 Hz is a "
        "typical adult male voice, 165 to 255 Hz a typical adult female one, "
        "and children sit above that.",
        "Models are trained mostly on adult read speech. Voices well above "
        "that range — children, or excited speakers — see word error rates "
        "spike.",
        FROM_NOTES,
        (
            Band(85, CAUTION, "very low"),
            Band(HIGH_PITCH_HZ, GOOD, "usual adult range"),
            Band(None, PROBLEM, "high, expect more errors"),
        ),
        (50, 400),
    ),
    _n(
        "f0",
        "f0_iqr_hz",
        "Pitch variation",
        "Hz",
        "How much the pitch moves around its middle.",
        "Almost no variation usually means the pitch tracker failed rather "
        "than that the speaker is monotone. A very wide spread on a "
        "single-speaker file means the same.",
        CONVENTIONAL,
        (
            Band(10, CAUTION, "flat — check the tracker"),
            Band(80, GOOD, "normal expressiveness"),
            Band(None, CAUTION, "wide — several voices, or tracking errors"),
        ),
        (0, 150),
    ),
    _n(
        "f0",
        "voiced_fraction",
        "Pitch found in this much speech",
        "",
        "The share of detected speech where a pitch could actually be "
        "measured.",
        "Consonants have no pitch, so this is never 1. But a low figure means "
        "the tracker struggled, and the pitch numbers above are then weak "
        "evidence.",
        CONVENTIONAL,
        (
            Band(0.3, PROBLEM, "tracking mostly failed"),
            Band(0.5, CAUTION, "patchy"),
            Band(None, GOOD, "reliable"),
        ),
        (0, 1),
        precision=3,
    ),
    _n(
        "f0",
        "f0_min_hz",
        "Lowest pitch",
        "Hz",
        "The lowest pitch measured anywhere in the speech.",
        "Close to 50 Hz usually means the tracker locked onto a harmonic "
        "rather than the voice.",
        CONVENTIONAL,
        (
            Band(55, CAUTION, "at the floor of the search range"),
            Band(None, GOOD, "plausible"),
        ),
        (50, 400),
    ),
    _n(
        "f0",
        "f0_max_hz",
        "Highest pitch",
        "Hz",
        "The highest pitch measured anywhere in the speech.",
        "Close to 500 Hz usually means the tracker locked onto a harmonic "
        "rather than the voice.",
        CONVENTIONAL,
        (
            Band(480, GOOD, "plausible"),
            Band(None, CAUTION, "at the ceiling of the search range"),
        ),
        (50, 500),
    ),
    _flag(
        "f0",
        "high_pitch",
        "High-pitched voice",
        "Whether the median pitch is above the point where error rates start "
        "climbing.",
        "Prefer an engine with broader pitch coverage than adult read speech.",
        FROM_NOTES,
        true_label="high",
        false_label="usual range",
    ),
)

BY_ANALYSIS: dict[str, tuple[Metric, ...]] = {}
for _m in METRICS:
    BY_ANALYSIS[_m.analysis] = BY_ANALYSIS.get(_m.analysis, ()) + (_m,)


# Flags that only restate a number another card already shows — the same
# measurement with a threshold applied. They belong on screen, because the
# threshold is the routing decision, but not as full cards: a grid of
# near-duplicates is exactly the clutter this layout exists to remove. The UI
# renders these as a strip of chips under the cards they refer to.
COMPACT_FLAGS = frozenset(
    {
        ("format", "narrowband"),
        ("format", "multichannel"),
        ("format", "heavy_compression"),
        ("bandwidth", "cliff_found"),
        ("bandwidth", "likely_upsampled"),
        ("rt60", "far_field"),
        ("density", "sparse_speech"),
        ("truncation", "short_file"),
        ("truncation", "speech_at_start"),
        ("truncation", "speech_at_end"),
        ("overlap", "high_overlap"),
        ("speakers", "single_speaker"),
        ("speakers", "needs_diarisation"),
        ("f0", "high_pitch"),
    }
)


def is_compact(metric: Metric) -> bool:
    """Whether this belongs in the chip strip rather than on a card."""
    return (metric.analysis, metric.key) in COMPACT_FLAGS


def classify(metric: Metric, value: Any) -> tuple[str, str]:
    """The status of one value, and the word that describes where it sits.

    A value that is absent, or that came from an analysis which failed or
    declined, is UNKNOWN — never GOOD. Not measured must not read as measured
    and fine.
    """
    if value is None:
        return UNKNOWN, "not measured"

    if metric.kind == "flag":
        if bool(value):
            return metric.true_status, metric.true_label
        return metric.false_status, metric.false_label

    if metric.kind == "text":
        return INFO, str(value)

    for band in metric.bands:
        if band.upper is None or float(value) <= band.upper:
            return band.status, band.label
    # Unreachable while the last band is open-ended, which is enforced by test.
    return UNKNOWN, "off the scale"


def metric_value(results: dict[str, Any], analysis: str, key: str) -> Any:
    """Read a metric, treating a failed or unusable analysis as absent.

    A card that cannot see a number must not colour itself green, so an
    analysis that errored or flagged itself unusable reads as None.
    """
    block = results.get(analysis)
    if not isinstance(block, dict) or "error" in block:
        return None
    if block.get("usable") is False:
        return None
    return block.get(key)


def value_of(results: dict[str, Any], metric: Metric) -> Any:
    """Read a metric out of a results dict, or None if it is not there."""
    return metric_value(results, metric.analysis, metric.key)


def read(results: dict[str, Any], metric: Metric) -> tuple[Any, str, str]:
    """Value, status and label for one metric against a results dict."""
    value = value_of(results, metric)
    status, label = classify(metric, value)
    return value, status, label


def format_value(metric: Metric, value: Any) -> str:
    """The number as it should appear on a card."""
    if value is None:
        return "—"
    if metric.kind == "flag":
        return metric.true_label if value else metric.false_label
    if metric.kind == "text":
        return str(value) or "—"
    if isinstance(value, (int, float)):
        return f"{float(value):.{metric.precision}f}"
    return str(value)


# ------------------------------------------------------------------- the plots

@dataclass(frozen=True)
class PlotGuide:
    """How to read one of the signal plots, for somebody who has not before."""

    caption: str
    normal: str
    abnormal: str


PLOTS: dict[str, PlotGuide] = {
    "waveform": PlotGuide(
        caption="Loudness over time. Green shading is where a model detected "
        "speech; red lines mark places the waveform hit the ceiling and "
        "flattened.",
        normal="Speech looks like a series of bursts with visible gaps "
        "between phrases, filling maybe half to three quarters of the "
        "vertical space. The green shading should line up with the bursts — "
        "if it does not, the voice activity model is struggling, and every "
        "measurement that depends on it is weaker than it looks.",
        abnormal="A solid block that touches the top and bottom edges means "
        "the recording is clipped. A flat line with a few small bumps means "
        "the audio is far too quiet. Green shading over visibly silent "
        "stretches means background noise is loud enough to be mistaken for "
        "speech.",
    ),
    "spectrum": PlotGuide(
        caption="How much energy the recording holds at each frequency, "
        "averaged over the whole file. The dashed line is where the energy "
        "was judged to stop.",
        normal="The line starts high on the left and slopes downwards. That "
        "slope is normal and is what speech looks like — it is not a fault. "
        "The dashed line should sit at the far right, at the highest "
        "frequency the file could hold.",
        abnormal="A near-vertical drop partway across, with the line falling "
        "off a shelf and staying flat afterwards, means something cut the "
        "audio short. A cliff at 4 kHz is a phone call. If that file also "
        "claims a 16 kHz sample rate, it was upsampled and the declared rate "
        "is misleading.",
    ),
    "pitch": PlotGuide(
        caption="The pitch of the voice over time, in hertz. Gaps are moments "
        "with no measurable pitch.",
        normal="A wandering line that stays within a band a hundred hertz or "
        "so wide, broken by gaps. Gaps are expected and not an error: "
        "consonants have no pitch, and neither does silence.",
        abnormal="A line jumping between two widely separated levels usually "
        "means two different speakers rather than one very expressive one — "
        "check the per-speaker figures below the plot, which is where the "
        "routing signal actually lives. A line pinned to the top or bottom "
        "of the plot means the tracker lost the voice and locked onto a "
        "harmonic.",
    ),
}
