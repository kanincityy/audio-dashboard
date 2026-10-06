"""Section 1 of notes.md: format and technical metadata.

Container-level facts, read without decoding. These are the cheapest signals
available and two of them — sample rate and channel count — are the strongest
routing inputs in the whole set, because they are exact rather than estimated.
"""

from __future__ import annotations

from typing import Any

from ..audio_io import AudioBundle, probe
from ..thresholds import NARROWBAND_HZ

# Codecs whose very presence signals heavy compression, regardless of the
# bitrate they report. notes.md:7.
LOSSY_SPEECH_CODECS = {"gsm", "gsm_ms", "amr_nb", "amr_wb", "amrnb", "amrwb", "g723_1"}

LOW_BITRATE_BPS = 32_000


def extract(bundle: AudioBundle, results: dict[str, Any]) -> dict[str, Any]:
    meta = probe(bundle.path)
    fmt = meta.get("format", {})
    streams = [s for s in meta.get("streams", []) if s.get("codec_type") == "audio"]
    stream = streams[0] if streams else {}

    codec = stream.get("codec_name", "")
    # Bitrate lives on the stream for most formats and only on the container
    # for some; ffprobe omits it entirely for lossless wav.
    bit_rate = stream.get("bit_rate") or fmt.get("bit_rate")
    bit_rate = int(bit_rate) if bit_rate else None

    sample_rate = int(stream.get("sample_rate", bundle.sr))
    n_channels = int(stream.get("channels", bundle.n_channels))

    return {
        "sample_rate_hz": sample_rate,
        "sample_rate_khz": round(sample_rate / 1000, 3),
        "channels": n_channels,
        "channel_layout": stream.get("channel_layout", ""),
        "codec": codec,
        "codec_long": stream.get("codec_long_name", ""),
        "container": fmt.get("format_name", ""),
        "bit_rate_bps": bit_rate,
        "bit_rate_kbps": round(bit_rate / 1000, 1) if bit_rate else None,
        "bit_depth": stream.get("bits_per_raw_sample") or stream.get("bits_per_sample"),
        "duration_s": round(bundle.duration, 3),
        "audio_stream_count": len(streams),
        "narrowband": sample_rate <= NARROWBAND_HZ,
        "multichannel": n_channels > 1,
        "heavy_compression": codec in LOSSY_SPEECH_CODECS
        or (bit_rate is not None and bit_rate < LOW_BITRATE_BPS),
    }
