"""Loading audio off disk, and asking ffprobe what it is.

Two separate jobs that both start from a path. ``probe`` reads the container
metadata without decoding anything, which is all section 1 of notes.md needs.
``load`` decodes, which everything else needs.

The decoded form is deliberately eager: a file small enough to upload to a
Streamlit app is small enough to hold in memory, and every analysis wants the
same samples, so decoding once and passing the bundle around beats each
extractor opening the file itself.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

# Silero is trained at 16 kHz and 8 kHz only. Everything that depends on VAD
# timing therefore works against a 16 kHz copy, and converts spans back to
# seconds rather than sample indices so the native-rate arrays stay usable.
VAD_SR = 16_000

AUDIO_SUFFIXES = {
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".oga",
    ".opus",
    ".aac",
    ".mp4",
}


@dataclass(frozen=True)
class AudioBundle:
    """One decoded file, in every form the extractors want.

    ``samples`` is mono because most metrics are defined on a single signal.
    ``channels`` keeps the original layout for the few that are not — a stereo
    file where each speaker sits on their own channel is a different situation
    from a true stereo recording, and section 1 is what tells them apart.
    """

    path: Path
    samples: np.ndarray  # mono, float32, native rate
    sr: int
    channels: np.ndarray | None  # (n_channels, n_samples), None when mono
    samples_16k: np.ndarray  # mono, float32, 16 kHz
    duration: float

    @property
    def n_channels(self) -> int:
        return 1 if self.channels is None else int(self.channels.shape[0])


def probe(path: Path) -> dict[str, Any]:
    """Container and stream metadata, straight from ffprobe.

    Returns the raw parsed JSON. Interpreting it is ``features.formatmeta``'s
    job — keeping the parse dumb means a field we do not use today is still
    there when we want it.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _decode_ffmpeg(path: Path, sr: int, n_channels: int) -> np.ndarray:
    """Decode to interleaved float32 PCM at the file's native rate.

    ``-ar``/``-ac`` are deliberately omitted so ffmpeg neither resamples nor
    downmixes: the point of this path is to get the same array soundfile would
    have given us, for the containers it cannot open.
    """
    result = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", str(path), "-f", "f32le", "-"],
        capture_output=True,
        check=True,
    )
    flat = np.frombuffer(result.stdout, dtype=np.float32)
    if n_channels > 1:
        # Drop a trailing partial frame rather than let reshape raise.
        usable = (len(flat) // n_channels) * n_channels
        return flat[:usable].reshape(-1, n_channels)
    return flat.reshape(-1, 1)


def load(path: Path) -> AudioBundle:
    """Decode a file into every representation the extractors need.

    libsndfile handles wav/flac/ogg/opus/mp3 but not the MPEG-4 family, so
    anything it refuses goes through ffmpeg instead. The two paths produce the
    same arrays; only the decoder differs.
    """
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except sf.LibsndfileError:
        meta = probe(path)
        stream = next(
            s for s in meta.get("streams", []) if s.get("codec_type") == "audio"
        )
        sr = int(stream["sample_rate"])
        data = _decode_ffmpeg(path, sr, int(stream.get("channels", 1)))

    # soundfile gives (frames, channels); everything downstream wants
    # (channels, frames) so channel 0 is a contiguous row.
    channels = np.ascontiguousarray(data.T)
    mono = channels.mean(axis=0) if channels.shape[0] > 1 else channels[0]
    mono = np.ascontiguousarray(mono, dtype=np.float32)

    if sr == VAD_SR:
        mono_16k = mono
    else:
        mono_16k = librosa.resample(mono, orig_sr=sr, target_sr=VAD_SR)
        mono_16k = np.ascontiguousarray(mono_16k, dtype=np.float32)

    return AudioBundle(
        path=path,
        samples=mono,
        sr=sr,
        channels=channels if channels.shape[0] > 1 else None,
        samples_16k=mono_16k,
        duration=len(mono) / sr if sr else 0.0,
    )
