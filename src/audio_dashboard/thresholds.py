"""Thresholds from notes.md, as named constants.

They live in one place because tuning them is the likely outcome of using this
tool. ``interpret`` builds its good/caution/problem bands from them, and the
extractors that set a flag at a threshold import it from here, so the flag and
the band shown on screen can never drift apart.
"""

# notes.md:41 and notes.md:12. The matrix says 12 dB, the prose says 15 dB;
# 12 is the harder threshold and the one the matrix acts on.
SNR_DB = 12.0
# notes.md:4. At or below this, the audio is telephony-band and the top of the
# speech spectrum was never recorded.
NARROWBAND_HZ = 8_000
SILENCE_RATIO = 0.70
OVERLAP_RATIO = 0.15
QUIET_RMS_DBFS = -30.0
# notes.md:18. Above this the recording is far-field rather than close-mic.
RT60_S = 0.8
# notes.md:16. Below this share of the available band, the top of the spectrum
# is empty and the file was probably upsampled from something narrower.
BANDWIDTH_RATIO = 0.60
# notes.md:36. Roughly a soprano speaking voice — above what most models saw
# much of in training.
HIGH_PITCH_HZ = 255.0
