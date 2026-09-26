1. Format & Technical Metadata
These container-level attributes reveal if audio degradation is due to lossy compression or band-limiting.
* Sample Rate (kHz): Distinguishes narrow-band (8 kHz telephony) from wide-band (16 kHz+) audio.
    * Routing impact: standard models (like Whisper base) can struggle with 8 kHz audio. Route 8 kHz telephony files to engines explicitly trained on call-center data (e.g., Deepgram Nova-2 Telephony or Speechmatics).
* Channels & Layout: Identifies mono vs. multi-channel (stereo/multitrack).
    * Routing impact: Multi-channel audio should be split per channel prior to ASR to eliminate cross-talk.
* Audio Codec & Bitrate: Identifies heavy compression formats (GSM, AMR, low-bitrate MP3).
    * Diagnostic impact: Low bitrates often cause spectral distortion that leads models to hallucinate text. 
2. Signal Quality & Acoustic Metrics
Signal quality metrics tell you why an ASR engine made insertion/deletion errors or hallucinated text.
* Signal-to-Noise Ratio (SNR in dB): Measures the ratio of clean speech to background noise. 
    * Routing impact: Low SNR ($<15\text{ dB}$) triggers pre-processing (noise suppression/denoising via DeepFilterNet or RNNoise) or routes to noise-robust models.
* Clipping & Distortion Percentage: Detects digital clipping (where volume exceeds 0 dBFS and flattens waveforms).
    * Diagnostic impact: Severe clipping ruins phonetic boundary detection, leading to high Word Error Rates (WER). 
* High-Frequency Loss / Artificial Upsampling: Probability that the top half of the frequency spectrum is missing. 
    * Diagnostic impact: Upsampled 8 kHz audio padded to 16 kHz misleads models into expecting high-frequency consonants that aren't there.
* Reverberation Time ($\text{RT}_{60}$): Measures echo/room acoustics.
    * Routing impact: High reverberation ($>0.8\text{s}$) indicates far-field mic recording. Route to far-field/dereverberation pipelines or models like Conformer-based STT. 
* Dynamic Range & Peak/RMS Volume: Peak vs. average loudness.
    * Routing impact: Audio that is too quiet (low RMS) requires Gain Normalization prior to ASR.
3. Speech Structure & Temporal Dynamics
These metrics measure how the content is spoken, which heavily impacts model stability.
* Speech-to-Silence Ratio / VAD Density: The proportion of non-silent speech segments vs. total duration. 
    * Diagnostic impact: Audio with long silent stretches causes continuous autoregressive models (like Whisper) to loop or hallucinate repetitive text.
* Utterance Length / Truncation Flags: Flags very short utterances ($<2$ seconds or $\le 6$ words) or clipped ends. 
    * Diagnostic impact: Short/truncated audio has the highest hallucination error rate across mainstream ASR engines.
* Overlapping Speech / Cross-talk Ratio: Percentage of time multiple speakers talk simultaneously.
    * Routing impact: High overlap requires diarization models capable of overlapping speech separation (e.g., PyAnnote) before running transcription.
* Speech Tempo / Words-per-Minute (WPM): Speaking speed derived from voice activity detection (VAD) burst timing.
    * Diagnostic impact: Ultra-fast speech leads to skipped words (deletions). 
4. Speaker & Demographic Profiling
Acoustic characteristics of the speaker's voice.
* Speaker Count (Diarization Hint): Estimated number of distinct speakers.
    * Routing impact: Single-speaker dictation can use fast/cheaper models; multi-speaker meetings require diarization-enabled endpoints.
* Fundamental Frequency ($F_0$) & Pitch Range: Pitch median and variance. 
    * Diagnostic impact: High fundamental pitch (e.g., children's voices or excited speakers) often causes significant WER spikes in models trained primarily on adult read speech. 
* Code-Switching / Dialect Flags: Detection of multiple languages spoken in a single file or strong non-native accents. 
    * Routing impact: Code-switching causes single-language models to fail. Route to multilingual models with explicit code-switching support (e.g., Qwen2-Audio or specialized multilingual STT).
Quick Implementation Matrix for Dynamic Routing
Extracted Signal Signal	Threshold / Condition	Recommended Pre-Processing / Dynamic Route
Low SNR ($<12\text{ dB}$)	Heavy background noise	Apply noise suppression $\rightarrow$ Route to noise-robust ASR engine.
Low Sample Rate ($8\text{ kHz}$)	Telephony / Call center audio	Skip 16 kHz upsampling $\rightarrow$ Route directly to Telephony-tuned ASR.
Short Duration / Sparse VAD	Utterance $<3\text{s}$ or silence $>70\%$	Trim leading/trailing silence $\rightarrow$ Use non-autoregressive or strict-prompt ASR to avoid hallucinations.
High Overlap Ratio	$>15\%$ overlapping speech	Run source separation / multi-channel diarization first.
Low RMS Loudness	$< -30\text{ dBFS}$	Apply peak/loudness normalization prior to feeding ASR.
