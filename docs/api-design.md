### GET /health
  A check that the server is up.

  - **In:** nothing
  - **Out:** `{"status": "ok"}`
  - **Codes:** 200

### GET /v1/analyses
  List the analyses this server can run, so a caller knows which names are valid.

  - **In:** nothing
  - **Out:** `{"analyses": [{"name": "snr", "label": "Signal-to-noise ratio",
    "group": "quality", "cost": "fast", "requires": [], "prefers": ["vad"],
    "note": "..."}, ...]}`
  - **Codes:** 200
  - **Notes:** built from `registry.ANALYSES`. `fn` is deliberately left out: it
    is a Python function and means nothing to a caller. `cost` is how a client
    knows `transcript` is billed.

### POST /v1/files
  - **In:** the audio file (multipart upload)
  - **Out:** `{"digest": "a1b2c3d4e5f60718", "size_bytes": 5242880, "filename": "call.mp3"}`
             The digest is the ID every later call uses.
  - **Codes:** 201 new, 200 repeat, 413 too large, 415 unsupported type, 400 no filename
  - **Notes:** the digest is the first 16 characters of the SHA-256 of the bytes
               (see `hashlib` in `api/files.py`). Bytes are stored in `uploads/` as
               `{digest}{suffix}`, so two different files cannot overwrite each
               other. The size limit is checked twice. The size the client claims is
               checked first, so an oversized upload is refused before it is read.
               The size actually read is checked after.

### POST /v1/files/{digest}/runs
  Start a run. The analyses execute in the background, so this call returns
  straight away.

  - **In:** `{"analyses": ["snr", "vad"], "force": false, "allow_billing": false}`
             `analyses` must name at least one analysis. `force` and `allow_billing`
             both default to false, which is the safe direction. `allow_billing`
             false means a run that would transcribe an uncached file is refused
             before anything runs, rather than billing by surprise.
  - **Out:** `{"run_id": "7f3c...", "ran": ["vad", "snr"], "results_url": "/v1/runs/7f3c..."}`
             plus a `Location` header with the same URL. `ran` is the full list
             that will execute, in order, including analyses that were pulled in
             as requirements. The results are not here. The caller follows the
             pointer and polls it.
  - **Codes:** 202 accepted, 404 no such digest, 422 an analysis name is not in the
               registry, 403 the selection would bill and `allow_billing` is false.
               The 403 body names the analysis, e.g. `{"detail": "transcript would
               be billed for this file; set allow_billing to run it."}`. It is
               checked with `registry.resolve` and `asr.is_cached` before the run
               is saved.
  - **Notes:** the run is saved as `queued` before the response goes out, so the
               `run_id` can be fetched immediately. The work then runs as a
               FastAPI background task (`registry.run`). It lives inside the server
               process, so a restart ends it (see the next endpoint).

### GET /v1/runs/{run_id}
  The state of one run, and its results once it is done.

  - **Out:** `{"run_id": ..., "digest": ..., "requested": [...], "ran": [...],
             "results": {analysis_name: {...}}, "created_at": ..., "force": false,
             "allow_billing": false, "status": "running", "error": null,
             "current": "rt60", "completed": 2, "total": 5}`
  - **Codes:** 200, 404
  - **Notes:** `status` is `queued`, `running`, `done` or `failed`. While the run
               is going, `current` names the analysis in progress and `completed`
               out of `total` is the progress. `ran` and `results` stay empty
               until the status is `done`. A failed run has a short message in
               `error` and the full detail is in the server log. A client polls
               until the status is `done` or `failed`.
               A run that was queued or running when the server restarted is marked
               `failed` at the next start, with the error "Server restarted before
               the run finished." Without this, a polling client would wait forever.
               Runs are stored in SQLite at `data/runs.db`.

### GET /v1/files/{digest}/runs
  Every run made for one file.

  - **Out:** a list of run records, in the same shape as `GET /v1/runs/{run_id}`,
             newest first.
  - **Codes:** 200, 404 no such digest
  - **Notes:** this replaces the earlier idea of `GET /v1/files/{digest}/results`
               (see Dropped). The list is indexed by digest in the database.

### GET /v1/files/{digest}/waveform
  A small outline of the audio, for drawing the waveform chart.

  - **In:** query value `points`, the number of slices. Default 4000, from 10 to 20000.
  - **Out:** `{"duration_s": 23.846, "sample_rate": 16000, "points": 4000,
             "min": [...], "max": [...]}`
             Slice `i` starts at `i * duration_s / points` seconds. `min` and `max`
             are the lowest and highest sample values in that slice, between -1 and 1.
  - **Codes:** 200, 404 no such digest, 422 `points` out of range or the file
               cannot be decoded
  - **Notes:** the outline is computed by `waveform.envelope`, which keeps the
               minimum and the maximum of each slice. Plain thinning of the samples
               would drop short peaks, and the clipped peaks are what the chart is
               there to show. The file is decoded on every request and nothing is
               cached yet.

### GET /v1/files/{digest}/audio
  The uploaded file, so a browser can play it in an `<audio>` element.

  - **Out:** the file bytes, with a media type guessed from the file name
             (e.g. `audio/mpeg`).
  - **Codes:** 200, 404 no such digest

### Dropped
  `GET /v1/files/{digest}/results` is not built. With run IDs, results are
  reachable at `/v1/runs/{run_id}`, and nobody has asked for "the latest run of
  this file" yet. A caller can take the first item of `GET /v1/files/{digest}/runs`.

### Deferred
  Routing verdicts get no endpoint. `routing.py` was removed from the repo after v1,
  and its thresholds now live in `thresholds.py`.

### To do
  **Local speech to text with Whisper.** Transcription currently needs a paid
  provider (AssemblyAI or ElevenLabs), so a person who clones the repo cannot use
  anything that depends on a transcript without API keys. A local Whisper model
  would remove that need.

  - `asr.py` already has a `provider` setting and one function per provider behind
    `transcribe()`, so a local provider is a third function.
  - Whisper gives word timestamps and word probabilities. `_confidence_from_logprob`
    already turns probabilities into the confidence scale the rest of the code uses.
  - Whisper does not tell speakers apart. The `speakers`, `overlap` and per-speaker
    pitch analyses, and the per-speaker timeline, all need speaker labels. There are
    two ways forward. One is to add a speaker separation step (e.g. pyannote or
    WhisperX). The other is to report those analyses as "not measured" when the
    provider has no speaker labels.
  - A local model costs no money but it does cost time, so the billing check should
    become a general check for expensive work. A run that would start a long local
    transcription should still need the caller to agree to it first.
  - `TranscribeConfig.cache_key` keeps existing cached transcripts valid when new
    settings are added, so the new provider should follow the same rule.
  - The run request may need a way to choose the provider, e.g. a `provider` field
    in the body of `POST /v1/files/{digest}/runs`.
