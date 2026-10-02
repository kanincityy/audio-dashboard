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
  - **Out:** `{"digest": "a1b2c3d4e5f60718", "bytes": 5242880, "filename": "call.mp3"}`
             The digest is the ID every later call uses.
  - **Codes:** 201 new, 200 repeat, 413, 415, 400
  - **Notes:** digest from `cache.file_digest`; bytes stored in `uploads/`

  ### POST /v1/files/{digest}/runs
  - **In:** `{"analyses": ["snr", "vad"], "force": false, "allow_billing": false}`
             `force` and `allow_billing` both default to false: the safe direction.
             `allow_billing` false means a run that would transcribe an uncached file
             is refused before anything runs, rather than billing by surprise.
  - **Out:** {"run_id": "7f3c...", "ran": ["vad", "snr"], "results_url": "/v1/runs/7f3c..."}
             plus a Location header with the same URL. The results themselves are NOT here:
             the caller follows the pointer (decision: run IDs, so a run is addressable and
             can later become a background job and a row in a runs table).
  - **Codes:** 201 created / 404 no such digest / 422 analysis name is not in the registry
               403 the selection would bill (uncached `transcript`) and allow_billing is false.
               Body names the analysis, e.g. {"detail": "transcript would be billed for this
               file; set allow_billing to run it"}. Checked with registry.resolve plus
               asr.is_cached BEFORE any analysis runs.
  - **Notes:** calls `registry.run`; `resolve` may pull in extra analyses, so the
               response should say which ones actually ran

  ### GET /v1/runs/{run_id}
  - **Out:** {"run_id": ..., "digest": ..., "ran": [...], "results": {analysis_name: {...}}}
  - **Codes:** 404, 200

  ### GET /v1/files/{digest}/results
  DROPPED for now. With run IDs, results are reachable at /v1/runs/{run_id}, and nobody
  has asked for "the latest run of this file" yet. Add it when a caller needs it.

  ### Deferred
  Routing verdicts get no endpoint. `routing.py` was removed from the repo after v1;
  its thresholds now live in `thresholds.py`.
