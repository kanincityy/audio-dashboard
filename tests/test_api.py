import io
from datetime import datetime, timezone

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from audio_dashboard import asr, cache
from audio_dashboard.api import files, runs, store
from audio_dashboard.api.main import app
from audio_dashboard.api.schemas import RunRecord

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Point every place the API writes to at this test's own temp folder.

    autouse runs it before every test in this file, so no test can forget a
    patch and leak into the real uploads/, cache or database. runs.py has its
    own copy of the UPLOAD_DIR name (it imported it), so it is patched too.
    """
    monkeypatch.setattr(files, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(runs, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "runs.db")
    store.init_db()


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyses_lists_snr():
    response = client.get("/v1/analyses")
    assert response.status_code == 200
    names = [analysis["name"] for analysis in response.json()["analyses"]]
    assert "snr" in names


def test_upload_new_file_returns_201():
    content = b"fake audio bytes"

    response = client.post("/v1/files", files={"file": ("test.wav", content)})

    assert response.status_code == 201
    data = response.json()
    assert data["size_bytes"] == len(content)
    assert data["filename"] == "test.wav"


def test_upload_same_file_twice_returns_200():
    content = b"fake audio bytes"

    first_response = client.post("/v1/files", files={"file": ("test.wav", content)})
    second_response = client.post("/v1/files", files={"file": ("test.wav", content)})

    assert first_response.status_code == 201
    assert second_response.status_code == 200


def test_upload_text_returns_415():
    content = b"fake audio bytes"

    response = client.post("/v1/files", files={"file": ("test.txt", content)})

    assert response.status_code == 415


def test_run_unknown_digest_returns_404():

    response = client.post("/v1/files/abc/runs", json={"analyses": ["snr"]})

    assert response.status_code == 404


def test_run_unknown_analysis_returns_422():
    upload = client.post("/v1/files", files={"file": ("test.wav", b"fake audio bytes")})
    digest = upload.json()["digest"]

    response = client.post(f"/v1/files/{digest}/runs", json={"analyses": ["snrr"]})

    assert response.status_code == 422
    assert "snrr" in response.json()["detail"]


def test_run_that_would_bill_returns_403(monkeypatch):
    monkeypatch.setattr(asr, "is_cached", lambda path, config: False)

    # Safety net: if the 403 check ever broke, the run would reach the paid
    # API. Fail loudly instead of billing.
    def no_paid_calls(*args, **kwargs):
        raise AssertionError("test tried to call the paid transcription API")

    monkeypatch.setattr(asr, "transcribe", no_paid_calls)
    upload = client.post("/v1/files", files={"file": ("test.wav", b"fake audio bytes")})
    digest = upload.json()["digest"]

    response = client.post(
        f"/v1/files/{digest}/runs", json={"analyses": ["transcript"]}
    )

    assert response.status_code == 403


def _one_second_tone() -> bytes:
    """A real, decodable WAV: 1 s of 440 Hz at 16 kHz."""
    sr = 16_000
    t = np.arange(sr) / sr
    buffer = io.BytesIO()
    sf.write(buffer, 0.5 * np.sin(2 * np.pi * 440 * t), sr, format="WAV")
    return buffer.getvalue()


def test_run_then_fetch_it():
    upload = client.post("/v1/files", files={"file": ("tone.wav", _one_second_tone())})
    digest = upload.json()["digest"]

    created = client.post(f"/v1/files/{digest}/runs", json={"analyses": ["levels"]})
    assert created.status_code == 202
    assert created.json()["ran"] == ["levels"]

    fetched = client.get(created.headers["location"])
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "done"
    # registry.run records a failing analysis as {"error": ...} instead of
    # raising, so check for a real value, not just the key.
    assert "peak_dbfs" in fetched.json()["results"]["levels"]


def test_list_runs_for_a_file():
    upload = client.post("/v1/files", files={"file": ("tone.wav", _one_second_tone())})
    digest = upload.json()["digest"]

    before = client.get(f"/v1/files/{digest}/runs")
    assert before.status_code == 200
    assert before.json() == []

    created = client.post(f"/v1/files/{digest}/runs", json={"analyses": ["levels"]})

    after = client.get(f"/v1/files/{digest}/runs")
    assert [run["run_id"] for run in after.json()] == [created.json()["run_id"]]


def test_list_runs_for_unknown_digest_returns_404():
    response = client.get("/v1/files/nope/runs")

    assert response.status_code == 404


def test_unknown_run_id_returns_404():
    response = client.get("/v1/runs/nope")

    assert response.status_code == 404


def test_broken_audio_run_is_marked_failed():
    upload = client.post("/v1/files", files={"file": ("test.wav", b"fake audio bytes")})
    digest = upload.json()["digest"]

    response = client.post(f"/v1/files/{digest}/runs", json={"analyses": ["levels"]})
    fetched = client.get(response.headers["location"])
    assert fetched.json()["status"] == "failed"


def test_runs_cut_off_by_a_restart_are_marked_failed():
    def saved(run_id, status):
        store.save_run(
            RunRecord(
                run_id=run_id,
                digest="abc",
                created_at=datetime.now(timezone.utc),
                requested=["snr"],
                force=False,
                allow_billing=False,
                status=status,
            )
        )

    saved("was-running", "running")
    saved("was-queued", "queued")
    saved("finished", "done")

    assert store.fail_interrupted_runs() == 2
    assert store.get_run("was-running").status == "failed"
    assert store.get_run("was-queued").status == "failed"
    assert store.get_run("finished").status == "done"
