from fastapi.testclient import TestClient

from audio_dashboard.api import files
from audio_dashboard.api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyses_lists_snr():
    response = client.get("/v1/analyses")
    assert response.status_code == 200
    names = [analysis["name"] for analysis in response.json()["analyses"]]
    assert "snr" in names


def test_upload_new_file_returns_201(tmp_path, monkeypatch):
    monkeypatch.setattr(files, "UPLOAD_DIR", tmp_path)
    content = b"fake audio bytes"

    response = client.post("/v1/files", files={"file": ("test.wav", content)})

    assert response.status_code == 201
    data = response.json()
    assert data["size_bytes"] == len(content)
    assert data["filename"] == "test.wav"
