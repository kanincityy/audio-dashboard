from fastapi.testclient import TestClient

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
