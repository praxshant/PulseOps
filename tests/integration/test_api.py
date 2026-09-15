from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_endpoint() -> None:
    response = client.post("/predict", json={"value": 7.5})
    assert response.status_code == 200
    assert response.json() == {"prediction": 7.5}
