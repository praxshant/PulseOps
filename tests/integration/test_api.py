"""Integration tests for the PulseOps FastAPI endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from api.main import REQUIRED_FEATURES, app

client = TestClient(app)

_VALID_FEATURES = {f: 1.0 for f in REQUIRED_FEATURES}
_MOCK_MODEL_TUPLE = (MagicMock(predict=lambda x: np.array([123.4])), "abc12345", "abc12345" * 2)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_endpoint_returns_prometheus_text() -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "pulseops_predictions_total" in response.text


def test_predict_returns_prediction_with_lineage(tmp_path) -> None:  # noqa: ANN001
    with patch("api.main._load_model", return_value=_MOCK_MODEL_TUPLE):
        with patch("api.main._INFERENCE_LOG_DIR", tmp_path):
            response = client.post(
                "/predict",
                json={
                    "org_code": "RJ1",
                    "forecast_date": "2026-09-16",
                    "features": _VALID_FEATURES,
                },
            )
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert body["org_code"] == "RJ1"
    assert body["forecast_date"] == "2026-09-16"
    assert "model_version" in body
    assert "run_id" in body
    assert "prediction_id" in body
    assert "latency_ms" in body


def test_predict_returns_422_when_features_missing() -> None:
    with patch("api.main._load_model", return_value=_MOCK_MODEL_TUPLE):
        response = client.post(
            "/predict",
            json={
                "org_code": "RJ1",
                "forecast_date": "2026-09-16",
                "features": {"day_of_week": 1.0},
            },
        )
    assert response.status_code == 422


def test_predict_returns_503_when_no_model_available(tmp_path) -> None:  # noqa: ANN001
    with patch("api.main._load_model", side_effect=RuntimeError("no model")):
        with patch("api.main._INFERENCE_LOG_DIR", tmp_path):
            response = client.post(
                "/predict",
                json={
                    "org_code": "RJ1",
                    "forecast_date": "2026-09-16",
                    "features": _VALID_FEATURES,
                },
            )
    assert response.status_code == 503
