"""FastAPI application for real model serving with inference observability."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from api.inference_log import log_inference_event
from monitoring.metrics import (
    PREDICTION_COUNT,
    PREDICTION_LATENCY,
    DATA_QUALITY_FAILURES,
)
from monitoring.data_quality import check_feature_completeness, check_feature_ranges, check_data_staleness

app = FastAPI(title="PulseOps API", version="0.3.0")

# --------------------------------------------------------------------------- #
# Model state                                                                   #
# --------------------------------------------------------------------------- #
_MODEL_CACHE: dict[str, Any] = {}
_INFERENCE_LOG_DIR = Path("artifacts/inference")
_ACTUALS_PATH = Path("data/processed/forecasting_dataset.parquet")


def _load_model() -> tuple[Any, str, str]:
    """Load the latest registered model from disk or return cached instance."""
    if "model" in _MODEL_CACHE:
        return _MODEL_CACHE["model"], _MODEL_CACHE["version"], _MODEL_CACHE["run_id"]

    import glob
    import json

    runs_root = Path("artifacts/runs")
    run_jsons = sorted(glob.glob(str(runs_root / "*/run.json")), reverse=True)

    for run_json_path in run_jsons:
        meta = json.loads(Path(run_json_path).read_text(encoding="utf-8"))
        gate = meta.get("quality_gate", {})
        if gate.get("passed") is True:
            model_path = Path(meta["model_artifact"])
            if model_path.exists():
                model = joblib.load(model_path)
                run_id = meta["run_id"]
                version = run_id[:8]
                _MODEL_CACHE.update({"model": model, "version": version, "run_id": run_id})
                return model, version, run_id

    raise RuntimeError(
        "No quality-gate-passing model found in artifacts/runs/. "
        "Run python -m models.train first."
    )


# --------------------------------------------------------------------------- #
# Feature columns (must match training)                                         #
# --------------------------------------------------------------------------- #
REQUIRED_FEATURES = [
    "day_of_week",
    "is_weekend",
    "lag_1_discharge",
    "lag_7_discharge",
    "lag_14_discharge",
    "lag_28_discharge",
    "rolling_7d_discharge",
    "rolling_14d_discharge",
    "rolling_28d_discharge",
    "previous_month_total_attendances",
    "previous_month_emergency_admissions",
    "previous_month_over_4_hours",
    "ae_context_available",
    "occupied_beds",
    "available_beds",
    "occupancy_rate",
    "bed_context_available",
]


# --------------------------------------------------------------------------- #
# Request / response schemas                                                    #
# --------------------------------------------------------------------------- #
class PredictionRequest(BaseModel):
    org_code: str = Field(..., description="NHS organisation code")
    forecast_date: str = Field(..., description="Date to forecast (YYYY-MM-DD)")
    features: dict[str, float] = Field(
        ..., description="Feature values keyed by feature name"
    )


class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Predicted next-day discharge count")
    org_code: str
    forecast_date: str
    model_version: str
    run_id: str
    prediction_id: str
    latency_ms: float


# --------------------------------------------------------------------------- #
# Endpoints                                                                     #
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict[str, str]:
    """Report service health for container orchestration."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics for scraping."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/drift")
def drift() -> dict[str, Any]:
    """Return latest drift report (mocked implementation for API)."""
    # In a full implementation, this would read from artifacts/monitoring/drift_report.json
    return {"status": "not_implemented_in_api"}


@app.get("/performance")
def performance() -> dict[str, Any]:
    """Compute and return rolling MAE/WAPE from inference log."""
    from monitoring.performance import compute_performance_report
    from monitoring.metrics import ROLLING_MAE, ROLLING_WAPE
    
    report = compute_performance_report(_INFERENCE_LOG_DIR / "inference.jsonl", _ACTUALS_PATH)
    if report.mae is not None:
        ROLLING_MAE.set(report.mae)
    if report.wape is not None:
        ROLLING_WAPE.set(report.wape)
        
    return report.model_dump()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Load the production model and return a real prediction."""
    t0 = time.perf_counter()

    # 1. Data Quality Checks
    missing = check_feature_completeness(request.features, REQUIRED_FEATURES)
    range_errors = check_feature_ranges(request.features)
    
    if missing or range_errors:
        DATA_QUALITY_FAILURES.inc()
        PREDICTION_COUNT.labels(status="error").inc()
        raise HTTPException(
            status_code=422,
            detail={"missing": missing, "range_errors": range_errors},
        )
        
    is_stale = check_data_staleness(request.forecast_date)

    try:
        model, version, run_id = _load_model()
    except RuntimeError as exc:
        PREDICTION_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    feature_row = pd.DataFrame([{f: request.features[f] for f in REQUIRED_FEATURES}])
    prediction_value = float(model.predict(feature_row)[0])
    latency_ms = (time.perf_counter() - t0) * 1000

    # Prometheus update (latency takes seconds, we have ms)
    PREDICTION_LATENCY.observe(latency_ms / 1000.0)
    PREDICTION_COUNT.labels(status="ok").inc()

    record = log_inference_event(
        log_dir=_INFERENCE_LOG_DIR,
        org_code=request.org_code,
        forecast_date=request.forecast_date,
        prediction=prediction_value,
        model_version=version,
        run_id=run_id,
        latency_ms=latency_ms,
        stale_data=is_stale,
    )

    return PredictionResponse(
        prediction=prediction_value,
        org_code=request.org_code,
        forecast_date=request.forecast_date,
        model_version=version,
        run_id=run_id,
        prediction_id=record["prediction_id"],
        latency_ms=round(latency_ms, 3),
    )
