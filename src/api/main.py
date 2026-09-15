"""FastAPI application for model health and prediction."""

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

app = FastAPI(title="PulseOps API", version="0.1.0")


class PredictionRequest(BaseModel):
    value: float = Field(..., description="Observed value used by the baseline model")


class PredictionResponse(BaseModel):
    prediction: float


@app.get("/health")
def health() -> dict[str, str]:
    """Report service health for container orchestration."""
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics for scraping."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Return the baseline prediction for one observation."""
    return PredictionResponse(prediction=request.value)
