"""Prometheus metrics exposed by the service."""

from prometheus_client import Counter, Histogram

PREDICTION_COUNT = Counter("pulseops_predictions_total", "Total predictions served")
PREDICTION_LATENCY = Histogram("pulseops_prediction_latency_seconds", "Prediction latency")
