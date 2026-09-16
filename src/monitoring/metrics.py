"""Prometheus metrics exposed by the service."""

from prometheus_client import Counter, Gauge, Histogram

PREDICTION_COUNT = Counter(
    "pulseops_predictions_total",
    "Total number of predictions served",
    ["status"],
)

PREDICTION_LATENCY = Histogram(
    "pulseops_prediction_latency_seconds",
    "Prediction latency in seconds",
)

DRIFT_STATUS = Gauge(
    "pulseops_drift_status",
    "Drift status (0=STABLE, 1=WARNING, 2=ALERT)",
)

ROLLING_MAE = Gauge(
    "pulseops_rolling_mae",
    "Rolling Mean Absolute Error",
)

ROLLING_WAPE = Gauge(
    "pulseops_rolling_wape",
    "Rolling Weighted Absolute Percentage Error",
)

DATA_QUALITY_FAILURES = Counter(
    "pulseops_data_quality_failures_total",
    "Total number of data quality check failures",
)
