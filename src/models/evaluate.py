"""Model evaluation metrics."""

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_predictions(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Return stable regression metrics for reports and MLflow logging."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
    }
