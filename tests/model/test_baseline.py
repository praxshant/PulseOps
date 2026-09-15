import pandas as pd

from models.evaluate import evaluate_predictions


def test_evaluation_returns_mae_and_rmse() -> None:
    metrics = evaluate_predictions(pd.Series([1.0, 3.0]), pd.Series([2.0, 2.0]))
    assert metrics == {"mae": 1.0, "rmse": 1.0, "wape": 0.5, "rows": 2}
