import pandas as pd

from models.baselines import previous_day_baseline, previous_weekday_baseline
from models.metrics import regression_metrics


def test_baselines_and_metrics() -> None:
    frame = pd.DataFrame(
        {
            "Org Code": ["A", "A"],
            "date": pd.to_datetime(["2026-04-01", "2026-04-08"]),
            "daily_discharges": [10.0, 12.0],
        }
    )
    assert previous_day_baseline(frame).tolist() == [10.0, 12.0]
    weekday_baseline = previous_weekday_baseline(frame)
    assert pd.isna(weekday_baseline.iloc[0])
    assert weekday_baseline.iloc[1] == 10.0
    assert regression_metrics(pd.Series([1.0, 3.0]), pd.Series([2.0, 2.0])) == {
        "mae": 1.0,
        "rmse": 1.0,
        "wape": 0.5,
        "rows": 2,
    }
