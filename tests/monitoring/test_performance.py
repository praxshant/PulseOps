import json

import pandas as pd

from monitoring.performance import compute_performance_report


def test_performance_computation(tmp_path) -> None:  # noqa: ANN001
    inference_log = tmp_path / "inference.jsonl"
    actuals = tmp_path / "actuals.parquet"
    
    # Write mock inference log
    records = [
        {"forecast_date": "2026-09-01", "org_code": "RJ1", "prediction": 100.0},
        {"forecast_date": "2026-09-02", "org_code": "RJ1", "prediction": 120.0},
    ]
    with open(inference_log, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    # Write mock actuals
    df = pd.DataFrame([
        {"date": "2026-09-01", "Org Code": "RJ1", "target_next_day": 110.0},
        {"date": "2026-09-02", "Org Code": "RJ1", "target_next_day": 120.0},
    ])
    df.to_parquet(actuals)
    
    report = compute_performance_report(inference_log, actuals, window_days=7)
    assert report.n_predictions == 2
    assert report.n_actuals_matched == 2
    assert report.mae == 5.0  # (|100-110| + |120-120|) / 2


def test_performance_returns_none_when_empty(tmp_path) -> None:  # noqa: ANN001
    inference_log = tmp_path / "inference.jsonl"
    actuals = tmp_path / "actuals.parquet"
    # Files don't exist
    report = compute_performance_report(inference_log, actuals)
    assert report.mae is None
    assert report.n_predictions == 0
