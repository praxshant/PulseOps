import pandas as pd

from models.train import FEATURE_SETS, _model


def test_feature_sets_are_numeric_candidate_columns() -> None:
    assert "lag_7_discharge" in FEATURE_SETS["calendar_lags_rolling"]
    assert "previous_month_total_attendances" in FEATURE_SETS["all_context"]


def test_candidate_models_fit_small_frame() -> None:
    features = pd.DataFrame(
        {
            "day_of_week": [0, 1, 2],
            "is_weekend": [0, 0, 0],
            "lag_1_discharge": [1.0, 2.0, None],
        }
    )
    target = pd.Series([2.0, 3.0, 4.0])
    for model_type in ["ridge", "hist_gradient_boosting"]:
        model = _model(model_type, 42)
        model.fit(features, target)
        assert len(model.predict(features)) == 3
