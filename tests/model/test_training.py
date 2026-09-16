"""Tests for model training, quality gate, and registration control."""

from __future__ import annotations

import pandas as pd
import pytest

from models.quality_gate import passes_quality_gate
from models.train import FEATURE_SETS, _model


def test_feature_sets_are_numeric_candidate_columns() -> None:
    assert "lag_7_discharge" in FEATURE_SETS["calendar_lags_rolling"]
    assert "previous_month_total_attendances" in FEATURE_SETS["all_context"]


_ALL_MODEL_TYPES = ["ridge", "hist_gradient_boosting", "extra_trees", "random_forest"]


@pytest.mark.parametrize("model_type", _ALL_MODEL_TYPES)
def test_candidate_models_fit_small_frame(model_type: str) -> None:
    features = pd.DataFrame(
        {
            "day_of_week": [0, 1, 2],
            "is_weekend": [0, 0, 0],
            "lag_1_discharge": [1.0, 2.0, None],
        }
    )
    target = pd.Series([2.0, 3.0, 4.0])
    model = _model(model_type, 42)
    model.fit(features, target)
    assert len(model.predict(features)) == 3


def _make_metrics(candidate_mae: float, baseline_mae: float, candidate_wape: float) -> dict:
    return {
        "validation": {
            "candidate": {"mae": candidate_mae, "rmse": 0.0, "wape": candidate_wape},
            "previous_day": {"mae": baseline_mae, "rmse": 0.0, "wape": 0.30},
            "previous_weekday": {"mae": baseline_mae + 1, "rmse": 0.0, "wape": 0.31},
        }
    }


def test_quality_gate_passes_when_candidate_beats_baselines() -> None:
    metrics = _make_metrics(candidate_mae=10.0, baseline_mae=20.0, candidate_wape=0.13)
    passed, reason = passes_quality_gate(metrics)
    assert passed is True
    assert "Passed" in reason


def test_quality_gate_fails_when_candidate_loses_to_baseline() -> None:
    metrics = _make_metrics(candidate_mae=25.0, baseline_mae=20.0, candidate_wape=0.13)
    passed, reason = passes_quality_gate(metrics)
    assert passed is False
    assert "does not beat" in reason


def test_quality_gate_fails_when_wape_too_high() -> None:
    metrics = _make_metrics(candidate_mae=10.0, baseline_mae=20.0, candidate_wape=0.25)
    passed, reason = passes_quality_gate(metrics)
    assert passed is False
    assert "WAPE" in reason


def test_quality_gate_fails_when_split_missing() -> None:
    passed, reason = passes_quality_gate({})
    assert passed is False
    assert "Missing" in reason
