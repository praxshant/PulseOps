import pandas as pd
import pytest
from monitoring.drift import population_stability_index, detect_drift, mean_shift, DriftStatus


def test_psi_identical_distributions() -> None:
    ref = pd.Series([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    cur = pd.Series([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    psi = population_stability_index(ref, cur)
    assert psi < 0.001


def test_psi_different_distributions() -> None:
    ref = pd.Series([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    cur = pd.Series([6, 7, 8, 9, 10, 6, 7, 8, 9, 10])
    psi = population_stability_index(ref, cur)
    assert psi > 0.5


def test_mean_shift_rejects_empty_series() -> None:
    ref = pd.Series([1, 2, 3])
    cur = pd.Series([], dtype=float)
    with pytest.raises(ValueError):
        mean_shift(ref, cur)


def test_detect_drift_thresholds() -> None:
    assert detect_drift(0.05) == DriftStatus.STABLE
    assert detect_drift(0.15) == DriftStatus.WARNING
    assert detect_drift(0.25) == DriftStatus.ALERT
