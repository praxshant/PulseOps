import pandas as pd
import pytest

from monitoring.drift import mean_shift


def test_mean_shift_is_absolute() -> None:
    assert mean_shift(pd.Series([1, 2]), pd.Series([4, 5])) == 3.0


def test_mean_shift_rejects_empty_series() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        mean_shift(pd.Series(dtype=float), pd.Series([1.0]))
