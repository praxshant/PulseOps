"""Lightweight distribution drift checks."""

import pandas as pd


def mean_shift(reference: pd.Series, current: pd.Series) -> float:
    """Return the absolute difference between reference and current means."""
    if reference.empty or current.empty:
        raise ValueError("Both reference and current series must be non-empty")
    return float(abs(reference.mean() - current.mean()))
