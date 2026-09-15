"""Build time-series features from preprocessed records."""

import pandas as pd


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features while preserving the target column."""
    result = frame.copy()
    timestamp = pd.to_datetime(result["timestamp"], utc=True)
    result["hour"] = timestamp.dt.hour
    result["day_of_week"] = timestamp.dt.dayofweek
    result["day_of_year"] = timestamp.dt.dayofyear
    return result
