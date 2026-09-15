"""Train and persist a baseline model."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyRegressor


def train_baseline(frame: pd.DataFrame, output_path: Path) -> Path:
    """Fit a mean baseline using the value column and persist it."""
    model = DummyRegressor(strategy="mean")
    model.fit(frame[["value"]], frame["value"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    return output_path
