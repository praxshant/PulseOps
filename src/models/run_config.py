"""Configuration and lineage metadata for reproducible training runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RunConfig:
    """Stable configuration recorded with every training run."""

    experiment_name: str = "pulseops-discharge-forecast"
    model_type: str = "hist_gradient_boosting"
    feature_set: str = "calendar_lags_rolling"
    random_seed: int = 42

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
