import pandas as pd
import pytest

from models.split import TemporalSplitConfig, chronological_split


def test_chronological_split_has_expected_date_boundaries() -> None:
    frame = pd.DataFrame(
        {
            "Org Code": ["A"] * 3,
            "date": pd.to_datetime(["2026-06-30", "2026-07-01", "2026-08-01"]),
        }
    )
    splits = chronological_split(frame, TemporalSplitConfig())
    assert splits["train"]["date"].max() == pd.Timestamp("2026-06-30")
    assert splits["validation"]["date"].min() == pd.Timestamp("2026-07-01")
    assert splits["test"]["date"].min() == pd.Timestamp("2026-08-01")


def test_chronological_split_rejects_empty_partition() -> None:
    frame = pd.DataFrame({"Org Code": ["A"], "date": pd.to_datetime(["2026-04-01"])})
    with pytest.raises(ValueError, match="empty"):
        chronological_split(frame, TemporalSplitConfig())
