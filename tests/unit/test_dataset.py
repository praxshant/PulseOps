import pandas as pd
import pytest

from data.dataset import dataset_manifest, validate_forecasting_dataset


def test_dataset_contract_rejects_duplicate_keys() -> None:
    frame = pd.DataFrame(
        {
            "Org Code": ["A", "A"],
            "date": pd.to_datetime(["2026-04-01", "2026-04-01"]),
            "daily_discharges": [1.0, 2.0],
            "target_next_day": [2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_forecasting_dataset(frame)


def test_dataset_manifest_has_schema_and_date_lineage(tmp_path) -> None:
    path = tmp_path / "dataset.parquet"
    frame = pd.DataFrame(
        {
            "Org Code": ["A"],
            "date": pd.to_datetime(["2026-04-01"]),
            "daily_discharges": [1.0],
            "target_next_day": [2.0],
        }
    )
    frame.to_parquet(path, index=False)
    validate_forecasting_dataset(frame)
    manifest = dataset_manifest(frame, path)
    assert len(manifest["dataset_hash"]) == 64
    assert manifest["date_min"] == "2026-04-01"
