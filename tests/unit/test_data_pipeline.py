import pandas as pd

from data.normalize import normalize_discharges
from data.validate import validate_normalized_discharges


def test_normalization_resolves_overlap_and_builds_next_day_target() -> None:
    frame = pd.DataFrame(
        {
            "Period": ["01/05/2026", "01/05/2026", "02/05/2026"],
            "Level": ["Provider", "Provider", "Provider"],
            "Org Code": ["ABC", "ABC", "ABC"],
            "Org Name": ["Example Trust"] * 3,
            "Metric": ["Number of patients discharged"] * 3,
            "Metric Type": ["Daily metric"] * 3,
            "Metric Group": ["Discharges"] * 3,
            "Value": ["-", 10, 12],
        }
    )

    result = normalize_discharges(frame)
    report = validate_normalized_discharges(result)

    assert report.passed
    assert result.loc[0, "daily_discharges"] == 10
    assert result.loc[0, "target_next_day"] == 12
    assert len(result) == 2


def test_validation_rejects_missing_columns() -> None:
    report = validate_normalized_discharges(pd.DataFrame({"Org Code": ["ABC"]}))
    assert not report.passed
    assert any("Missing canonical columns" in error for error in report.errors)
