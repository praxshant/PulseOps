import pandas as pd

from features.build import build_forecasting_dataset


def _context_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    beds = pd.DataFrame(
        {
            "Org Code": ["ABC"],
            "date": pd.to_datetime(["2026-01-02"]),
            "G&A beds occupied": [90.0],
            "G&A beds available": [100.0],
            "G&A occupancy rate": [0.9],
            "bed_context_available": [1],
        }
    )
    ae = pd.DataFrame(
        {
            "Org Code": ["ABC"],
            "month": pd.to_datetime(["2025-12-01"]),
            "total_attendances": [1000.0],
            "total_emergency_admissions": [200.0],
            "total_over_4_hours": [50.0],
        }
    )
    return beds, ae


def _discharges(values: list[tuple[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=["date", "daily_discharges"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["Org Code"] = "ABC"
    frame["Org Name"] = "Example Trust"
    frame["target_next_day"] = frame["daily_discharges"].shift(-1)
    return frame[["Org Code", "Org Name", "date", "daily_discharges", "target_next_day"]]


def test_lag_uses_calendar_date_not_previous_row() -> None:
    discharges = _discharges(
        [
            ("2026-01-01", 10.0),
            ("2026-01-02", 20.0),
            ("2026-01-04", 40.0),
        ]
    )
    beds, ae = _context_tables()

    result = build_forecasting_dataset(discharges, beds, ae)

    missing_day_row = result.loc[result["date"].eq(pd.Timestamp("2026-01-04"))].iloc[0]
    assert pd.isna(missing_day_row["lag_1_discharge"])
    assert missing_day_row["lag_7_discharge"] != 20.0


def test_future_value_does_not_change_features_at_prediction_date() -> None:
    beds, ae = _context_tables()
    original = build_forecasting_dataset(
        _discharges(
            [
                ("2026-01-01", 10.0),
                ("2026-01-02", 20.0),
                ("2026-01-03", 30.0),
            ]
        ),
        beds,
        ae,
    )
    changed_future = build_forecasting_dataset(
        _discharges(
            [
                ("2026-01-01", 10.0),
                ("2026-01-02", 20.0),
                ("2026-01-03", 3000.0),
            ]
        ),
        beds,
        ae,
    )

    feature_columns = ["lag_1_discharge", "lag_7_discharge", "rolling_7d_discharge"]
    original_row = original.loc[original["date"].eq(pd.Timestamp("2026-01-02")), feature_columns].iloc[0]
    changed_row = changed_future.loc[changed_future["date"].eq(pd.Timestamp("2026-01-02")), feature_columns].iloc[0]
    pd.testing.assert_series_equal(original_row, changed_row)


def test_context_flags_distinguish_available_context() -> None:
    beds, ae = _context_tables()
    result = build_forecasting_dataset(
        _discharges([("2026-01-01", 10.0), ("2026-01-02", 20.0)]),
        beds,
        ae,
    )

    january_row = result.loc[result["date"].eq(pd.Timestamp("2026-01-01"))].iloc[0]
    assert january_row["bed_context_available"] == 0
    assert january_row["ae_context_available"] == 1
    assert january_row["previous_month_total_attendances"] == 1000.0
