from monitoring.data_quality import check_feature_completeness, check_feature_ranges, check_data_staleness


def test_feature_completeness() -> None:
    features = {"a": 1, "b": 2}
    req = ["a", "b", "c"]
    missing = check_feature_completeness(features, req)
    assert missing == ["c"]
    
    missing_none = check_feature_completeness({"a": 1, "b": None, "c": 3}, req)
    assert missing_none == ["b"]


def test_feature_ranges() -> None:
    features = {"lag_1_discharge": -5, "day_of_week": 1, "previous_month_total_attendances": -100}
    errors = check_feature_ranges(features)
    assert len(errors) == 2
    assert "lag_1_discharge cannot be negative" in errors
    assert "previous_month_total_attendances cannot be negative" in errors


def test_data_staleness() -> None:
    # 2010 is extremely stale
    assert check_data_staleness("2010-01-01") is True
    # If unparseable, doesn't crash, returns False
    assert check_data_staleness("not-a-date") is False
