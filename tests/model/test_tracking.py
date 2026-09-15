import json

from models.tracking import RunTracker


def test_tracker_writes_ordered_timestamped_events(tmp_path) -> None:
    tracker = RunTracker(tmp_path)
    tracker.log("dataset_loaded", rows=10)
    tracker.finish(status="success")

    records = [json.loads(line) for line in tracker.events_path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == [
        "run_created",
        "dataset_loaded",
        "run_completed",
    ]
    assert all(record["run_id"] == tracker.run_id for record in records)
    assert all("timestamp" in record for record in records)
    assert len(tracker.run_id) == 32
