from pathlib import Path
from unittest.mock import patch
from models.registry import get_production_champion
from models.quality_gate import is_improvement
from models.retrain import retrain


def test_get_production_champion_returns_none_when_empty(tmp_path) -> None:  # noqa: ANN001
    assert get_production_champion(tmp_path) is None


def test_is_improvement() -> None:
    assert is_improvement(9.0, 10.0, 0.01) is True
    # Exactly 1%
    assert is_improvement(9.9, 10.0, 0.01) is True
    # Less than 1%
    assert is_improvement(9.95, 10.0, 0.01) is False
    # Worse
    assert is_improvement(11.0, 10.0, 0.01) is False


@patch("models.retrain.run_training")
@patch("models.retrain.get_production_champion")
@patch("models.retrain.get_champion_validation_mae")
def test_retrain_does_not_replace_when_gate_fails(mock_get_mae, mock_get_champ, mock_run) -> None:  # noqa: ANN001
    mock_run.return_value = {
        "run_id": "failed123",
        "quality_gate": {"passed": False, "reason": "Failed WAPE"},
    }
    res = retrain(Path("dummy"), register_if_better=False)
    assert res.gate_passed is False
    assert res.champion_replaced is False


@patch("models.retrain.run_training")
@patch("models.retrain.get_production_champion")
@patch("models.retrain.get_champion_validation_mae")
def test_retrain_does_not_replace_when_no_improvement(mock_get_mae, mock_get_champ, mock_run) -> None:  # noqa: ANN001
    mock_run.return_value = {
        "run_id": "passed123",
        "quality_gate": {"passed": True},
        "metrics": {"validation": {"candidate": {"mae": 9.95}}}
    }
    mock_get_champ.return_value = {"run_id": "champ123"}
    mock_get_mae.return_value = 10.0
    
    res = retrain(Path("dummy"), register_if_better=False)
    assert res.gate_passed is True
    assert res.champion_replaced is False
