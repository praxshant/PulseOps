from pathlib import Path

from models.benchmark import MODEL_TYPES
from models.train import _model


def test_benchmark_algorithms_are_supported() -> None:
    assert MODEL_TYPES == ("ridge", "hist_gradient_boosting", "extra_trees", "random_forest")
    for model_type in MODEL_TYPES:
        assert _model(model_type, 42) is not None


def test_benchmark_output_path_is_local_artifact() -> None:
    assert Path("artifacts/model_benchmark.json").parts[0] == "artifacts"
