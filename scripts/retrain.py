"""CLI entrypoint for retraining pipeline."""

import argparse
from pathlib import Path

from models.retrain import retrain


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PulseOps automated retraining pipeline")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/forecasting_dataset.parquet"),
    )
    args = parser.parse_args()
    
    result = retrain(dataset_path=args.dataset)
    print(f"Retrain Result: {result.model_dump_json(indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
