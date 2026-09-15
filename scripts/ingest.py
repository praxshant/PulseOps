"""CLI entry point for data ingestion."""

import argparse
from pathlib import Path

from data.validate import validate_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV file to validate")
    args = parser.parse_args()
    validate_csv(args.input)
    print(f"Validated {args.input}")


if __name__ == "__main__":
    main()
