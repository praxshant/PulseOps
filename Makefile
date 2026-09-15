.PHONY: install test lint run-api train evaluate drift

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	python -m ruff check .

run-api:
	python -m uvicorn pulseops.api.main:app --reload

train:
	python scripts/train.py data/processed/sample.csv

evaluate:
	python scripts/evaluate.py data/processed/sample.csv artifacts/models/baseline.joblib

drift:
	python scripts/detect_drift.py data/processed/reference.csv data/processed/current.csv
