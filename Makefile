.PHONY: install test lint run-api validate-data build-features train-model

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	python -m ruff check .

run-api:
	python -m uvicorn api.main:app --reload

validate-data:
	python -m data.validate

build-features:
	python -m features.build

train-model:
	python -m models.train
