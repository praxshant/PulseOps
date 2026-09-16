.PHONY: install test lint run-api validate-data build-features train-model benchmark-models up down retrain drift

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

benchmark-models:
	python -m models.benchmark

up:
	docker compose -f deployment/docker-compose.yml up -d

down:
	docker compose -f deployment/docker-compose.yml down

retrain:
	python -m scripts.retrain

drift:
	python -m monitoring.drift
