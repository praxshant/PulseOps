# PulseOps

PulseOps is a reproducible production ML service for hospital operational forecasting.
The project uses notebooks for exploratory data analysis and Python modules for
reproducible data processing, feature engineering, model training, evaluation,
serving, and monitoring.

## Architecture

```text
Raw NHS data -> EDA / data understanding -> validation -> normalization
                                                        |
                                                        v
feature engineering -> training -> MLflow -> evaluation -> model registry
                                                        |
                                                        v
FastAPI -> Docker -> cloud -> monitoring -> drift detection -> retraining
```

## Layout

- `src/data`: ingestion, validation, normalization, and preprocessing.
- `src/features`: deterministic feature engineering.
- `src/models`: training, evaluation, and prediction.
- `src/api`: FastAPI serving boundary.
- `src/monitoring`: metrics and drift checks.
- `scripts`: reproducible command-line workflows.
- `tests`: unit, integration, and model tests.
- `deployment`: Docker Compose, Dockerfile, and Prometheus configuration.

## Local setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install -e ".[dev]"
python -m pytest
python -m uvicorn api.main:app --reload
```

The API is available at `http://localhost:8000`; interactive docs are at `/docs`.

## Pipeline commands

```bash
$env:PYTHONPATH="src"
python -m data.validate
```

The current data foundation reads the immutable NHS files in `data/raw/` and
writes reproducible normalized tables to `data/interim/`. Raw and derived NHS
datasets are intentionally excluded from source control and can be regenerated
from the documented acquisition and normalization workflow.

Before using real NHS or operational data, record dataset provenance, licensing, schema, and retrieval date in `data/README.md`. Keep raw data out of source control.

## Engineering decisions

- Scripts are thin entry points; reusable behavior lives under the flat `src` packages.
- The notebook is for EDA and decision records; production logic stays in Python modules.
- Validation happens before normalization and feature engineering.
- A baseline model provides a measurable floor before more advanced models are introduced.
- Tests cover data contracts, API health, model metrics, and drift behavior.
- MLflow settings are centralized in `src/config.py` so local and deployed runs can use the same code path.
