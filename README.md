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

- `src/data`: ingestion, validation, normalization, dataset contracts, and splits.
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
python -m features.build
python -m models.train
```

The current data foundation reads the immutable NHS files in `data/raw/` and
writes reproducible normalized tables to `data/interim/`. Raw and derived NHS
datasets are intentionally excluded from source control and can be regenerated
from the documented acquisition, normalization, and feature-building workflow.
The feature build writes `data/processed/forecasting_dataset.parquet` locally.
Training records split boundaries, feature sets, dataset and schema hashes, Git
metadata, environment versions, predictions, and MAE/RMSE/WAPE metrics under
the ignored `artifacts/runs/` directory.

The current temporal evaluation uses April-June for training, July for
validation, and August for test. Baselines are previous-day and previous-week
same-weekday discharge counts. The first candidate model is
`HistGradientBoostingRegressor`; model metrics are published only after a
training run has been completed.

## First training result

Run `20260915T141039Z` trained `HistGradientBoostingRegressor` with the
`calendar_lags_rolling` feature set and random seed `42`.

| Split | Model | MAE | RMSE | WAPE | Rows |
|---|---|---:|---:|---:|---:|
| Validation | Candidate | 10.37 | 14.76 | 12.92% | 3,627 |
| Validation | Previous day | 20.68 | 29.09 | 25.75% | 3,627 |
| Validation | Previous weekday | 22.16 | 30.71 | 27.65% | 2,808 |
| Test | Candidate | 10.70 | 15.78 | 13.97% | 3,510 |
| Test | Previous day | 20.58 | 29.15 | 26.87% | 3,510 |
| Test | Previous weekday | 21.43 | 30.98 | 28.09% | 2,691 |

The candidate beats both naive baselines on the held-out test split. This is a
baseline engineering result, not a claim of annual seasonality or production
readiness. The run used dataset SHA-256
`9eed502b65038be2eda91527714fa6d39191b6f859c285e9c8058b9ca1acaa00`, schema
hash `e904102d7b2c6a96c53ee152d12f2baf78b3152aa5662895fed5a5f79a88ebc9`, and
Git commit `ab515a3202cda75efbd168b4ac8454c684bfb7fb`.

Before using real NHS or operational data, record dataset provenance, licensing, schema, and retrieval date in `data/README.md`. Keep raw data out of source control.

## Engineering decisions

- Scripts are thin entry points; reusable behavior lives under the flat `src` packages.
- The notebook is for EDA and decision records; production logic stays in Python modules.
- Validation happens before normalization and feature engineering.
- A baseline model provides a measurable floor before more advanced models are introduced.
- Tests cover data contracts, API health, model metrics, and drift behavior.
- MLflow settings are centralized in `src/config.py` so local and deployed runs can use the same code path.
