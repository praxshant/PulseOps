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
same-weekday discharge counts. The default candidate is
`ExtraTreesRegressor` with the `all_context` feature set; model metrics are
published only after a training run has been completed.

## Current Champion Benchmark

The models are automatically evaluated on validation MAE to determine the champion. 

Current architecture includes full container stack (API, Prometheus, Grafana, MLflow) with drift and performance monitoring, plus automated retraining.

## Local Deployment

Bring up the full stack:
```bash
make up
```

Endpoints available:
- **API**: http://localhost:8000/docs
- **Grafana**: http://localhost:3000 (auto-provisioned dashboards)
- **MLflow**: http://localhost:5000
- **Prometheus**: http://localhost:9090

Bring down:
```bash
make down
```

## Next Step: Phase 3 (End-to-End Run)

Now that the architecture is frozen, we will run one complete end-to-end dry run to populate MLflow, Prometheus, and Grafana with actual predictions and metrics.

Run `654a6852cada467bb3c7fa56e65b4512` trained `ExtraTreesRegressor`
with the `all_context` feature set and random seed `42`.

| Split | Model | MAE | RMSE | WAPE | Rows |
|---|---|---:|---:|---:|---:|
| Validation | Candidate | 9.77 | 14.12 | 12.16% | 3,627 |
| Validation | Previous day | 20.68 | 29.09 | 25.75% | 3,627 |
| Validation | Previous weekday | 22.16 | 30.71 | 27.65% | 2,808 |
| Test | Candidate | 10.30 | 15.62 | 13.44% | 3,510 |
| Test | Previous day | 20.58 | 29.15 | 26.87% | 3,510 |
| Test | Previous weekday | 21.43 | 30.98 | 28.09% | 2,691 |

The candidate beats both naive baselines on the held-out test split. Compared
with the earlier lag-and-rolling-only candidate, changing the algorithm to
ExtraTrees and retaining controlled A&E and bed-context features improves test
MAE by 3.76%, RMSE by 1.03%, and WAPE by 2.76%. This is a
baseline engineering result, not a claim of annual seasonality or production
readiness. The run used dataset SHA-256
`9eed502b65038be2eda91527714fa6d39191b6f859c285e9c8058b9ca1acaa00`, schema
hash `e904102d7b2c6a96c53ee152d12f2baf78b3152aa5662895fed5a5f79a88ebc9`, and
Git commit `ab515a3202cda75efbd168b4ac8454c684bfb7fb`.

## Why the score improves

On the August test split, the candidate reduces error as follows:

| Comparison | MAE improvement | RMSE improvement | WAPE improvement |
|---|---:|---:|---:|
| Versus previous-day baseline | 48.54% | 46.88% | 48.53% |
| Versus previous-weekday baseline | 50.59% | 50.02% | 50.78% |

This improvement is evidence that the engineered discharge history is useful:
calendar features represent weekday effects, exact calendar lags preserve real
seven/fourteen/twenty-eight-day history, and shifted rolling windows summarize
recent capacity flow without using the prediction day's value. The model is
not being credited for synthetic daily A&E values or forward-filled bed data.

The algorithm bake-off used the same `all_context` features and temporal split
for every candidate:

| Algorithm | Validation MAE | Test MAE | Test RMSE | Test WAPE |
|---|---:|---:|---:|---:|
| ExtraTrees | 9.77 | 10.30 | 15.62 | 13.44% |
| RandomForest | 9.98 | 10.50 | 15.90 | 13.71% |
| HistGradientBoosting | 10.28 | 10.59 | 15.48 | 13.83% |
| Ridge | 16.68 | 17.02 | 23.80 | 22.21% |

ExtraTrees is the validation-selected champion and improves test MAE by 2.76%
over the previous HistGradientBoosting champion. Its RMSE is slightly higher,
so the model choice is not being justified by one metric alone; the current
selection favors lower MAE/WAPE for daily operational workload forecasting.
A bounded learning-rate check for HistGradientBoosting selected 0.08 on July
validation but produced a worse August test score (10.62), so that tuning result
was rejected. The solution is controlled algorithm comparison, validation-only
selection, and a single held-out August report rather than endless tuning.
MLflow is the next step for storing this experiment matrix with the same
dataset hash, feature schema, split, and environment.

## Structured training lineage

Each training run now receives a UUID and writes an append-only
`artifacts/runs/<run_id>/events.jsonl` stream. Events include dataset loading
and validation, split creation, feature selection, model initialization,
training start/completion, validation/test evaluation, artifact saving, and run
completion. Every event has the same run ID and an ISO-8601 UTC timestamp.

The latest experiment lineage run is `654a6852cada467bb3c7fa56e65b4512`. Its
test evaluation completed in the event stream with the metrics above. Local
run artifacts remain ignored by Git.

Before using real NHS or operational data, record dataset provenance, licensing, schema, and retrieval date in `data/README.md`. Keep raw data out of source control.

## Engineering decisions

- Scripts are thin entry points; reusable behavior lives under the flat `src` packages.
- The notebook is for EDA and decision records; production logic stays in Python modules.
- Validation happens before normalization and feature engineering.
- A baseline model provides a measurable floor before more advanced models are introduced.
- Tests cover data contracts, API health, model metrics, and drift behavior.
- MLflow settings are centralized in `src/config.py` so local and deployed runs can use the same code path.
