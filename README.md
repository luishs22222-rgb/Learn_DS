# Learning Data Science: two end-to-end local ML projects

This repo is a hands-on path through the full machine learning lifecycle —
not just modeling, but the engineering around it: fetching data via SQL,
feature engineering, training with experiment tracking, comparing and
promoting models, orchestrating the pipeline, and serving predictions
through an API. Everything runs **entirely locally** — no cloud account
needed, just Docker and Python.

There are two self-contained projects, built the same way on purpose, so
the second reinforces the *pattern* rather than teaching it from scratch:

| | [`regression-house-prices/`](regression-house-prices/) | [`classification-churn/`](classification-churn/) |
|---|---|---|
| Problem type | Regression (predict a price) | Classification (predict churn yes/no) |
| Why first / second | Metrics are intuitive ($ off by X) | Metrics are genuinely harder to read correctly — precision/recall/ROC-AUC, class imbalance |
| Models | Linear Regression, Random Forest Regressor | Logistic Regression, Random Forest Classifier |

**Do them in order.** The classification project's README assumes you
already understand the architecture from the regression one, and spends
its extra depth on the metrics rather than re-explaining the pipeline.

## How to read this repo

1. **[`docs/glossary.md`](docs/glossary.md)** — every core concept used by
   both projects (EDA, features, train/test split, loss vs. metric,
   model registry, DAGs, etc.), explained once. Read it top to bottom
   before diving into either project, then use it as a reference.
2. **`regression-house-prices/README.md`** — the full step-by-step build
   and run guide for project 1. Start here.
3. **`classification-churn/README.md`** — project 2, same shape, with a
   deep-dive section on classification metrics specifically.
4. **[`docs/roadmap-phase2.md`](docs/roadmap-phase2.md)** — not built yet.
   Once you're comfortable with both projects above, this describes the
   next phase: a drift-detection / performance-monitoring pipeline that
   automatically retrains a model when production data changes, fed by
   synthetic "data shift" injection. Both projects were deliberately built
   to make this easy to add later (see that doc for why).

## What each project actually contains

Both projects follow the identical shape:

- A **mock relational database** (SQLite) standing in for a production
  database — real tables, real SQL joins, queried with `pandas.read_sql`.
- A **synthetic data generator** (numpy/pandas) that seeds that database,
  with an explicit, documented, parameterized data-generating process
  (this is also what makes phase 2's drift simulation possible later).
- An **EDA notebook** exploring the generated data.
- **Feature engineering** (null imputation, one-hot encoding, scaling) via
  a scikit-learn `ColumnTransformer`, wrapped with the model in a single
  `Pipeline` so training and serving can never silently diverge.
- A **training script** that trains candidate models, logs everything to
  **MLflow** (experiment tracking), evaluates them, and promotes the new
  model to `Production` in the MLflow **model registry** only if it beats
  the current one.
- An **Airflow DAG** (run via Docker Compose) that orchestrates the whole
  pipeline above as a scheduled/triggerable graph of tasks.
- A **FastAPI** service with `POST /train` (triggers the Airflow DAG),
  `POST /predict` (serves a prediction from the current `Production` model
  and stores the input + prediction + model version), and a few
  supporting endpoints.
- **Tests** (pytest) for the feature engineering, metrics, and API.

## Prerequisites

- **Docker Desktop** running (for Airflow + MLflow + the API, all local).
- **Python 3.12** — the projects target this version specifically.
  Check with `python3.12 --version`; installed here via Homebrew.
- **[`uv`](https://github.com/astral-sh/uv)** for fast, reproducible Python
  environments (already installed on this machine).

Each project's own README has the exact commands to get from zero to a
running pipeline and a served prediction — including a fast "no Docker
yet" path to build intuition on the ML side before adding orchestration.

## A note on scope

This is a learning repo, intentionally scoped to run on one laptop with no
paid services: SQLite instead of a hosted database, MLflow's local
file/SQLite backend instead of a hosted tracking server, Airflow's
`LocalExecutor` instead of a distributed one. The concepts transfer
directly to cloud/production setups (a managed Postgres instead of SQLite,
a hosted MLflow or SageMaker/Vertex registry instead of local MLflow, a
managed Airflow instead of Docker Compose) — but none of that is needed to
learn what's actually happening.
