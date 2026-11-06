# House Price Prediction — a local, end-to-end regression ML system

This project is a complete, working machine learning system for predicting
house sale prices from property features. It's project 1 of a two-project
learning path (see the [repo root README](../README.md)); it's the
regression project on purpose, because its metrics (dollars off) are easy
to sanity-check, which lets you focus on the *pipeline* rather than on
interpreting the model.

Every core concept it uses is explained once in
**[`../docs/glossary.md`](../docs/glossary.md)** — this README links to it
rather than re-explaining EDA, one-hot encoding, train/test splits, etc.
Read the glossary first if any term below is unfamiliar.

## What you're building

A system that can:
1. Pull training data out of a mock relational database using real SQL.
2. Train two candidate regression models and track every run in MLflow.
3. Promote a new model to "Production" only if it's actually better than
   whatever is currently serving predictions.
4. Run that whole process as an orchestrated, re-triggerable Airflow DAG.
5. Serve predictions from the current Production model over an HTTP API,
   and persist every prediction made.

## Architecture

```
                    ┌─────────────────────────┐
                    │   SQLite (mock "prod" DB) │
                    │  locations / properties /  │
                    │  sales / predictions       │
                    └───────────┬────────────────┘
                                │ SQL (JOIN, fetch, insert)
        ┌───────────────────────┼─────────────────────────┐
        │                       │                          │
        ▼                       ▼                          ▼
┌───────────────┐   ┌────────────────────┐      ┌─────────────────────┐
│  Airflow DAG    │  │   train.py           │      │   FastAPI (app/)      │
│  (Docker Compose)│─▶│  train -> compare    │      │  POST /train  ───────▶│ triggers the DAG
│  house_price_    │  │  -> promote          │      │  POST /predict         │ via Airflow REST API
│  training        │  └──────────┬───────────┘      │  GET /predictions      │
└───────────────┘             │                    │  GET /model-info       │
                                ▼                    └───────────┬────────────┘
                        ┌───────────────┐                        │ loads Production
                        │    MLflow       │◀───────────────────────┘  model, writes
                        │  tracking +     │                              predictions back
                        │  model registry │                              to SQLite
                        └───────────────┘
```

Two databases are involved and it's worth being explicit about the
difference, since it trips people up: **Postgres** in `docker-compose.yml`
is Airflow's own internal metadata database (DAG runs, task state, users) —
you never query it directly. The **SQLite** file under `data/db/` is our
mock *business* database (the actual house listings) — that's the one
`db.py` runs SQL against.

## Prerequisites

- Docker Desktop, running.
- Python 3.12 (`python3.12 --version`).
- [`uv`](https://github.com/astral-sh/uv) for the local virtualenv.

## Part 1 — run it locally first, no Docker

Get the ML pipeline itself working end to end before adding orchestration —
it's much easier to debug a training bug without Airflow in the loop.

```bash
cd regression-house-prices
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev]"

# Seed the mock database with a fresh synthetic batch (2000 properties)
.venv/bin/python scripts/seed_db.py

# In a second terminal: run a local MLflow tracking server
.venv/bin/mlflow server --host 127.0.0.1 --port 5001 \
    --backend-store-uri sqlite:///.mlflow_local/mlflow.db \
    --default-artifact-root ./.mlflow_local/artifacts

# Back in the first terminal: train
MLFLOW_TRACKING_URI=http://127.0.0.1:5001 .venv/bin/python -m house_prices.train
```

Open http://127.0.0.1:5001 to see the run in the MLflow UI. The first run
has nothing to compare against, so the better of the two candidates is
promoted to `Production` automatically. **Run `train.py` again** (same
data) and you'll see the opposite: the new candidate ties the existing
Production model, so it's *not* promoted — proving the "only promote if
strictly better" logic actually branches, not just that it always
promotes. (We verified all three real branches while building this:
first-ever promotion, a rejected non-improvement, and an accepted
improvement after reseeding with different data.)

Now serve it:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5001 .venv/bin/uvicorn app.main:app --port 8001

curl -X POST http://127.0.0.1:8001/predict -H "Content-Type: application/json" -d '{
  "sqft": 1850, "bedrooms": 3, "bathrooms": 2.0,
  "year_built": 1998, "lot_size": 6200, "neighborhood_name": "Riverside"
}'

curl http://127.0.0.1:8001/predictions
```

You should see a prediction back, and then see that same prediction listed
via `/predictions` — it was written to the `predictions` table in the same
SQLite file. Try omitting `lot_size` from the request too: the
preprocessing pipeline imputes it, exactly like it does at training time.

Explore `notebooks/01_eda.ipynb` at this point as well (`.venv/bin/jupyter
lab`) — it's the EDA pass that motivated the feature engineering choices
below.

**Stop the local `mlflow server` and `uvicorn` processes before moving to
Part 2** — they bind the same host ports (5001, 8001) that Docker Compose
will also try to use. On macOS in particular, having both bound
simultaneously doesn't always error cleanly; it can silently route your
requests to whichever process is listening, which is confusing to debug.

## Part 2 — the full stack with Docker Compose (Airflow + MLflow + API)

```bash
docker compose build
docker compose up -d
```

First boot takes a few minutes: Postgres has to become healthy, then
`airflow-init` runs database migrations and creates the default user, then
the webserver/scheduler start. **On the very first run after starting the
stack, give the scheduler a few minutes before triggering the DAG** — we
observed the first task past DAG-parsing time taking close to 5 minutes to
get picked up right after a cold start (Airflow's scheduler doing its
initial DAG-file processing pass); every run after that started within
seconds.

Once `docker compose ps` shows `airflow-webserver` as `healthy`:

- Airflow UI: http://localhost:8081 (login `airflow` / `airflow` — this is
  Docker Compose's local dev setup, created by the `airflow-init` service;
  never reuse credentials like this outside a throwaway local project).
- MLflow UI: http://localhost:5001
- API: http://localhost:8001 (same endpoints as Part 1)

Trigger the pipeline either from the Airflow UI (unpause `house_price_training`,
then hit the "trigger" button) or through the API, which is what a real
caller would do:

```bash
curl -X POST http://localhost:8001/train
```

Watch it run in the Airflow UI's Grid view. Once it succeeds, check:

```bash
curl http://localhost:8001/model-info
curl -X POST http://localhost:8001/predict -H "Content-Type: application/json" -d '{
  "sqft": 1850, "bedrooms": 3, "bathrooms": 2.0,
  "year_built": 1998, "lot_size": 6200, "neighborhood_name": "Riverside"
}'
```

and the MLflow UI's "house-price-model" registered model page, which
should show the version that was just promoted to `Production`.

When you're done: `docker compose down` (keeps the named volumes, so the
seeded data, Airflow metadata, and MLflow registry survive the next
`docker compose up`). Add `-v` if you want a fully clean slate.

## File-by-file walkthrough

- **`src/house_prices/config.py`** — every path/URL/constant in one place
  (`DB_PATH`, `MLFLOW_TRACKING_URI`, `AIRFLOW_BASE_URL`, the random seed).
  Reads from environment variables with local-dev defaults, so the same
  code runs unmodified locally and in Docker.
- **`src/house_prices/db.py`** — the mock relational database. Real DDL
  (three related tables, see the glossary's "Relational database" entry),
  and `fetch_training_data()` runs one SQL query that `JOIN`s all three.
- **`src/house_prices/data_generation.py`** — the synthetic data generator.
  The pricing relationship (`PricingCoefficients`) is an explicit,
  documented formula with every coefficient as a function argument — not
  because a toy project needs configurability, but because that's exactly
  what will let a future drift-simulation phase reuse this generator with
  different coefficients instead of needing a new data source (see
  [`../docs/roadmap-phase2.md`](../docs/roadmap-phase2.md)).
- **`src/house_prices/features.py`** — the `ColumnTransformer` (median
  imputation + scaling for numerics, one-hot encoding for
  `neighborhood_name`) and the train/test split.
- **`src/house_prices/evaluate.py`** — MAE/RMSE/R² computation, shared by
  training and (if you extend this later) monitoring.
- **`src/house_prices/train.py`** — trains both candidates as separate
  MLflow runs, then implements the actual "compare vs. previous, promote if
  better" decision against whatever is currently in the `Production` stage.
- **`src/house_prices/predict.py`** — loads `models:/house-price-model/Production`
  and caches it in memory for serving.
- **`app/main.py`** — the FastAPI app. `/train` calls Airflow's REST API
  (`POST /api/v1/dags/house_price_training/dagRuns`) rather than running
  training in-process, so the API stays fast and the heavy work stays on
  the orchestrator. `/predict` writes every prediction to SQLite.
- **`airflow_dags/training_dag.py`** — the DAG. Notice the tasks that need
  `house_prices` use `@task.external_python`, pointed at a *separate*
  virtualenv baked into the Airflow image
  (`/home/airflow/house_prices_venv`) rather than Airflow's own
  environment — see the comment at the top of `Dockerfile.airflow` for why
  (short version: Airflow 2.10 pins old versions of shared libraries like
  SQLAlchemy, and MLflow's own dependencies conflict with those pins; this
  is a common, realistic way to resolve it, not a workaround for a mistake).
- **`docker-compose.yml`** — Postgres (Airflow's metadata DB only) +
  Airflow init/webserver/scheduler + MLflow + the API. The MLflow service
  deliberately does *not* set `--default-artifact-root` to a plain local
  path — see the comment inline. A plain local path only works when the
  MLflow *client* and *server* share a filesystem; here they're different
  containers, so artifacts must be proxied over HTTP (MLflow's
  `mlflow-artifacts:/` scheme, the default when `--serve-artifacts` is on).
  We hit this exact bug while building the project (`PermissionError:
  '/mlflow'` from inside the Airflow container) — it's a good example of a
  mistake that only shows up once client and server actually run on
  separate machines/containers, not on a single laptop.
- **`tests/`** — `test_features.py` and `test_evaluate.py` are pure unit
  tests. `test_api.py` spins up the FastAPI app against a throwaway
  MLflow store and SQLite file (see `conftest.py`) — no Docker required to
  run the test suite.

## The regression metrics, with this project's actual numbers

Running the pipeline against the default synthetic batch (`seed_db.py`
with its defaults) produced:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | $14,012 | $17,645 | 0.964 |
| Random Forest (200 trees) | $18,270 | $23,336 | 0.937 |

Linear regression wins here, and that's not a fluke worth "fixing" — look
at `data_generation.py`: the true price relationship *is* linear (a sum of
coefficients times features, plus Gaussian noise), so a linear model is
the correctly-specified one, and the random forest is spending its extra
flexibility fitting noise instead of a real nonlinear pattern it doesn't
need. This is a good illustration of why you train more than one model
type rather than assuming "more complex = better."

Reading the numbers: **RMSE ($17,645)** is what `train.py` uses to compare
candidates and decide promotion, because squaring errors first means a few
badly-wrong predictions move it more than MAE — appropriate when large
errors are worse than proportionally more small ones. **MAE ($14,012)** is
the plainer "on average, off by" number. **R² (0.964)** says the model
explains about 96% of the variance in sale price — the closer to 1, the
better, with 0 meaning "no better than always predicting the mean."

## Troubleshooting (things we actually hit)

- **`PermissionError: [Errno 13] Permission denied: '/mlflow'` in a
  training task.** This means MLflow is trying to write artifacts to a
  local filesystem path that only exists in the *server's* container. Make
  sure `docker-compose.yml`'s `mlflow` service is not passing
  `--default-artifact-root` as a plain path — see the walkthrough above.
- **First DAG run after `docker compose up` seems stuck in `running`.**
  Give it a few minutes — this was a real, reproducible delay on cold
  start in our testing, not a hang. Every subsequent run started quickly.
- **`/predict` or `/model-info` returns stale-looking data, or requests
  seem to hit the wrong process.** Check for a leftover local `mlflow
  server` or `uvicorn` process still bound to ports 5001/8001 from Part 1
  — `lsof -iTCP:5001 -sTCP:LISTEN` / `lsof -iTCP:8001 -sTCP:LISTEN`. Kill
  them before starting Docker Compose.
- **`docker compose build` fails resolving dependencies for the Airflow
  image.** If you add a new dependency to `pyproject.toml`, remember it
  only needs to satisfy the *separate* `house_prices_venv` (unconstrained),
  not Airflow's own constraints file — see `Dockerfile.airflow`. You
  shouldn't need to touch the constraints URL at all for normal changes.
- **MLflow model-registry `FutureWarning`s about stages being
  deprecated.** Expected and harmless for this project (pinned to
  `mlflow<2.17`, where stages are fully functional, just discouraged for
  new code going forward) — see the note in `train.py`. A newer MLflow
  would use "aliases" instead; stages were kept here because they map
  directly onto the `Staging → Production` mental model this project
  teaches.

## What's next

Once you're comfortable with this project and its classification sibling
(`../classification-churn/`), see
[`../docs/roadmap-phase2.md`](../docs/roadmap-phase2.md) for the planned
drift-detection and auto-retraining phase — this project's `predictions`
table (model version + inputs + output, all timestamped) and the
parameterized `data_generation.py` were both built with that phase in mind.
