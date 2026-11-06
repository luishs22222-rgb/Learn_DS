# Customer Churn Prediction — a local, end-to-end classification ML system

This is project 2 of the two-project repo. It follows the exact same
architecture as [`../regression-house-prices/`](../regression-house-prices/)
on purpose — if you haven't done that one yet, do it first; this README
assumes you already understand the pipeline shape and spends its extra
space on the part the parent README flagged as genuinely harder:
**classification metrics**.

## What you're building

A system that predicts whether a customer will churn (cancel their
subscription), end to end:

- A mock SQLite "production database" of customers, their support tickets,
  and whether they churned, seeded by a synthetic generator with an
  explicit, tunable churn-probability formula.
- Feature engineering (null imputation, one-hot encoding, scaling) and two
  candidate classifiers (Logistic Regression, Random Forest), trained and
  logged to MLflow, with the better one promoted to `Production` in
  MLflow's model registry — only if it actually beats what's already there.
- An Airflow DAG (Docker Compose) orchestrating that whole pipeline.
- A FastAPI service that triggers training and serves predictions —
  storing every prediction for future evaluation (see
  [`../docs/roadmap-phase2.md`](../docs/roadmap-phase2.md)).

Concepts are explained once in [`../docs/glossary.md`](../docs/glossary.md);
this README links there instead of repeating it, except for classification
metrics, which get a full section below with this project's actual numbers.

## Architecture

```
                     ┌─────────────────────────┐
                     │   Airflow DAG (Docker)   │
                     │  generate → validate →   │
                     │  features → train →      │
                     │  compare → promote       │
                     └────────────┬─────────────┘
                                  │ reads/writes
                                  ▼
     ┌────────────────┐   ┌──────────────┐   ┌───────────────────┐
     │ churn.db        │   │   MLflow      │   │   FastAPI          │
     │ (SQLite, mock   │◀──│ tracking +    │──▶│ POST /train        │
     │ relational DB)  │   │ model registry│   │ POST /predict       │
     └────────────────┘   └──────────────┘   │ GET  /predictions   │
              ▲                               │ GET  /model-info    │
              └───────────── predictions ─────┴──────────────────────┘
```

Same shape as project 1, three tables joined via SQL instead of a flat
file: `customers`, `support_tickets` (aggregated via a SQL `GROUP BY` —
project 1 only needed a `JOIN`), and `churn_labels`.

## Prerequisites

Same as project 1: Docker Desktop running, Python 3.12, `uv`. See
[`../README.md`](../README.md) if you haven't set these up.

## Part 1 — run it locally first, no Docker

```bash
cd classification-churn
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Seed the mock database with a fresh synthetic batch (2500 customers)
python scripts/seed_db.py

# In a second terminal: run a local MLflow tracking server
mlflow server --host 0.0.0.0 --port 5002 \
  --backend-store-uri sqlite:///.mlflow_local/mlflow.db \
  --default-artifact-root ./.mlflow_local/artifacts

# Back in the first terminal: train
python -m customer_churn.train
```

You'll see two MLflow runs (`logistic_regression`, `random_forest`) each
print their metrics, then a line like:

```
No existing Production model -> promoting logistic_regression (version 1, roc_auc=0.7627) to Production.
```

Run `python -m customer_churn.train` **again** without reseeding: you'll
see the new candidate get **rejected** —

```
New candidate logistic_regression (roc_auc=0.7627) did NOT beat current Production version 1 (roc_auc=0.7627) -> keeping existing Production model.
```

— because on identical data, both models are deterministic (fixed random
seeds), so the "new" run ties the existing Production model exactly rather
than strictly beating it. Now reseed with a different seed
(`python scripts/seed_db.py --seed 7`) and train once more: this time the
new candidate genuinely differs and, if it scores higher, gets promoted
and the old version is archived. All three branches of the promotion logic
are real code paths you can trigger this way, not just described in a
comment.

Now serve it:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8002
```

```bash
curl http://localhost:8002/model-info

curl -X POST http://localhost:8002/predict -H "Content-Type: application/json" -d '{
  "tenure_months": 3, "contract_type": "month-to-month",
  "monthly_charges": 88.0, "total_charges": 264.0, "support_ticket_count": 2
}'

curl http://localhost:8002/predictions
```

Reseed back to the default before moving on (`python scripts/seed_db.py`),
and remember to stop the local `mlflow server`/`uvicorn` processes before
Part 2 — see Troubleshooting.

## Part 2 — the full stack with Docker Compose (Airflow + MLflow + API)

```bash
docker compose build
docker compose up -d
```

Once `docker compose ps` shows `airflow-webserver` as `healthy`
(typically well under a minute):

- Airflow UI: http://localhost:8082 (login `airflow` / `airflow` — Docker
  Compose's local dev setup, created by `airflow-init`; never reuse
  credentials like this outside a throwaway local project).
- MLflow UI: http://localhost:5002
- API: http://localhost:8002 (same endpoints as Part 1)

```bash
curl -X POST http://localhost:8002/train
```

This does two things against Airflow's REST API: it unpauses the
`customer_churn_training` DAG (new DAGs are paused by default — see
Troubleshooting for why this matters) and then triggers a run, so this one
call is enough even if you've never opened the Airflow UI. Watch it run in
the UI's Grid view, then check:

```bash
curl http://localhost:8002/model-info
curl -X POST http://localhost:8002/predict -H "Content-Type: application/json" -d '{
  "tenure_months": 2, "contract_type": "month-to-month",
  "monthly_charges": 95.0, "total_charges": 190.0, "support_ticket_count": 3
}'
```

and the MLflow UI's "churn-model" registered model page, which should show
the version that was just promoted to `Production`.

When you're done: `docker compose down` (keeps the named volumes — seeded
data, Airflow metadata, MLflow registry all survive the next
`docker compose up`). Add `-v` for a fully clean slate.

## File-by-file walkthrough

- **`src/customer_churn/config.py`** — every path/URL/constant in one
  place. Same shape as project 1's `config.py`.
- **`src/customer_churn/db.py`** — the mock relational database.
  `fetch_training_data()`'s query does a `JOIN` (like project 1) *plus* a
  `GROUP BY` subquery that turns raw `support_tickets` rows into a
  `support_ticket_count` feature, with `COALESCE(..., 0)` for customers
  with zero tickets — a join-cardinality gap, not a NULL, which is a
  distinct SQL idea from the genuine NULLs `total_charges` has.
- **`src/customer_churn/data_generation.py`** — the synthetic generator.
  `ChurnCoefficients` is an explicit logistic formula
  (`churn_probability = sigmoid(intercept + tenure_coef·tenure + ...)`),
  every coefficient a function argument, same reasoning as project 1's
  `PricingCoefficients` (see [`../docs/roadmap-phase2.md`](../docs/roadmap-phase2.md)).
  Tuned to produce a **deliberately imbalanced** ~20-25% churn rate — see
  the metrics section below for why that matters.
- **`src/customer_churn/features.py`** — the `ColumnTransformer`, plus a
  **stratified** train/test split (`stratify=y`) — project 1 didn't need
  this since its regression target isn't a rare class that a random split
  could accidentally under-represent on one side.
- **`src/customer_churn/evaluate.py`** — accuracy/precision/recall/F1/ROC-AUC
  plus the raw confusion-matrix counts.
- **`src/customer_churn/train.py`** — same "compare vs. previous, promote
  if better" shape as project 1, but the comparison metric is **ROC-AUC**,
  not accuracy (see below for why), and evaluating the current Production
  model uses `mlflow.sklearn.load_model` (not the generic `pyfunc` flavor)
  specifically so `.predict_proba()` is available for that comparison.
- **`src/customer_churn/predict.py`** — loads `models:/churn-model/Production`
  and returns both the hard class and the churn probability (a classifier
  serving endpoint that only returns 0/1 throws away the information you'd
  need to change the decision threshold later — see the metrics section).
- **`app/main.py`** — same endpoint shape as project 1. One deliberate
  addition: `POST /train` first calls Airflow's REST API to **unpause**
  `customer_churn_training` before triggering a run — see this file's
  docstring and Troubleshooting for why.
- **`airflow_dags/training_dag.py`** — same DAG shape as project 1
  (`generate → validate → features → train → compare → promote`), same
  `@task.external_python` pattern pointed at a separate
  `/home/airflow/customer_churn_venv` (see `Dockerfile.airflow`'s comment
  for why Airflow's and MLflow's dependencies can't share one environment).
  `validate_data` additionally sanity-checks the churn rate itself is
  within a plausible range, since a broken generator producing e.g. 0% or
  100% churn would otherwise train "successfully" on garbage.
- **`docker-compose.yml`** — identical shape to project 1's, on different
  host ports (8082 Airflow / 5002 MLflow / 8002 API) so both projects can
  run at the same time without colliding.
- **`tests/`** — same shape as project 1: `test_features.py`,
  `test_evaluate.py` (including a test that deliberately shows a
  100%-"no churn" model scoring 80% accuracy with 0% recall — the exact
  failure mode this project's promotion metric is chosen to avoid), and
  `test_api.py` against a throwaway MLflow store, no Docker required.

## The classification metrics, with this project's actual numbers

Running the pipeline against the default synthetic batch (`seed_db.py`
with its defaults, 2500 customers, 500 held out for the test split)
produced:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 80.2% | 65.7% | 20.9% | 0.317 | **0.763** |
| Random Forest (200 trees) | 79.2% | 59.4% | 17.3% | 0.268 | 0.736 |

Logistic regression wins, for the same reason linear regression won in
project 1: the synthetic churn probability *is* a logistic function of the
features (see `ChurnCoefficients`), so the correctly-specified simple
model has an inherent edge over a more flexible one fitting the same
signal plus noise.

Logistic regression's confusion matrix on the 500-row test set (110 actual
churners, 390 actual stayers):

| | Predicted: stayed | Predicted: churned |
|---|---|---|
| **Actual: stayed** | 378 (TN) | 12 (FP) |
| **Actual: churned** | 87 (FN) | 23 (TP) |

Walking through what each number means, in churn terms:

- **Why not just use accuracy (80.2%)?** A model that predicted "no churn"
  for every single customer would score **78.0%** accuracy on this exact
  test set (390 actual stayers ÷ 500) — nearly matching our trained
  model's 80.2% — while catching precisely zero churners. Accuracy barely
  moved because churners are rare; it's the wrong ruler for this problem.
  This is why `train.py` promotes on ROC-AUC instead (see `evaluate.py`'s
  docstring), and `tests/test_evaluate.py` has a test that reproduces this
  exact scenario on purpose.
- **Recall (20.9%)** is the uncomfortable number: of the 110 customers who
  actually churned, the model only flagged 23 of them (23/110) — it missed
  87. A missed churner (a **false negative**) is a customer who leaves
  with zero retention outreach, because the model never raised a flag.
- **Precision (65.7%)** is more reassuring: of the 35 customers the model
  did flag as high-risk (23 + 12), about two-thirds (23/35) really were
  about to churn. A **false positive** here means a retention offer (a
  discount, a call from support) spent on someone who wasn't actually
  leaving — a real cost, but usually a much cheaper mistake than a missed
  churner.
- **The precision/recall tradeoff, concretely**: the model outputs a
  *probability* (that's why `/predict` returns `churn_probability`, not
  just the 0/1 class), and the 0.5 cutoff used for the table above is just
  a default. Lowering the threshold (flag more customers as at-risk) would
  catch more true churners (higher recall) at the cost of more wasted
  retention offers (lower precision); raising it does the opposite. Which
  way to move it is a *business* decision (how expensive is a retention
  offer vs. how expensive is losing a customer silently), not a modeling
  one — the model's job is to produce a well-*ranked* probability; picking
  the threshold is a separate step this project deliberately leaves to
  you.
- **ROC-AUC (0.763)** sidesteps the threshold question entirely: it's the
  probability that, given one random churner and one random stayer, the
  model scores the churner higher. 0.5 would be a coin flip; 1.0 would be
  perfect ranking. 0.763 says the model has learned a real, useful signal
  — clearly better than chance, with room to improve (e.g. more/better
  features) before it'd be reliable enough to automate retention spend
  entirely on its output.

## Troubleshooting (things we actually hit)

- **`POST /train` seemed to do nothing.** Airflow pauses every DAG by
  default the moment it's first parsed (`is_paused: true`). A paused DAG
  still *accepts* a manually triggered run through the REST API — the run
  gets created and sits in `queued` — but the scheduler will never
  actually execute it until the DAG is unpaused. We hit this directly:
  after triggering a run and polling its state, it sat in `queued` for
  several minutes with no errors anywhere in the scheduler logs, until we
  unpaused the DAG via `PATCH /api/v1/dags/customer_churn_training
  {"is_paused": false}`, at which point it went `queued → running →
  success` within about 20 seconds. Rather than document "remember to
  unpause it first" as a step you have to know about, `app/main.py`'s
  `/train` endpoint now does that PATCH itself before every trigger (see
  its docstring) — so this should no longer bite you, but if you trigger
  runs directly against Airflow's API/UI instead of through `/train`,
  remember the DAG needs to be unpaused at least once.
- **Local `mlflow server` / `uvicorn` processes squatting on Docker's
  ports.** If you ran Part 1 and left the local `mlflow server` (port
  5002) or `uvicorn` (port 8002) running, `docker compose up` will still
  start, but the API/MLflow containers' ports silently won't be reachable
  the way you expect (or the leftover local process answers instead of
  the container). Kill anything bound to 5002/8002 before Part 2 —
  `lsof -i :5002 -i :8002` to find it.
- **MLflow image version matters.** `ghcr.io/mlflow/mlflow:latest`
  currently resolves to an MLflow 3.x server, which isn't wire-compatible
  with the `mlflow==2.16.x` client pinned in `pyproject.toml`. The image
  tag is pinned to `v2.16.2` for exactly this reason — don't bump one
  without the other.
- **Artifact storage across containers.** The MLflow service intentionally
  does *not* set `--default-artifact-root` to a plain local path — see the
  comment in `docker-compose.yml`. A plain local path only works when
  client and server share a filesystem; here they're different
  containers, so artifacts are proxied over HTTP via MLflow's
  `mlflow-artifacts:/` scheme instead (project 1 hit the literal
  `PermissionError` version of this bug first; this project's
  `docker-compose.yml` was written with the fix already in place).
- **Airflow's dependency pins vs. MLflow's.** Airflow 2.10's constraints
  file conflicts with mlflow's own transitive pins (SQLAlchemy, pyarrow).
  `Dockerfile.airflow` resolves this with a *separate* virtualenv for this
  project's dependencies rather than installing them into Airflow's own
  environment — see that file's comment for the full reasoning.

## What's next

Once you're comfortable with both projects, [`../docs/roadmap-phase2.md`](../docs/roadmap-phase2.md)
describes the planned next phase: a drift-detection / evaluation pipeline
that watches the `predictions` table this project already populates, and
automatically re-triggers training when the data or the model's live
performance has genuinely shifted.
