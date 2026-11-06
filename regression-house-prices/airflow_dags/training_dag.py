"""Airflow DAG: orchestrates the house-price training pipeline.

generate_new_batch -> validate_data -> build_features -> train_candidates
                    -> evaluate_and_compare -> promote_if_better

Airflow's job here is exactly what a hand-run script can't give you: each
task above is independently retried/logged/scheduled, and the whole graph
is visible and re-triggerable from the UI or the REST API (which is what
the FastAPI `/train` endpoint calls). See docs/glossary.md
("Orchestration / DAG").

Note training/promotion happens inside train_and_maybe_promote() itself
(see train.py) since that logic already needs to be callable standalone
for local runs; this DAG's train/evaluate/promote tasks call into it.

Tasks that need to `import house_prices...` run via
`@task.external_python`, pointed at a separate virtualenv
(/home/airflow/house_prices_venv) baked into the Airflow image -- see
Dockerfile.airflow for why this project's ML dependencies (mlflow, pandas,
pyarrow...) live in their own environment instead of Airflow's own.
"""
from __future__ import annotations

import pendulum
from airflow.decorators import dag, task

DEFAULT_ARGS = {
    "owner": "learning-ds",
    "retries": 1,
}

HOUSE_PRICES_PYTHON = "/home/airflow/house_prices_venv/bin/python"


@dag(
    dag_id="house_price_training",
    schedule=None,  # triggered manually / via the FastAPI /train endpoint, not on a timer
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["house-prices", "regression"],
)
def house_price_training_dag():
    @task.external_python(python=HOUSE_PRICES_PYTHON)
    def generate_new_batch() -> str:
        from house_prices.data_generation import seed_database

        seed_database()
        return "seeded"

    @task.external_python(python=HOUSE_PRICES_PYTHON)
    def validate_data(_: str) -> str:
        """Cheap sanity checks before we trust this data enough to train on it.

        Not a full data-quality framework (see docs/glossary.md, "Data
        validation") -- just the checks that matter for this toy dataset.
        """
        from house_prices.db import fetch_training_data

        df = fetch_training_data()

        expected_columns = {
            "sqft", "bedrooms", "bathrooms", "year_built", "lot_size",
            "neighborhood_name", "location_score", "sale_price", "sale_date",
        }
        missing = expected_columns - set(df.columns)
        if missing:
            raise ValueError(f"Training data is missing expected columns: {missing}")

        MIN_ROWS = 500
        if len(df) < MIN_ROWS:
            raise ValueError(f"Only {len(df)} rows fetched, expected at least {MIN_ROWS}.")

        null_rate = df["lot_size"].isna().mean()
        MAX_NULL_RATE = 0.25
        if null_rate > MAX_NULL_RATE:
            raise ValueError(f"lot_size null rate {null_rate:.2%} exceeds ceiling {MAX_NULL_RATE:.0%}.")

        return "valid"

    @task.external_python(python=HOUSE_PRICES_PYTHON)
    def build_features(_: str) -> str:
        """Feature engineering itself is fit fresh inside train_candidates
        (it must be fit only on the train split, never on the full data) --
        this task exists to make that pipeline stage explicit in the DAG
        graph and to fail fast if the raw fetch/shape is unusable."""
        from house_prices.db import fetch_training_data
        from house_prices.features import split_features_target

        df = fetch_training_data()
        X, y = split_features_target(df)
        if X.isna().all(axis=None):
            raise ValueError("Feature frame is entirely null.")
        return "features-ok"

    @task.external_python(python=HOUSE_PRICES_PYTHON)
    def train_candidates(_: str) -> dict:
        from house_prices.train import train_and_maybe_promote

        return train_and_maybe_promote()

    @task
    def evaluate_and_compare(result: dict) -> dict:
        print(
            f"Best candidate: {result['best_candidate']} "
            f"(rmse={result['best_rmse']:.2f}); "
            f"previous production rmse={result['previous_production_rmse']}"
        )
        return result

    @task
    def promote_if_better(result: dict) -> None:
        if result["promoted"]:
            print(f"Promoted version {result['best_version']} to Production.")
        else:
            print("Kept existing Production model; new candidate was not better.")

    batch = generate_new_batch()
    validated = validate_data(batch)
    features_ok = build_features(validated)
    train_result = train_candidates(features_ok)
    compared = evaluate_and_compare(train_result)
    promote_if_better(compared)


house_price_training_dag()
