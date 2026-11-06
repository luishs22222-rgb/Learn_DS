"""Train candidate models, log them to MLflow, and promote the best one.

This is the concrete implementation of "compare new model vs. previous
model, promote if better" (see docs/glossary.md, "Model registry &
promotion"):

1. Train each candidate model as its own MLflow *run*, logging params,
   regression metrics, and the fitted sklearn Pipeline as a versioned
   model artifact (experiment tracking).
2. Pick the best of the freshly-trained candidates by RMSE.
3. Look up whatever model currently holds the `Production` stage in the
   MLflow *model registry*.
   - If there isn't one yet, promote the new best candidate straight away
     (nothing to compare against).
   - If there is one, evaluate it on the *same* held-out test split and
     only promote the new candidate if it strictly beats it. Otherwise the
     current Production model is left untouched.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from house_prices.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_SEED,
    REGISTERED_MODEL_NAME,
)
from house_prices.db import fetch_training_data
from house_prices.evaluate import compute_regression_metrics
from house_prices.features import build_preprocessor, train_test_split_data


def default_candidates() -> dict[str, BaseEstimator]:
    return {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=RANDOM_SEED
        ),
    }


@dataclass
class TrainedCandidate:
    name: str
    pipeline: Pipeline
    metrics: dict[str, float]
    run_id: str
    model_version: str


def _train_one_candidate(
    name: str,
    estimator: BaseEstimator,
    X_train,
    y_train,
    X_test,
    y_test,
) -> TrainedCandidate:
    pipeline = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", estimator)])

    with mlflow.start_run(run_name=name) as run:
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        metrics = compute_regression_metrics(y_test, preds)

        mlflow.log_param("model_type", name)
        mlflow.log_params({f"model__{k}": v for k, v in estimator.get_params().items()})
        mlflow.log_metrics(metrics)

        model_info = mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
        version = model_info.registered_model_version

        print(f"[{name}] run_id={run.info.run_id} version={version} metrics={metrics}")
        return TrainedCandidate(
            name=name, pipeline=pipeline, metrics=metrics, run_id=run.info.run_id, model_version=version
        )


def _current_production_rmse(client: MlflowClient, X_test, y_test) -> tuple[str | None, float | None]:
    versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
    if not versions:
        return None, None
    prod_version = versions[0]
    prod_model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}/Production")
    preds = prod_model.predict(X_test)
    metrics = compute_regression_metrics(y_test, preds)
    return str(prod_version.version), metrics["rmse"]


def train_and_maybe_promote(
    candidates: dict[str, BaseEstimator] | None = None,
    df: pd.DataFrame | None = None,
) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = df if df is not None else fetch_training_data()
    X_train, X_test, y_train, y_test = train_test_split_data(df)

    candidates = candidates or default_candidates()
    trained = [
        _train_one_candidate(name, estimator, X_train, y_train, X_test, y_test)
        for name, estimator in candidates.items()
    ]
    best = min(trained, key=lambda c: c.metrics["rmse"])
    print(f"Best freshly-trained candidate: {best.name} (rmse={best.metrics['rmse']:.2f})")

    client = MlflowClient()
    prod_version, prod_rmse = _current_production_rmse(client, X_test, y_test)

    result = {
        "best_candidate": best.name,
        "best_rmse": best.metrics["rmse"],
        "best_version": best.model_version,
        "previous_production_version": prod_version,
        "previous_production_rmse": prod_rmse,
        "promoted": False,
    }

    if prod_version is None:
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME, version=best.model_version, stage="Production"
        )
        result["promoted"] = True
        print(
            f"No existing Production model -> promoting {best.name} "
            f"(version {best.model_version}, rmse={best.metrics['rmse']:.2f}) to Production."
        )
    elif best.metrics["rmse"] < prod_rmse:
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=best.model_version,
            stage="Production",
            archive_existing_versions=True,
        )
        result["promoted"] = True
        print(
            f"New candidate {best.name} (rmse={best.metrics['rmse']:.2f}) beats current "
            f"Production version {prod_version} (rmse={prod_rmse:.2f}) -> promoted, old version archived."
        )
    else:
        print(
            f"New candidate {best.name} (rmse={best.metrics['rmse']:.2f}) did NOT beat current "
            f"Production version {prod_version} (rmse={prod_rmse:.2f}) -> keeping existing Production model."
        )

    return result


if __name__ == "__main__":
    outcome = train_and_maybe_promote()
    print(outcome)
    sys.exit(0)
