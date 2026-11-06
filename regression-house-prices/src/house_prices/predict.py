"""Loads the current Production model from the MLflow registry and predicts.

This is *online* (synchronous, one-request-at-a-time) serving -- see
docs/glossary.md ("Batch vs. online serving"). The model is cached in
memory after first load to avoid re-fetching it from MLflow on every
request; call `refresh_model()` to force a reload (e.g. right after a new
promotion).
"""
from __future__ import annotations

import threading

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from house_prices.config import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME

_lock = threading.Lock()
_cached_model = None
_cached_version: str | None = None


def _production_version(client: MlflowClient) -> str | None:
    versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
    return str(versions[0].version) if versions else None


def refresh_model():
    global _cached_model, _cached_version
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    version = _production_version(client)
    if version is None:
        raise RuntimeError(
            f"No '{REGISTERED_MODEL_NAME}' version is in the Production stage yet. "
            "Run training first (see README)."
        )
    with _lock:
        _cached_model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}/Production")
        _cached_version = version
    return _cached_model, _cached_version


def get_production_model():
    if _cached_model is None:
        return refresh_model()
    return _cached_model, _cached_version


def predict_one(features: dict) -> tuple[float, str]:
    model, version = get_production_model()
    df = pd.DataFrame([features])
    prediction = float(model.predict(df)[0])
    return prediction, version
