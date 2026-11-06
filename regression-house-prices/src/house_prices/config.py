"""Central place for paths, env vars, and constants used across the project."""
from __future__ import annotations

import os
from pathlib import Path

# Repo-relative paths. PROJECT_ROOT works whether we're running from a local
# venv (repo checked out on disk) or inside a container (see Dockerfiles,
# which set HOUSE_PRICES_ROOT=/app).
PROJECT_ROOT = Path(os.environ.get("HOUSE_PRICES_ROOT", Path(__file__).resolve().parents[2]))
DB_PATH = Path(os.environ.get("HOUSE_PRICES_DB_PATH", PROJECT_ROOT / "data" / "db" / "house_prices.db"))

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_EXPERIMENT_NAME = "house-price-regression"
REGISTERED_MODEL_NAME = "house-price-model"

AIRFLOW_BASE_URL = os.environ.get("AIRFLOW_BASE_URL", "http://localhost:8081")
AIRFLOW_DAG_ID = "house_price_training"
AIRFLOW_USERNAME = os.environ.get("AIRFLOW_USERNAME", "airflow")
AIRFLOW_PASSWORD = os.environ.get("AIRFLOW_PASSWORD", "airflow")

RANDOM_SEED = 42

NUMERIC_FEATURES = ["sqft", "bedrooms", "bathrooms", "year_built", "lot_size"]
CATEGORICAL_FEATURES = ["neighborhood_name"]
TARGET_COLUMN = "sale_price"
