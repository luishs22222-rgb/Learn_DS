"""Central place for paths, env vars, and constants used across the project."""
from __future__ import annotations

import os
from pathlib import Path

# Repo-relative paths. PROJECT_ROOT works whether we're running from a local
# venv (repo checked out on disk) or inside a container (see Dockerfiles,
# which set CHURN_ROOT=/app).
PROJECT_ROOT = Path(os.environ.get("CHURN_ROOT", Path(__file__).resolve().parents[2]))
DB_PATH = Path(os.environ.get("CHURN_DB_PATH", PROJECT_ROOT / "data" / "db" / "churn.db"))

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5002")
MLFLOW_EXPERIMENT_NAME = "customer-churn-classification"
REGISTERED_MODEL_NAME = "churn-model"

AIRFLOW_BASE_URL = os.environ.get("AIRFLOW_BASE_URL", "http://localhost:8082")
AIRFLOW_DAG_ID = "customer_churn_training"
AIRFLOW_USERNAME = os.environ.get("AIRFLOW_USERNAME", "airflow")
AIRFLOW_PASSWORD = os.environ.get("AIRFLOW_PASSWORD", "airflow")

RANDOM_SEED = 42

NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "total_charges", "support_ticket_count"]
CATEGORICAL_FEATURES = ["contract_type"]
TARGET_COLUMN = "churned"
