"""Test isolation: point the whole app at throwaway DB/MLflow stores.

These env vars must be set *before* anything under customer_churn/ or app/
is imported anywhere in the test session, since customer_churn.config reads
them once at import time. conftest.py is always imported first by pytest,
so this is sufficient -- no monkeypatching of already-bound module
constants required.
"""
import os
import shutil
from pathlib import Path

_TMP_DIR = Path(__file__).resolve().parent / ".tmp_test_state"
if _TMP_DIR.exists():
    shutil.rmtree(_TMP_DIR)
_TMP_DIR.mkdir(parents=True)

os.environ["CHURN_DB_PATH"] = str(_TMP_DIR / "test_churn.db")
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_TMP_DIR / 'test_mlflow.db'}"

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from customer_churn.config import REGISTERED_MODEL_NAME  # noqa: E402
from customer_churn.features import build_preprocessor  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def registered_production_model():
    """Fits and registers a tiny real model as Production, so /predict and
    /model-info have something to serve without needing a full training run
    or Docker/Airflow up."""
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("customer-churn-classification-tests")

    df = pd.DataFrame(
        {
            "tenure_months": [2, 40, 5, 60, 12, 55, 3, 30] * 3,
            "contract_type": ["month-to-month", "two-year", "month-to-month", "two-year",
                               "one-year", "two-year", "month-to-month", "one-year"] * 3,
            "monthly_charges": [85.0, 45.0, 95.0, 40.0, 65.0, 42.0, 90.0, 60.0] * 3,
            "total_charges": [170.0, 1800.0, 475.0, 2400.0, 780.0, 2310.0, 270.0, 1800.0] * 3,
            "support_ticket_count": [2, 0, 3, 0, 1, 0, 2, 1] * 3,
        }
    )
    y = pd.Series([1, 0, 1, 0, 0, 0, 1, 0] * 3)

    pipeline = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", LogisticRegression(max_iter=1000))])
    with mlflow.start_run(run_name="test-fixture-model") as run:
        pipeline.fit(df, y)
        mlflow.log_metrics(
            {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "roc_auc": 1.0}
        )
        model_info = mlflow.sklearn.log_model(
            pipeline, artifact_path="model", registered_model_name=REGISTERED_MODEL_NAME
        )

    client = MlflowClient()
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=model_info.registered_model_version,
        stage="Production",
    )
    yield
