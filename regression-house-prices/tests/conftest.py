"""Test isolation: point the whole app at throwaway DB/MLflow stores.

These env vars must be set *before* anything under house_prices/ or app/
is imported anywhere in the test session, since house_prices.config reads
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

os.environ["HOUSE_PRICES_DB_PATH"] = str(_TMP_DIR / "test_house_prices.db")
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_TMP_DIR / 'test_mlflow.db'}"

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from house_prices.config import REGISTERED_MODEL_NAME  # noqa: E402
from house_prices.features import build_preprocessor  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def registered_production_model():
    """Fits and registers a tiny real model as Production, so /predict and
    /model-info have something to serve without needing a full training run
    or Docker/Airflow up."""
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("house-price-regression-tests")

    df = pd.DataFrame(
        {
            "sqft": [1200, 1800, 2200, 2400, 1500, 3000, 1100, 2000] * 3,
            "bedrooms": [2, 3, 3, 4, 2, 5, 1, 3] * 3,
            "bathrooms": [1.0, 2.0, 2.0, 3.0, 1.5, 3.5, 1.0, 2.0] * 3,
            "year_built": [1990, 2000, 1985, 2010, 1995, 2015, 1975, 2005] * 3,
            "lot_size": [5000, 5500, 6000, 7000, 4800, 8000, 4000, 6500] * 3,
            "neighborhood_name": ["Riverside", "Old Town", "Riverside", "Maple Heights",
                                   "Sunset District", "Riverside", "Old Town", "Maple Heights"] * 3,
        }
    )
    y = pd.Series([300000, 350000, 320000, 500000, 310000, 600000, 250000, 420000] * 3)

    pipeline = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", LinearRegression())])
    with mlflow.start_run(run_name="test-fixture-model") as run:
        pipeline.fit(df, y)
        mlflow.log_metrics({"mae": 1.0, "rmse": 2.0, "r2": 0.9})
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
