"""FastAPI serving layer.

Two jobs, matching the two halves of "serving" in an ML system:
  - POST /train:   trigger the orchestrated training pipeline (Airflow DAG)
  - POST /predict: serve a prediction from the current Production model,
                    and persist the input + output for future evaluation
                    (see docs/roadmap-phase2.md for why we store this).
"""
from __future__ import annotations

import mlflow
import requests
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient

from customer_churn.config import (
    AIRFLOW_BASE_URL,
    AIRFLOW_DAG_ID,
    AIRFLOW_PASSWORD,
    AIRFLOW_USERNAME,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
)
from customer_churn.db import fetch_recent_predictions, init_schema, insert_prediction
from customer_churn.predict import predict_one, refresh_model
from customer_churn.schemas import (
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    TrainTriggerResponse,
)

app = FastAPI(title="Customer Churn Prediction API", version="0.1.0")


@app.on_event("startup")
def on_startup() -> None:
    init_schema()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    client = MlflowClient()
    versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
    if not versions:
        return ModelInfoResponse(model_name=REGISTERED_MODEL_NAME, production_version=None, metrics=None)
    version = versions[0]
    run = client.get_run(version.run_id)
    return ModelInfoResponse(
        model_name=REGISTERED_MODEL_NAME,
        production_version=str(version.version),
        metrics=dict(run.data.metrics),
    )


@app.post("/train", response_model=TrainTriggerResponse)
def trigger_training():
    """Triggers the Airflow DAG run via Airflow's REST API.

    Requires the Docker Compose stack to be up (Airflow webserver reachable
    at AIRFLOW_BASE_URL). Uses Airflow's default local admin credentials
    (see README for how those are set up in docker-compose.yml).

    Airflow pauses every DAG by default the first time it's parsed
    (`is_paused: true`) -- a *paused* DAG will happily accept a manually
    triggered run via this exact endpoint, but the scheduler leaves it
    sitting in `queued` forever instead of running it. We hit this directly
    while building this project (see the README's troubleshooting section),
    so rather than document "remember to unpause it in the UI first" as a
    required manual step, we just unpause it here before every trigger --
    idempotent (a no-op if it's already unpaused) and it's what makes this
    endpoint actually usable by a caller who's never opened the Airflow UI.
    """
    dag_url = f"{AIRFLOW_BASE_URL}/api/v1/dags/{AIRFLOW_DAG_ID}"
    try:
        requests.patch(
            dag_url,
            json={"is_paused": False},
            auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
            timeout=10,
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502, detail=f"Could not reach Airflow at {dag_url}: {exc}"
        ) from exc

    url = f"{AIRFLOW_BASE_URL}/api/v1/dags/{AIRFLOW_DAG_ID}/dagRuns"
    try:
        resp = requests.post(
            url,
            json={},
            auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD),
            timeout=10,
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502, detail=f"Could not reach Airflow at {url}: {exc}"
        ) from exc

    if resp.status_code not in (200, 201):
        raise HTTPException(
            status_code=502, detail=f"Airflow rejected the trigger request: {resp.status_code} {resp.text}"
        )

    dag_run_id = resp.json().get("dag_run_id")
    return TrainTriggerResponse(
        triggered=True, detail="DAG run triggered.", airflow_dag_run_id=dag_run_id
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        predicted_class, predicted_probability, model_version = predict_one(request.model_dump())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    prediction_id = insert_prediction(
        input_data=request.model_dump(),
        predicted_class=predicted_class,
        predicted_probability=predicted_probability,
        model_name=REGISTERED_MODEL_NAME,
        model_version=model_version,
    )
    return PredictResponse(
        churned_prediction=predicted_class,
        churn_probability=predicted_probability,
        model_name=REGISTERED_MODEL_NAME,
        model_version=model_version,
        prediction_id=prediction_id,
    )


@app.get("/predictions")
def list_predictions(limit: int = 20):
    df = fetch_recent_predictions(limit=limit)
    return df.to_dict(orient="records")


@app.post("/model/refresh")
def refresh():
    """Forces the API to reload the Production model (e.g. right after a promotion)."""
    _, version = refresh_model()
    return {"reloaded": True, "production_version": version}
