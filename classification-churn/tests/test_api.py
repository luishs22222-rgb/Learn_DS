import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    # Using TestClient as a context manager triggers FastAPI's startup
    # lifespan event (init_schema()) -- without it, the predictions table
    # is never created.
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_model_info_reports_production_version(client):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["production_version"] is not None
    assert body["metrics"]["roc_auc"] == 1.0


def test_predict_stores_and_returns_prediction(client):
    payload = {
        "tenure_months": 3,
        "contract_type": "month-to-month",
        "monthly_charges": 88.0,
        "total_charges": 264.0,
        "support_ticket_count": 2,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["churned_prediction"] in (0, 1)
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["model_version"] is not None

    listed = client.get("/predictions").json()
    assert any(row["id"] == body["prediction_id"] for row in listed)


def test_predict_handles_missing_total_charges(client):
    payload = {
        "tenure_months": 1,
        "contract_type": "month-to-month",
        "monthly_charges": 75.0,
        "support_ticket_count": 0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
