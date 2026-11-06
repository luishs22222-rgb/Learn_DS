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
    assert body["metrics"]["rmse"] == 2.0


def test_predict_stores_and_returns_prediction(client):
    payload = {
        "sqft": 1850,
        "bedrooms": 3,
        "bathrooms": 2.0,
        "year_built": 1998,
        "lot_size": 6200,
        "neighborhood_name": "Riverside",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["predicted_price"], float)
    assert body["model_version"] is not None

    listed = client.get("/predictions").json()
    assert any(row["id"] == body["prediction_id"] for row in listed)


def test_predict_handles_missing_lot_size(client):
    payload = {
        "sqft": 2400,
        "bedrooms": 4,
        "bathrooms": 3.0,
        "year_built": 1975,
        "neighborhood_name": "Old Town",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
