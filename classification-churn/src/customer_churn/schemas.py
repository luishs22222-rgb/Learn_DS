"""Pydantic request/response models for the FastAPI app."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    tenure_months: int = Field(..., ge=0, example=14)
    contract_type: str = Field(..., example="month-to-month")
    monthly_charges: float = Field(..., ge=0, example=79.5)
    total_charges: float | None = Field(None, ge=0, example=1113.0)
    support_ticket_count: int = Field(0, ge=0, example=1)


class PredictResponse(BaseModel):
    churned_prediction: int
    churn_probability: float
    model_name: str
    model_version: str
    prediction_id: int


class ModelInfoResponse(BaseModel):
    model_name: str
    production_version: str | None
    metrics: dict[str, float] | None


class TrainTriggerResponse(BaseModel):
    triggered: bool
    detail: str
    airflow_dag_run_id: str | None = None
