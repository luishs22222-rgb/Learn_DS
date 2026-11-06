"""Pydantic request/response models for the FastAPI app."""
from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    sqft: float = Field(..., gt=0, example=1850)
    bedrooms: int = Field(..., ge=0, example=3)
    bathrooms: float = Field(..., ge=0, example=2.0)
    year_built: int = Field(..., ge=1800, le=2026, example=1998)
    lot_size: float | None = Field(None, ge=0, example=6200)
    neighborhood_name: str = Field(..., example="Riverside")


class PredictResponse(BaseModel):
    predicted_price: float
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
