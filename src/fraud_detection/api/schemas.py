from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    model_name: str = Field(default="cnn")
    feature_vector: list[float]
    transaction_id: str
    user_identifier: str | None = None
    merchant_identifier: str | None = None


class PredictionResponse(BaseModel):
    model_name: str
    prediction: str
    confidence: float
    fraud_probability: float
    timestamp: datetime
    interpretability: dict[str, Any]


class SwitchModelRequest(BaseModel):
    model_name: str
    checkpoint_path: str


class HealthResponse(BaseModel):
    status: str
    active_model: str | None
