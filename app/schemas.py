"""Strict request/response contracts of the inference API.

The per-transaction contract (``Transaction``) is imported from the shared
library: it is the very model that validated the training data.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from fraud_detection.schema import TRANSACTION_EXAMPLE, Transaction

MAX_BATCH_SIZE: Final = 1_000


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"instances": [TRANSACTION_EXAMPLE]}]},
    )

    instances: list[Transaction] = Field(
        min_length=1, max_length=MAX_BATCH_SIZE, description="Transactions to score (batch)."
    )


class RiskLevel(StrEnum):
    LOW = "low"  # approve
    MEDIUM = "medium"  # step-up authentication / manual review
    HIGH = "high"  # decline / block


class Prediction(BaseModel):
    transaction_id: str | None
    fraud_probability: float = Field(ge=0.0, le=1.0)
    is_fraud: bool = Field(description="fraud_probability >= decision_threshold")
    risk_level: RiskLevel


class PredictResponse(BaseModel):
    model_name: str
    model_version: str = Field(description="Content hash (sha256 prefix) of the served artifact.")
    quantization: str
    decision_threshold: float
    review_threshold: float
    predictions: list[Prediction]
    inference_ms: float = Field(description="Featurization + inference time inside the service.")


class ModelInfo(BaseModel):
    name: str
    version: str
    sha256: str
    path: str
    quantization: str
    file_bytes: int
    resident_bytes: int
    n_trees: int
    tree_depth: int
    features: list[str]
    decision_threshold: float
    review_threshold: float
    schema_fingerprint: str
    trained_at: str | None
    validation_roc_auc: float | None
    validation_pr_auc: float | None


class Health(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_version: str | None = None
