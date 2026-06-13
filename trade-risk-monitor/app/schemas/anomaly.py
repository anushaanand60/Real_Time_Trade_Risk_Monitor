from pydantic import BaseModel, field_validator
from decimal import Decimal
from datetime import datetime
from typing import Optional, Any
import json

class AnomalyTrainRequest(BaseModel):
    contamination: Optional[float] = 0.01

class AnomalyTrainResponse(BaseModel):
    status: str
    trained_at: str
    contamination: float
    training_rows: int

class AnomalyScoreResponse(BaseModel):
    portfolio_id: int
    timestamp: datetime
    anomaly_score: Optional[float] = None
    is_anomaly: Optional[bool] = None
    anomaly_explanation: Optional[list[str]] = None

    @field_validator("anomaly_explanation", mode="before")
    @classmethod
    def parse_explanation(cls, v: Any) -> Optional[list[str]]:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return [v]
        return v

class AnomalyHistoryResponse(BaseModel):
    id: int
    portfolio_id: int
    timestamp: datetime
    anomaly_score: Optional[float] = None
    is_anomaly: Optional[bool] = None
    anomaly_explanation: Optional[list[str]] = None

    @field_validator("anomaly_explanation", mode="before")
    @classmethod
    def parse_explanation(cls, v: Any) -> Optional[list[str]]:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return [v]
        return v
    model_config = {"from_attributes": True}
