from pydantic import BaseModel
from datetime import datetime
from typing import Optional, Dict, List

class RiskClassifierTrainResponse(BaseModel):
    status: str
    trained_at: str
    selected_model: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    confusion_matrix: List[List[int]]
    class_counts: Dict[str, int]
    top_features: List[str]
    training_rows: int
    test_rows: int
    training_time_ms: float

class RiskRegimeResponse(BaseModel):
    portfolio_id: int
    timestamp: datetime
    risk_regime: Optional[str] = None
    risk_regime_probability: Optional[float] = None
    predicted_var_forecast: Optional[float] = None
    is_anomaly: Optional[bool] = None

class RiskRegimeHistoryResponse(BaseModel):
    timestamp: datetime
    risk_regime: str
    risk_regime_probability: float
    predicted_var_forecast: Optional[float] = None
    is_anomaly: Optional[bool] = None
