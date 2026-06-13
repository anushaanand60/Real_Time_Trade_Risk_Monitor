from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class VarForecastTrainResponse(BaseModel):
    status: str
    trained_at: str
    model_type: str
    mae: float
    rmse: float
    r2: float
    outperformed_baseline: bool
    top_features: list[str]
    training_rows: int
    test_rows: int
    training_time_ms: float

class VarForecastResponse(BaseModel):
    portfolio_id: int
    timestamp: datetime
    historical_var_95: float
    predicted_var_forecast: Optional[float] = None
