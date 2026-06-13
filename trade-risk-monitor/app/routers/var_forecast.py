from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.features import FeatureSnapshot
from app.schemas.var_forecast import VarForecastTrainResponse, VarForecastResponse
from app.services.var_forecaster import train_var_forecaster

router = APIRouter()

@router.post("/var/train", response_model=VarForecastTrainResponse)
def train_forecast(db: Session = Depends(get_db)):
    try:
        res = train_var_forecaster(db)
        return {
            "status": "success",
            "trained_at": res["trained_at"],
            "model_type": res["model_type"],
            "mae": res["mae"],
            "rmse": res["rmse"],
            "r2": res["r2"],
            "outperformed_baseline": res["outperformed_baseline"],
            "top_features": res["top_features"],
            "training_rows": res["training_rows"],
            "test_rows": res["test_rows"],
            "training_time_ms": res["training_time_ms"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/portfolios/{portfolio_id}/var/forecast", response_model=VarForecastResponse)
def get_latest_var_forecast(portfolio_id: int, db: Session = Depends(get_db)):
    snapshot = db.query(FeatureSnapshot).filter(
        FeatureSnapshot.portfolio_id == portfolio_id,
        FeatureSnapshot.snapshot_type == "PORTFOLIO"
    ).order_by(FeatureSnapshot.timestamp.desc(), FeatureSnapshot.id.desc()).first()
    if not snapshot:
        raise HTTPException(status_code=404, detail="No feature snapshots found for portfolio")
    return {
        "portfolio_id": portfolio_id,
        "timestamp": snapshot.timestamp,
        "historical_var_95": float(snapshot.risk_var_95 or 0.0),
        "predicted_var_forecast": float(snapshot.risk_var_forecast) if snapshot.risk_var_forecast is not None else None
    }
