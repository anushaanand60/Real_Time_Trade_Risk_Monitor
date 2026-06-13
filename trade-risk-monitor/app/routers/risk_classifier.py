from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.features import FeatureSnapshot
from app.schemas.risk_classifier import RiskClassifierTrainResponse, RiskRegimeResponse, RiskRegimeHistoryResponse
from app.services.risk_classifier import train_risk_classifier

router = APIRouter()

@router.post("/risk-classifier/train", response_model=RiskClassifierTrainResponse)
def train_classifier(db: Session = Depends(get_db)):
    try:
        res = train_risk_classifier(db)
        key_map = {
            "GradientBoosting": "gradient_boosting",
            "RandomForest": "random_forest",
            "LogisticRegression": "logistic_regression"
        }
        winner = res[key_map[res["selected_model"]]]
        return {
            "status": "success",
            "trained_at": res["trained_at"],
            "selected_model": res["selected_model"],
            "accuracy": winner["accuracy"],
            "macro_f1": winner["macro_f1"],
            "weighted_f1": winner["weighted_f1"],
            "confusion_matrix": res["confusion_matrix"],
            "class_counts": res["class_counts"],
            "top_features": res["top_features"],
            "training_rows": res["training_rows"],
            "test_rows": res["test_rows"],
            "training_time_ms": res["training_time_ms"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/portfolios/{portfolio_id}/risk-regime", response_model=RiskRegimeResponse)
def get_current_risk_regime(portfolio_id: int, db: Session = Depends(get_db)):
    snapshot = db.query(FeatureSnapshot).filter(
        FeatureSnapshot.portfolio_id == portfolio_id,
        FeatureSnapshot.snapshot_type == "PORTFOLIO"
    ).order_by(FeatureSnapshot.timestamp.desc(), FeatureSnapshot.id.desc()).first()
    if not snapshot:
        raise HTTPException(status_code=404, detail="No feature snapshots found for portfolio")
    return {
        "portfolio_id": portfolio_id,
        "timestamp": snapshot.timestamp,
        "risk_regime": snapshot.risk_regime,
        "risk_regime_probability": float(snapshot.risk_regime_probability) if snapshot.risk_regime_probability is not None else None,
        "predicted_var_forecast": float(snapshot.risk_var_forecast) if snapshot.risk_var_forecast is not None else None,
        "is_anomaly": snapshot.is_anomaly
    }

@router.get("/portfolios/{portfolio_id}/risk-regime/history", response_model=list[RiskRegimeHistoryResponse])
def get_risk_regime_history(portfolio_id: int, limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    snapshots = db.query(FeatureSnapshot).filter(
        FeatureSnapshot.portfolio_id == portfolio_id,
        FeatureSnapshot.snapshot_type == "PORTFOLIO",
        FeatureSnapshot.risk_regime.isnot(None)
    ).order_by(FeatureSnapshot.timestamp.desc(), FeatureSnapshot.id.desc()).limit(limit).all()
    return [
        {
            "timestamp": s.timestamp,
            "risk_regime": s.risk_regime,
            "risk_regime_probability": float(s.risk_regime_probability),
            "predicted_var_forecast": float(s.risk_var_forecast) if s.risk_var_forecast is not None else None,
            "is_anomaly": s.is_anomaly
        }
        for s in snapshots
    ]
