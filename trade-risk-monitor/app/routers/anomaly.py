from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.features import FeatureSnapshot
from app.schemas.anomaly import AnomalyTrainRequest, AnomalyTrainResponse, AnomalyScoreResponse, AnomalyHistoryResponse
from app.services.anomaly_detector import train_global_anomaly_model
from app.services.market_simulator import generate_simulation_data

router = APIRouter()

@router.post("/anomaly/train", response_model=AnomalyTrainResponse)
def train_anomaly(payload: AnomalyTrainRequest, db: Session = Depends(get_db)):
    try:
        res = train_global_anomaly_model(db, contamination=payload.contamination)
        return {
            "status": "success",
            "trained_at": res["trained_at"],
            "contamination": res["contamination"],
            "training_rows": res["training_rows"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/portfolios/{portfolio_id}/simulate-data")
def simulate_portfolio_data(portfolio_id: int, num_trades: int = Query(100, ge=1, le=1000), db: Session = Depends(get_db)):
    generate_simulation_data(db, portfolio_id, num_trades)
    return {"status": "success", "message": f"Generated {num_trades} trades for portfolio {portfolio_id}"}

@router.get("/portfolios/{portfolio_id}/anomaly/score", response_model=AnomalyScoreResponse)
def get_latest_anomaly_score(portfolio_id: int, db: Session = Depends(get_db)):
    snapshot = db.query(FeatureSnapshot).filter(
        FeatureSnapshot.portfolio_id == portfolio_id,
        FeatureSnapshot.snapshot_type == "PORTFOLIO"
    ).order_by(FeatureSnapshot.timestamp.desc(), FeatureSnapshot.id.desc()).first()
    if not snapshot:
        raise HTTPException(status_code=404, detail="No feature snapshots found for portfolio")
    return snapshot

@router.get("/portfolios/{portfolio_id}/anomaly/history", response_model=list[AnomalyHistoryResponse])
def get_anomaly_history(portfolio_id: int, limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    snapshots = db.query(FeatureSnapshot).filter(
        FeatureSnapshot.portfolio_id == portfolio_id,
        FeatureSnapshot.snapshot_type == "PORTFOLIO",
        FeatureSnapshot.is_anomaly == True
    ).order_by(FeatureSnapshot.timestamp.desc(), FeatureSnapshot.id.desc()).limit(limit).all()
    return snapshots
