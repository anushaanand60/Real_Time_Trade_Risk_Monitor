from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.trade import Trade
from app.models.features import FeatureSnapshot
from app.schemas.trade import TradeCreate, TradeResponse
from app.services.position import update_position_logic
from app.services.alerts import run_post_trade_alerts
from app.services.feature_generation import generate_features_for_trade
from app.services.anomaly_detector import score_anomaly
from app.services.var_forecaster import predict_var_forecast
from app.services.risk_classifier import predict_risk_regime
from app.core.redis import redis_delete

router = APIRouter()

@router.get("/portfolios/{portfolio_id}/trades", response_model=list[TradeResponse])
def get_trades(portfolio_id: int, skip: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)):
    trades = (
        db.query(Trade)
        .filter(Trade.portfolio_id == portfolio_id)
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [TradeResponse.model_validate(t, from_attributes=True) for t in trades]

@router.post("/trades", response_model=TradeResponse)
def create_trade(payload: TradeCreate, db: Session = Depends(get_db)):
    trade = Trade(
        portfolio_id=payload.portfolio_id,
        ticker=payload.ticker,
        quantity=payload.quantity,
        price=payload.price,
        side=payload.side,
    )
    db.add(trade)
    db.flush()
    position_alerts = update_position_logic(db, trade)
    risk_alerts = run_post_trade_alerts(db, trade)
    all_alerts = position_alerts + risk_alerts
    snapshot = generate_features_for_trade(db, trade)
    score_anomaly(db, snapshot)
    predict_var_forecast(db, snapshot)
    predict_risk_regime(db, snapshot)
    db.commit()
    db.refresh(trade)
    redis_delete(f"portfolio:{trade.portfolio_id}:positions")
    response = TradeResponse.model_validate(trade, from_attributes=True)
    response.alerts = [{"alert_type": a.alert_type, "message": a.message} for a in all_alerts]
    return response
