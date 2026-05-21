from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.trade import Trade
from app.schemas.trade import TradeCreate, TradeResponse
from app.services.position import update_position_logic

router = APIRouter()


@router.get("/portfolios/{portfolio_id}/trades", response_model=list[TradeResponse])
def get_trades(
    portfolio_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
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
    alerts = update_position_logic(db, trade)
    db.commit()
    db.refresh(trade)
    response = TradeResponse.model_validate(trade, from_attributes=True)
    response.alerts = alerts
    return response
