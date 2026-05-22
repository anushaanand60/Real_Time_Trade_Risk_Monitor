import random
from decimal import Decimal, ROUND_HALF_UP
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.trade import Trade
from app.models.position import Position
from app.models.alert import Alert
from app.models.portfolio import Portfolio
from app.services.position import update_position_logic
from app.services.alerts import run_post_trade_alerts
from app.services.risk import compute_portfolio_var
from app.core.redis import redis_delete

router = APIRouter()
TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

@router.post("/portfolios/{portfolio_id}/simulate")
def simulate_trades(portfolio_id: int, db: Session = Depends(get_db)):
    db.query(Alert).filter(Alert.portfolio_id == portfolio_id).delete()
    db.query(Trade).filter(Trade.portfolio_id == portfolio_id).delete()
    db.query(Position).filter(Position.portfolio_id == portfolio_id).delete()
    db.flush()
    if not db.query(Portfolio).filter(Portfolio.id == portfolio_id).first():
        db.add(Portfolio(id=portfolio_id, name=f"SimPortfolio-{portfolio_id}"))
        db.commit()
    redis_delete(f"portfolio:{portfolio_id}:positions")
    triggered_alerts = []
    for _ in range(100):
        ticker = random.choice(TICKERS)
        side = "BUY" if random.random() < 0.70 else "SELL"
        quantity = Decimal(str(random.randint(1, 100)))
        price = Decimal(str(random.uniform(50.0, 500.0))).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )
        if side == "SELL":
            existing = db.query(Position).filter(
                Position.portfolio_id == portfolio_id,
                Position.ticker == ticker,
            ).first()
            if existing is None or existing.net_quantity < quantity:
                side = "BUY"
        trade = Trade(
            portfolio_id=portfolio_id,
            ticker=ticker,
            quantity=quantity,
            price=price,
            side=side,
        )
        db.add(trade)
        db.flush()
        position_alerts = update_position_logic(db, trade)
        risk_alerts = run_post_trade_alerts(db, trade)
        triggered_alerts.extend(
            [{"alert_type": a.alert_type, "message": a.message} for a in position_alerts + risk_alerts]
        )
    db.commit()
    redis_delete(f"portfolio:{portfolio_id}:positions")
    positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
    position_data = []
    net_worth = Decimal("0.0000")
    for p in positions:
        pos_value = p.net_quantity * p.avg_price
        net_worth += pos_value
        position_data.append({
            "ticker": p.ticker,
            "net_quantity": str(p.net_quantity),
            "avg_price": str(p.avg_price),
            "unrealized_pnl": str(p.unrealized_pnl),
        })
    var_result = compute_portfolio_var(db, portfolio_id)
    return {
        "portfolio_id": portfolio_id,
        "trades_executed": 100,
        "positions": position_data,
        "net_worth": str(net_worth.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)),
        "var": {
            "var_value": str(var_result["var_value"]),
            "confidence_level": str(var_result["confidence_level"]),
            "window_days": var_result["window_days"],
            "insufficient_data": var_result["insufficient_data"],
        },
        "alerts_triggered": triggered_alerts,
    }
