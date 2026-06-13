from decimal import Decimal
from sqlalchemy.orm import Session
from app.models.position import Position
from app.models.alert import Alert
from app.models.trade import Trade
from app.services.risk import compute_portfolio_var
from app.core.config import settings

def check_concentration(db: Session, trade: Trade) -> list[Alert]:
    alerts = []
    positions = db.query(Position).filter(Position.portfolio_id == trade.portfolio_id).all()
    total_value = sum(abs(p.net_quantity*p.avg_price) for p in positions)
    if total_value == Decimal("0"):
        return alerts
    for p in positions:
        position_value = abs(p.net_quantity*p.avg_price)
        concentration = position_value/total_value
        if concentration >settings.CONCENTRATION_THRESHOLD:
            alert = Alert(
                portfolio_id=trade.portfolio_id,
                alert_type="CONCENTRATION_BREACH",
                message=f"{p.ticker} concentration {concentration:.4f} exceeds threshold {settings.CONCENTRATION_THRESHOLD}",
            )
            db.add(alert)
            alerts.append(alert)
    return alerts

def check_var_breach(db: Session, trade: Trade) -> list[Alert]:
    alerts = []
    result = compute_portfolio_var(db, trade.portfolio_id, as_of=trade.timestamp)
    if not result["insufficient_data"] and result["var_value"] > settings.VAR_THRESHOLD:
        alert = Alert(
            portfolio_id=trade.portfolio_id,
            alert_type="VAR_BREACH",
            message=f"Portfolio VaR {result['var_value']} exceeds threshold {settings.VAR_THRESHOLD}",
        )
        db.add(alert)
        alerts.append(alert)
    return alerts

def run_post_trade_alerts(db: Session, trade: Trade) -> list[Alert]:
    alerts = []
    alerts.extend(check_concentration(db, trade))
    alerts.extend(check_var_breach(db, trade))
    return alerts
