from decimal import Decimal
from sqlalchemy.orm import Session
from sqlalchemy import func, cast, Date
import numpy as np
from app.models.trade import Trade

def compute_portfolio_var(db: Session, portfolio_id: int, confidence: Decimal = Decimal("0.95"), window: int = 30) -> dict:
    daily_values = (
        db.query(
            cast(Trade.timestamp, Date).label("trade_date"),
            func.sum(
                func.case(
                    (Trade.side == "BUY", Trade.quantity * Trade.price),
                    else_=-Trade.quantity * Trade.price,
                )
            ).label("daily_value"),
        )
        .filter(Trade.portfolio_id == portfolio_id)
        .group_by(cast(Trade.timestamp, Date))
        .order_by(cast(Trade.timestamp, Date))
        .limit(window)
        .all()
    )
    if len(daily_values) < 2:
        return {
            "portfolio_id": portfolio_id,
            "var_value": Decimal("0.0000"),
            "confidence_level": confidence,
            "window_days": window,
            "insufficient_data": True,
        }
    values = [float(row.daily_value) for row in daily_values]
    daily_pnl_changes = [values[i] - values[i - 1] for i in range(1, len(values))]
    if len(daily_pnl_changes) < 10:
        return {
            "portfolio_id": portfolio_id,
            "var_value": Decimal("0.0000"),
            "confidence_level": confidence,
            "window_days": window,
            "insufficient_data": True,
        }
    pnl_array = np.array(daily_pnl_changes)
    percentile_value = float(np.percentile(pnl_array, (1 - float(confidence)) * 100))
    var_value = abs(Decimal(str(round(percentile_value, 4))))
    return {
        "portfolio_id": portfolio_id,
        "var_value": var_value,
        "confidence_level": confidence,
        "window_days": window,
        "insufficient_data": False,
    }
