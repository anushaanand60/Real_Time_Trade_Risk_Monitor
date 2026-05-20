from decimal import Decimal
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from app.models.trade import Trade
from app.models.position import Position
from app.models.alert import Alert


def update_position_logic(db: Session, trade: Trade) -> list[Alert]:
    position = db.query(Position).filter(
        Position.portfolio_id == trade.portfolio_id,
        Position.ticker == trade.ticker,
    ).first()

    if position is None and trade.side == "BUY":
        position = Position(
            portfolio_id=trade.portfolio_id,
            ticker=trade.ticker,
            net_quantity=Decimal("0.0000"),
            avg_price=Decimal("0.0000"),
            unrealized_pnl=Decimal("0.0000"),
        )
        db.add(position)

    if trade.side == "BUY":
        new_net_qty = position.net_quantity + trade.quantity
        new_avg_price = (
            (position.net_quantity * position.avg_price) + (trade.quantity * trade.price)
        ) / new_net_qty
        position.net_quantity = new_net_qty
        position.avg_price = new_avg_price
        position.last_updated = datetime.now(timezone.utc)

    elif trade.side == "SELL":
        new_net_qty = position.net_quantity - trade.quantity
        unrealized_pnl = (trade.price - position.avg_price) * new_net_qty
        position.net_quantity = new_net_qty
        position.unrealized_pnl = unrealized_pnl
        position.last_updated = datetime.now(timezone.utc)

    return []
