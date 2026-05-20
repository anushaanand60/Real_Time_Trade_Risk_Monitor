from sqlalchemy import Column, Integer, String, DateTime, Numeric, ForeignKey
from datetime import datetime, timezone
from app.database import Base


class Position(Base):
    __tablename__ = "positions"

    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), primary_key=True)
    ticker = Column(String, primary_key=True)
    net_quantity = Column(Numeric(precision=18, scale=4), nullable=False)
    avg_price = Column(Numeric(precision=18, scale=4), nullable=False)
    unrealized_pnl = Column(Numeric(precision=18, scale=4), nullable=False)
    last_updated = Column(DateTime, default=lambda: datetime.now(timezone.utc))
