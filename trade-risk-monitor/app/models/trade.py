from sqlalchemy import Column, Integer, String, DateTime, Numeric, ForeignKey, CheckConstraint
from datetime import datetime, timezone
from app.database import Base

class Trade(Base):
    __tablename__ = "trades"
    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    ticker = Column(String, nullable=False, index=True)
    quantity = Column(Numeric(precision=18, scale=4), nullable=False)
    price = Column(Numeric(precision=18, scale=4), nullable=False)
    side = Column(String, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        CheckConstraint("side IN ('BUY', 'SELL')", name="check_side_value"),
    )
