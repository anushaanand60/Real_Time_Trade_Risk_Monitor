from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from datetime import datetime, timezone
from app.database import Base


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    alert_type = Column(String, nullable=False)
    message = Column(String, nullable=False)
    triggered_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
