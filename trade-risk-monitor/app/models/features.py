from sqlalchemy import Column, Integer, String, DateTime, Numeric, ForeignKey, Index, Boolean
from datetime import datetime, timezone
from app.database import Base

class FeatureSnapshot(Base):
    __tablename__ = "feature_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    ticker = Column(String, nullable=True)
    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    snapshot_type = Column(String, nullable=False)

    exp_net_exposure = Column(Numeric(precision=18, scale=4), nullable=True)
    exp_gross_exposure = Column(Numeric(precision=18, scale=4), nullable=True)
    exp_weight = Column(Numeric(precision=18, scale=4), nullable=True)

    beh_trade_count_1h = Column(Integer, nullable=True)
    beh_avg_trade_size = Column(Numeric(precision=18, scale=4), nullable=True)
    beh_volume_1h = Column(Numeric(precision=18, scale=4), nullable=True)

    pos_net_quantity = Column(Numeric(precision=18, scale=4), nullable=True)
    pos_avg_price = Column(Numeric(precision=18, scale=4), nullable=True)
    pos_unrealized_pnl = Column(Numeric(precision=18, scale=4), nullable=True)

    risk_var_95 = Column(Numeric(precision=18, scale=4), nullable=True)
    risk_hhi_concentration = Column(Numeric(precision=18, scale=4), nullable=True)

    vol_rolling_volatility_5t = Column(Numeric(precision=18, scale=4), nullable=True)
    vol_rolling_volatility_30t = Column(Numeric(precision=18, scale=4), nullable=True)

    anomaly_score = Column(Numeric(precision=18, scale=4), nullable=True)
    is_anomaly = Column(Boolean, nullable=True)
    anomaly_explanation = Column(String, nullable=True)
    risk_var_forecast = Column(Numeric(precision=18, scale=4), nullable=True)
    risk_regime = Column(String, nullable=True)
    risk_regime_probability = Column(Numeric(precision=18, scale=4), nullable=True)

    __table_args__ = (
        Index("idx_portfolio_timestamp", "portfolio_id", "timestamp"),
        Index("idx_portfolio_ticker_timestamp", "portfolio_id", "ticker", "timestamp"),
    )
