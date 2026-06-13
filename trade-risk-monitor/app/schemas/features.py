from pydantic import BaseModel
from decimal import Decimal
from datetime import datetime
from typing import Optional

class FeatureSnapshotResponse(BaseModel):
    id: int
    portfolio_id: int
    ticker: Optional[str] = None
    trade_id: Optional[int] = None
    timestamp: datetime
    snapshot_type: str
    exp_net_exposure: Optional[Decimal] = None
    exp_gross_exposure: Optional[Decimal] = None
    exp_weight: Optional[Decimal] = None
    beh_trade_count_1h: Optional[int] = None
    beh_avg_trade_size: Optional[Decimal] = None
    beh_volume_1h: Optional[Decimal] = None
    pos_net_quantity: Optional[Decimal] = None
    pos_avg_price: Optional[Decimal] = None
    pos_unrealized_pnl: Optional[Decimal] = None
    risk_var_95: Optional[Decimal] = None
    risk_hhi_concentration: Optional[Decimal] = None
    vol_rolling_volatility_5t: Optional[Decimal] = None
    vol_rolling_volatility_30t: Optional[Decimal] = None
    model_config = {"from_attributes": True}
