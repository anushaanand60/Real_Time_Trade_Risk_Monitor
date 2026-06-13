from pydantic import BaseModel
from decimal import Decimal
from datetime import datetime

class PositionResponse(BaseModel):
    portfolio_id: int
    ticker: str
    net_quantity: Decimal
    avg_price: Decimal
    unrealized_pnl: Decimal
    last_updated: datetime
    model_config = {"from_attributes": True}
