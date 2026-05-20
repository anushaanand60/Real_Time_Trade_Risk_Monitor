from pydantic import BaseModel
from decimal import Decimal
from datetime import datetime
from typing import Any


class TradeCreate(BaseModel):
    portfolio_id: int
    ticker: str
    quantity: Decimal
    price: Decimal
    side: str


class TradeResponse(BaseModel):
    id: int
    portfolio_id: int
    ticker: str
    quantity: Decimal
    price: Decimal
    side: str
    timestamp: datetime
    alerts: list[Any] = []

    model_config = {"from_attributes": True}
