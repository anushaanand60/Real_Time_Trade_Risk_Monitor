import json
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.portfolio import Portfolio
from app.models.position import Position
from app.schemas.portfolio import PortfolioCreate, PortfolioResponse
from app.schemas.position import PositionResponse
from app.core.redis import redis_get, redis_set
from app.services.risk import compute_portfolio_var

router = APIRouter()

@router.post("/portfolios", response_model=PortfolioResponse)
def create_portfolio(payload: PortfolioCreate, db: Session = Depends(get_db)):
    portfolio = Portfolio(name=payload.name)
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)
    return portfolio

@router.get("/portfolios/{portfolio_id}/positions")
def get_positions(portfolio_id: int, db: Session = Depends(get_db)):
    cache_key = f"portfolio:{portfolio_id}:positions"
    cached = redis_get(cache_key)
    if cached is not None:
        return JSONResponse(content=json.loads(cached), headers={"X-Cache": "HIT"})
    positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
    result = [PositionResponse.model_validate(p, from_attributes=True).model_dump(mode="json") for p in positions]
    redis_set(cache_key, json.dumps(result, default=str))
    return JSONResponse(content=result, headers={"X-Cache": "MISS"})

@router.get("/portfolios/{portfolio_id}/var")
def get_portfolio_var(portfolio_id: int, db: Session = Depends(get_db)):
    result = compute_portfolio_var(db, portfolio_id)
    return {
        "portfolio_id": result["portfolio_id"],
        "var_value": str(result["var_value"]),
        "confidence_level": str(result["confidence_level"]),
        "window_days": result["window_days"],
        "insufficient_data": result["insufficient_data"],
    }
