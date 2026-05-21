from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.portfolio import Portfolio
from app.models.position import Position
from app.schemas.portfolio import PortfolioCreate, PortfolioResponse
from app.schemas.position import PositionResponse

router = APIRouter()

@router.post("/portfolios", response_model=PortfolioResponse)
def create_portfolio(payload: PortfolioCreate, db: Session = Depends(get_db)):
    portfolio = Portfolio(name=payload.name)
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)
    return portfolio

@router.get("/portfolios/{portfolio_id}/positions", response_model=list[PositionResponse])
def get_positions(portfolio_id: int, db: Session = Depends(get_db)):
    positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
    return positions
