from fastapi import APIRouter

router = APIRouter()

@router.post("/portfolios/{portfolio_id}/simulate")
def simulate_trades(portfolio_id: int):
    return {"status": "simulation stub", "portfolio_id": portfolio_id}
