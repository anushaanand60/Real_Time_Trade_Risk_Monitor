import os
from fastapi import FastAPI
from app.database import engine, Base
from app.routers.portfolio import router as portfolio_router
from app.routers.trade import router as trade_router

if os.getenv("TESTING") != "True":
    Base.metadata.create_all(bind=engine)

app = FastAPI()
app.include_router(portfolio_router)
app.include_router(trade_router)

@app.get("/health")
def health_check():
    return {"status": "ok"}
