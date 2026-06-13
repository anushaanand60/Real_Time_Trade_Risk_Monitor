import os
os.environ["TESTING"] = "True"
import pytest
from decimal import Decimal
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base, get_db
from app.main import app

SQLALCHEMY_TEST_URL = "sqlite:///./test_risk_engine.db"
engine = create_engine(SQLALCHEMY_TEST_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

mock_redis_store = {}

def mock_redis_get(key):
    return mock_redis_store.get(key)

def mock_redis_set(key, value, ttl=30):
    mock_redis_store[key] = value

def mock_redis_delete(key):
    mock_redis_store.pop(key, None)

@pytest.fixture(autouse=True)
def setup_db():
    print("RISK KEYS:", Base.metadata.tables.keys())
    Base.metadata.create_all(bind=engine)
    mock_redis_store.clear()
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client():
    with patch("app.routers.portfolio.redis_get", side_effect=mock_redis_get):
        with patch("app.routers.portfolio.redis_set", side_effect=mock_redis_set):
            with patch("app.routers.trade.redis_delete", side_effect=mock_redis_delete):
                with TestClient(app) as c:
                    yield c

def test_position_increments_decrements(client):
    resp = client.post("/portfolios", json={"name": "TestPortfolio"})
    pid = resp.json()["id"]
    client.post("/trades", json={"portfolio_id": pid, "ticker": "AAPL", "quantity": "100", "price": "150.0000", "side": "BUY"})
    client.post("/trades", json={"portfolio_id": pid, "ticker": "AAPL", "quantity": "50", "price": "160.0000", "side": "BUY"})
    positions = client.get(f"/portfolios/{pid}/positions").json()
    assert len(positions) == 1
    assert Decimal(positions[0]["net_quantity"]) == Decimal("150.0000")
    client.post("/trades", json={"portfolio_id": pid, "ticker": "AAPL", "quantity": "30", "price": "170.0000", "side": "SELL"})
    positions = client.get(f"/portfolios/{pid}/positions").json()
    assert Decimal(positions[0]["net_quantity"]) == Decimal("120.0000")
    client.post("/trades", json={"portfolio_id": pid, "ticker": "AAPL", "quantity": "120", "price": "180.0000", "side": "SELL"})
    positions = client.get(f"/portfolios/{pid}/positions").json()
    assert Decimal(positions[0]["net_quantity"]) == Decimal("0.0000")

def test_cache_hit_miss(client):
    resp = client.post("/portfolios", json={"name": "CacheTest"})
    pid = resp.json()["id"]
    client.post("/trades", json={"portfolio_id": pid, "ticker": "GOOG", "quantity": "10", "price": "100.0000", "side": "BUY"})
    r1 = client.get(f"/portfolios/{pid}/positions")
    assert r1.headers["X-Cache"] == "MISS"
    r2 = client.get(f"/portfolios/{pid}/positions")
    assert r2.headers["X-Cache"] == "HIT"

def test_concentration_breach_alert(client):
    resp = client.post("/portfolios", json={"name": "ConcentrationTest"})
    pid = resp.json()["id"]
    client.post("/trades", json={"portfolio_id": pid, "ticker": "MSFT", "quantity": "10", "price": "100.0000", "side": "BUY"})
    client.post("/trades", json={"portfolio_id": pid, "ticker": "TSLA", "quantity": "10", "price": "100.0000", "side": "BUY"})
    r = client.post("/trades", json={"portfolio_id": pid, "ticker": "MSFT", "quantity": "90", "price": "100.0000", "side": "BUY"})
    alerts = r.json()["alerts"]
    concentration_alerts = [a for a in alerts if a["alert_type"] == "CONCENTRATION_BREACH"]
    assert len(concentration_alerts)>=1
    assert "MSFT" in concentration_alerts[0]["message"]

def test_var_insufficient_data(client):
    resp = client.post("/portfolios", json={"name": "VarTest"})
    pid = resp.json()["id"]
    for i in range(5):
        client.post("/trades", json={"portfolio_id": pid, "ticker": "AMZN", "quantity": "10", "price": str(100 + i), "side": "BUY"})
    r = client.get(f"/portfolios/{pid}/var")
    data = r.json()
    assert data["insufficient_data"] is True

def test_avg_price_precision_multiple_buys(client):
    resp = client.post("/portfolios", json={"name": "PrecisionTest"})
    pid = resp.json()["id"]
    client.post("/trades", json={"portfolio_id": pid, "ticker": "NVDA", "quantity": "33", "price": "127.3300", "side": "BUY"})
    client.post("/trades", json={"portfolio_id": pid, "ticker": "NVDA", "quantity": "17", "price": "213.7700", "side": "BUY"})
    client.post("/trades", json={"portfolio_id": pid, "ticker": "NVDA", "quantity": "50", "price": "99.1234", "side": "BUY"})
    positions = client.get(f"/portfolios/{pid}/positions").json()
    nvda = [p for p in positions if p["ticker"] == "NVDA"][0]
    net_qty = Decimal("100")
    expected_avg = (Decimal("33")*Decimal("127.3300")+Decimal("17")*Decimal("213.7700")+Decimal("50")*Decimal("99.1234"))/net_qty
    assert Decimal(nvda["net_quantity"]) == net_qty
    actual_avg = Decimal(nvda["avg_price"])
    assert abs(actual_avg-expected_avg)<Decimal("0.0001")