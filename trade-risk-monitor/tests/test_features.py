import os
os.environ["TESTING"] = "True"
import pytest
from decimal import Decimal
from datetime import datetime
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base, get_db
from app.main import app
from app.models import Portfolio, Trade, Position, Alert, FeatureSnapshot

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
    print("KEYS:", Base.metadata.tables.keys())
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

def test_feature_generation_and_api(client):
    resp = client.post("/portfolios", json={"name": "FeatureTest"})
    pid = resp.json()["id"]
    
    t1 = client.post("/trades", json={"portfolio_id": pid, "ticker": "AAPL", "quantity": "100", "price": "150.0000", "side": "BUY"})
    assert t1.status_code == 200
    
    latest_port = client.get(f"/portfolios/{pid}/features/latest?snapshot_type=PORTFOLIO").json()
    assert latest_port["snapshot_type"] == "PORTFOLIO"
    assert latest_port["ticker"] is None
    assert Decimal(latest_port["exp_net_exposure"]) == Decimal("15000.0000")
    assert Decimal(latest_port["exp_gross_exposure"]) == Decimal("15000.0000")
    assert Decimal(latest_port["exp_weight"]) == Decimal("1.0000")
    assert latest_port["beh_trade_count_1h"] == 1
    assert Decimal(latest_port["beh_avg_trade_size"]) == Decimal("100.0000")
    assert Decimal(latest_port["beh_volume_1h"]) == Decimal("15000.0000")
    
    latest_pos = client.get(f"/portfolios/{pid}/features/latest?ticker=AAPL").json()
    assert latest_pos["snapshot_type"] == "POSITION"
    assert latest_pos["ticker"] == "AAPL"
    assert Decimal(latest_pos["pos_net_quantity"]) == Decimal("100.0000")
    assert Decimal(latest_pos["pos_avg_price"]) == Decimal("150.0000")
    assert Decimal(latest_pos["exp_weight"]) == Decimal("1.0000")
    
    client.post("/trades", json={"portfolio_id": pid, "ticker": "AAPL", "quantity": "50", "price": "160.0000", "side": "BUY"})
    client.post("/trades", json={"portfolio_id": pid, "ticker": "MSFT", "quantity": "100", "price": "200.0000", "side": "BUY"})
    
    latest_port = client.get(f"/portfolios/{pid}/features/latest?snapshot_type=PORTFOLIO").json()
    assert Decimal(latest_port["exp_gross_exposure"]) == Decimal("44000.0000")
    expected_hhi = ((Decimal("24000.0000") / Decimal("44000.0000"))**2 + (Decimal("20000.0000") / Decimal("44000.0000"))**2).quantize(Decimal("0.0001"))
    assert Decimal(latest_port["risk_hhi_concentration"]) == expected_hhi
    
    history_port = client.get(f"/portfolios/{pid}/features/history?snapshot_type=PORTFOLIO").json()
    assert len(history_port) == 3
    
    history_aapl = client.get(f"/portfolios/{pid}/features/history?ticker=AAPL").json()
    assert len(history_aapl) == 3
    
    history_msft = client.get(f"/portfolios/{pid}/features/history?ticker=MSFT").json()
    assert len(history_msft) == 1
