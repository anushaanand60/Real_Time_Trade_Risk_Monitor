import os
os.environ["TESTING"] = "True"
import pytest
import json
import joblib
from decimal import Decimal
from datetime import datetime
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base, get_db
from app.main import app
from app.models import Portfolio, Trade, Position, Alert, FeatureSnapshot
from app.services.anomaly_detector import score_anomaly, FEATURES

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
    Base.metadata.create_all(bind=engine)
    mock_redis_store.clear()
    if os.path.exists("models/anomaly_detector_v1.joblib"):
        os.remove("models/anomaly_detector_v1.joblib")
    yield
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("models/anomaly_detector_v1.joblib"):
        os.remove("models/anomaly_detector_v1.joblib")

@pytest.fixture
def client():
    with patch("app.routers.portfolio.redis_get", side_effect=mock_redis_get):
        with patch("app.routers.portfolio.redis_set", side_effect=mock_redis_set):
            with patch("app.routers.trade.redis_delete", side_effect=mock_redis_delete):
                with TestClient(app) as c:
                    yield c

def test_anomaly_detection_pipeline(client):
    resp = client.post("/anomaly/train", json={"contamination": 0.01})
    assert resp.status_code == 400
    assert "Insufficient data for training" in resp.json()["detail"]
    
    sim_resp = client.post("/portfolios/1/simulate-data?num_trades=110")
    assert sim_resp.status_code == 200
    
    db = TestingSessionLocal()
    snapshots_count = db.query(FeatureSnapshot).filter(FeatureSnapshot.snapshot_type == "PORTFOLIO").count()
    assert snapshots_count >= 110
    
    train_resp = client.post("/anomaly/train", json={"contamination": 0.02})
    assert train_resp.status_code == 200
    train_data = train_resp.json()
    assert train_data["status"] == "success"
    assert train_data["training_rows"] >= 110
    assert train_data["contamination"] == 0.02
    
    assert os.path.exists("models/anomaly_detector_v1.joblib")
    payload = joblib.load("models/anomaly_detector_v1.joblib")
    assert "model" in payload
    assert "means" in payload
    assert "stds" in payload
    assert "trained_at" in payload
    assert payload["contamination"] == 0.02
    assert payload["training_rows"] >= 110
    
    pos_snapshot = db.query(FeatureSnapshot).filter(FeatureSnapshot.snapshot_type == "POSITION").first()
    if pos_snapshot:
        score_anomaly(db, pos_snapshot)
        assert pos_snapshot.anomaly_score is None
        assert pos_snapshot.is_anomaly is None
    
    t_normal = client.post("/trades", json={"portfolio_id": 1, "ticker": "AAPL", "quantity": "10", "price": "150.0000", "side": "BUY"})
    assert t_normal.status_code == 200
    
    latest_score_resp = client.get("/portfolios/1/anomaly/score")
    assert latest_score_resp.status_code == 200
    latest_score_data = latest_score_resp.json()
    assert "anomaly_score" in latest_score_data
    assert "is_anomaly" in latest_score_data
    
    t_anomaly = client.post("/trades", json={"portfolio_id": 1, "ticker": "AAPL", "quantity": "100", "price": "99999.0000", "side": "BUY"})
    assert t_anomaly.status_code == 200
    
    latest_score_resp = client.get("/portfolios/1/anomaly/score")
    latest_score_data = latest_score_resp.json()
    assert latest_score_data["is_anomaly"] is True
    assert latest_score_data["anomaly_explanation"] is not None
    assert isinstance(latest_score_data["anomaly_explanation"], list)
    assert len(latest_score_data["anomaly_explanation"]) <= 3
    
    history_resp = client.get("/portfolios/1/anomaly/history")
    assert history_resp.status_code == 200
    history_data = history_resp.json()
    assert len(history_data) >= 1
    assert any(h["is_anomaly"] is True for h in history_data)
    
    payload_reload = joblib.load("models/anomaly_detector_v1.joblib")
    assert payload_reload is not None
    assert "features" in payload_reload
    assert payload_reload["features"] == FEATURES
    assert "anomaly_count" in payload_reload
    assert "anomaly_rate" in payload_reload
    assert "training_time_ms" in payload_reload

    payload_mismatch = dict(payload_reload)
    payload_mismatch["features"] = ["different_feature"]
    joblib.dump(payload_mismatch, "models/anomaly_detector_v1.joblib")
    
    latest_snapshot = db.query(FeatureSnapshot).filter(
        FeatureSnapshot.portfolio_id == 1,
        FeatureSnapshot.snapshot_type == "PORTFOLIO"
    ).order_by(FeatureSnapshot.timestamp.desc(), FeatureSnapshot.id.desc()).first()
    
    with pytest.raises(ValueError) as exc:
        score_anomaly(db, latest_snapshot)
    assert "Model features mismatch" in str(exc.value)
    
    joblib.dump(payload_reload, "models/anomaly_detector_v1.joblib")
    score_anomaly(db, latest_snapshot)
    assert latest_snapshot.is_anomaly is not None
    
    db.close()
