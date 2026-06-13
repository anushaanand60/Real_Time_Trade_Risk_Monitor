import os
os.environ["TESTING"] = "True"
import pytest
import joblib
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base, get_db
from app.main import app
from app.models.features import FeatureSnapshot
from app.services.var_forecaster import predict_var_forecast, FEATURES

# Use test database
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
    if os.path.exists("models/var_forecaster_v1.joblib"):
        os.remove("models/var_forecaster_v1.joblib")
    yield
    Base.metadata.drop_all(bind=engine)
    if os.path.exists("models/var_forecaster_v1.joblib"):
        os.remove("models/var_forecaster_v1.joblib")

@pytest.fixture
def client():
    with patch("app.routers.portfolio.redis_get", side_effect=mock_redis_get):
        with patch("app.routers.portfolio.redis_set", side_effect=mock_redis_set):
            with patch("app.routers.trade.redis_delete", side_effect=mock_redis_delete):
                with TestClient(app) as c:
                    yield c

def test_var_forecaster_pipeline(client):
    resp = client.post("/var/train")
    assert resp.status_code == 400
    assert "Insufficient data for training" in resp.json()["detail"]

    sim_resp = client.post("/portfolios/1/simulate-data?num_trades=310")
    assert sim_resp.status_code == 200

    db = TestingSessionLocal()
    snapshots_count = db.query(FeatureSnapshot).filter(FeatureSnapshot.snapshot_type == "PORTFOLIO").count()
    assert snapshots_count >= 310

    train_resp = client.post("/var/train")
    assert train_resp.status_code == 200
    train_data = train_resp.json()
    assert train_data["status"] == "success"
    assert "trained_at" in train_data
    assert "model_type" in train_data
    assert "mae" in train_data
    assert "rmse" in train_data
    assert "r2" in train_data
    assert "outperformed_baseline" in train_data
    assert "top_features" in train_data
    assert train_data["training_rows"] > 0
    assert train_data["test_rows"] > 0
    assert train_data["training_time_ms"] > 0

    assert os.path.exists("models/var_forecaster_v1.joblib")
    payload = joblib.load("models/var_forecaster_v1.joblib")
    assert "model" in payload
    assert "trained_at" in payload
    assert payload["features"] == FEATURES
    assert "model_type" in payload
    assert "selected_model" in payload
    assert "mae" in payload
    assert "rmse" in payload
    assert "r2" in payload
    assert "random_forest" in payload
    assert "gradient_boosting" in payload
    assert "baseline_metrics" in payload
    assert "outperformed_baseline" in payload
    assert "feature_importances" in payload
    assert "top_features" in payload
    assert payload["training_rows"] > 0
    assert payload["test_rows"] > 0
    assert payload["training_time_ms"] > 0

    importances = payload["feature_importances"]
    sorted_features = payload["top_features"]
    assert len(importances) == len(FEATURES)
    assert len(sorted_features) == len(FEATURES)
    for i in range(len(sorted_features) - 1):
        assert importances[sorted_features[i]] >= importances[sorted_features[i + 1]]

    t_trade = client.post("/trades", json={"portfolio_id": 1, "ticker": "AAPL", "quantity": "10", "price": "150.0000", "side": "BUY"})
    assert t_trade.status_code == 200

    latest_snapshot = db.query(FeatureSnapshot).filter(
        FeatureSnapshot.portfolio_id == 1,
        FeatureSnapshot.snapshot_type == "PORTFOLIO"
    ).order_by(FeatureSnapshot.timestamp.desc(), FeatureSnapshot.id.desc()).first()
    assert latest_snapshot.risk_var_forecast is not None

    forecast_resp = client.get("/portfolios/1/var/forecast")
    assert forecast_resp.status_code == 200
    forecast_data = forecast_resp.json()
    assert forecast_data["portfolio_id"] == 1
    assert "timestamp" in forecast_data
    assert "historical_var_95" in forecast_data
    assert forecast_data["predicted_var_forecast"] is not None

    payload_mismatch = dict(payload)
    payload_mismatch["features"] = ["mismatch_feature"]
    joblib.dump(payload_mismatch, "models/var_forecaster_v1.joblib")

    with pytest.raises(ValueError) as exc:
        predict_var_forecast(db, latest_snapshot)
    assert "Model features mismatch" in str(exc.value)

    db.close()
