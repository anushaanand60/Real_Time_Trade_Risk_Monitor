import os
os.environ["TESTING"] = "True"
import pytest

pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
import joblib
import random
import numpy as np
from decimal import Decimal
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base, get_db
from app.main import app
from app.models import Portfolio, Trade, Position, Alert, FeatureSnapshot
from app.services.position import update_position_logic
from app.services.alerts import run_post_trade_alerts
from app.services.feature_generation import generate_features_for_trade
from app.services.anomaly_detector import score_anomaly
from app.services.var_forecaster import predict_var_forecast
from app.services.risk_classifier import predict_risk_regime, FEATURES

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
    for model_file in ["models/risk_classifier_v1.joblib", "models/var_forecaster_v1.joblib"]:
        if os.path.exists(model_file):
            os.remove(model_file)
    yield
    Base.metadata.drop_all(bind=engine)
    for model_file in ["models/risk_classifier_v1.joblib", "models/var_forecaster_v1.joblib"]:
        if os.path.exists(model_file):
            os.remove(model_file)

@pytest.fixture
def client():
    with patch("app.routers.portfolio.redis_get", side_effect=mock_redis_get):
        with patch("app.routers.portfolio.redis_set", side_effect=mock_redis_set):
            with patch("app.routers.trade.redis_delete", side_effect=mock_redis_delete):
                with TestClient(app) as c:
                    yield c

def test_risk_classifier_pipeline(client):
    resp = client.post("/risk-classifier/train")
    assert resp.status_code == 400
    assert "Insufficient data for training" in resp.json()["detail"]

    db = TestingSessionLocal()
    if not db.query(Portfolio).filter(Portfolio.id == 1).first():
        db.add(Portfolio(id=1, name="TestPortfolio"))
        db.commit()

    start_time = datetime.now(timezone.utc) - timedelta(days=320)
    for i in range(320):
        ticker = random.choice(["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"])
        side = "BUY" if random.random() < 0.70 else "SELL"
        quantity = Decimal(str(random.randint(10, 100)))
        price = Decimal(str(random.uniform(100.0, 300.0)))
        if side == "SELL":
            existing = db.query(Position).filter(Position.portfolio_id == 1, Position.ticker == ticker).first()
            if existing is None or existing.net_quantity < quantity:
                side = "BUY"
        trade = Trade(
            portfolio_id=1,
            ticker=ticker,
            quantity=quantity,
            price=price,
            side=side,
            timestamp=start_time + timedelta(days=i, minutes=random.randint(0, 60))
        )
        db.add(trade)
        db.flush()
        update_position_logic(db, trade)
        run_post_trade_alerts(db, trade)
        snapshot = generate_features_for_trade(db, trade)
        score_anomaly(db, snapshot)
        predict_var_forecast(db, snapshot)
    db.commit()

    snapshots_count = db.query(FeatureSnapshot).filter(FeatureSnapshot.snapshot_type == "PORTFOLIO").count()
    assert snapshots_count >= 320

    var_train_resp = client.post("/var/train")
    assert var_train_resp.status_code == 200, f"VaR training failed: {var_train_resp.json()}"
    assert var_train_resp.json()["status"] == "success"

    train_resp = client.post("/risk-classifier/train")
    assert train_resp.status_code == 200
    train_data = train_resp.json()
    assert train_data["status"] == "success"
    assert "trained_at" in train_data
    assert "selected_model" in train_data
    assert "accuracy" in train_data
    assert "macro_f1" in train_data
    assert "weighted_f1" in train_data
    assert "confusion_matrix" in train_data
    assert "class_counts" in train_data
    assert "top_features" in train_data
    assert train_data["training_rows"] > 0
    assert train_data["test_rows"] > 0
    assert train_data["training_time_ms"] > 0

    assert os.path.exists("models/risk_classifier_v1.joblib")
    payload = joblib.load("models/risk_classifier_v1.joblib")
    assert "model" in payload
    assert "trained_at" in payload
    assert payload["features"] == FEATURES
    assert "selected_model" in payload
    assert "logistic_regression" in payload
    assert "random_forest" in payload
    assert "gradient_boosting" in payload
    assert "baseline_metrics" in payload
    assert "confusion_matrix" in payload
    assert "class_counts" in payload
    assert "feature_importances" in payload
    assert "top_features" in payload
    assert "var_quantiles" in payload
    assert len(payload["var_quantiles"]) == 3

    importances = payload["feature_importances"]
    sorted_features = payload["top_features"]
    assert len(importances) == len(FEATURES)
    assert len(sorted_features) == 10
    
    t_trade = client.post("/trades", json={"portfolio_id": 1, "ticker": "AAPL", "quantity": "10", "price": "150.0000", "side": "BUY"})
    assert t_trade.status_code == 200

    latest_snapshot = db.query(FeatureSnapshot).filter(
        FeatureSnapshot.portfolio_id == 1,
        FeatureSnapshot.snapshot_type == "PORTFOLIO"
    ).order_by(FeatureSnapshot.timestamp.desc(), FeatureSnapshot.id.desc()).first()
    assert latest_snapshot.risk_regime in ["Low", "Moderate", "High", "Critical"]
    assert latest_snapshot.risk_regime_probability is not None
    assert 0.0 <= float(latest_snapshot.risk_regime_probability) <= 1.0

    regime_resp = client.get("/portfolios/1/risk-regime")
    assert regime_resp.status_code == 200
    regime_data = regime_resp.json()
    assert regime_data["portfolio_id"] == 1
    assert "timestamp" in regime_data
    assert regime_data["risk_regime"] in ["Low", "Moderate", "High", "Critical"]
    assert regime_data["risk_regime_probability"] is not None
    assert regime_data["predicted_var_forecast"] is not None

    history_resp = client.get("/portfolios/1/risk-regime/history")
    assert history_resp.status_code == 200
    history_data = history_resp.json()
    assert len(history_data) >= 1
    assert history_data[0]["risk_regime"] in ["Low", "Moderate", "High", "Critical"]

    db.query(Alert).filter(Alert.portfolio_id == 1).delete()
    db.commit()

    prev_snap = FeatureSnapshot(
        portfolio_id=1,
        ticker=None,
        snapshot_type="PORTFOLIO",
        risk_regime="Moderate",
        risk_regime_probability=0.80,
        timestamp=latest_snapshot.timestamp - timedelta(seconds=1)
    )
    db.add(prev_snap)
    db.commit()

    real_payload = joblib.load("models/risk_classifier_v1.joblib")
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([2])          # force "High"
    mock_model.predict_proba.return_value = np.array([[0.05, 0.05, 0.85, 0.05]])
    patched_payload = {**real_payload, "model": mock_model}

    with patch("app.services.risk_classifier.joblib.load", return_value=patched_payload):
        predict_risk_regime(db, latest_snapshot)

    db.flush()
    alerts = db.query(Alert).filter(Alert.portfolio_id == 1, Alert.alert_type == "RISK_REGIME_CHANGE").all()
    assert len(alerts) >= 1
    assert "Moderate" in alerts[0].message
    assert "High" in alerts[0].message

    payload_mismatch = dict(payload)
    payload_mismatch["features"] = ["mismatch_feature"]
    joblib.dump(payload_mismatch, "models/risk_classifier_v1.joblib")

    with pytest.raises(ValueError) as exc:
        predict_risk_regime(db, latest_snapshot)
    assert "Model features mismatch" in str(exc.value)

    db.close()
