import os
import json
import time
import logging
from datetime import datetime, timezone
from sqlalchemy.orm import Session
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from app.models.features import FeatureSnapshot
from app.models.alert import Alert

logger = logging.getLogger(__name__)

FEATURES = [
    "exp_net_exposure",
    "exp_gross_exposure",
    "beh_trade_count_1h",
    "beh_avg_trade_size",
    "beh_volume_1h",
    "risk_var_95",
    "risk_hhi_concentration",
    "vol_rolling_volatility_5t",
    "vol_rolling_volatility_30t",
]

MODEL_PATH = "models/anomaly_detector_v1.joblib"

def train_global_anomaly_model(db: Session, contamination: float = 0.01) -> dict:
    start_time = time.perf_counter()
    snapshots = db.query(FeatureSnapshot).filter(
        FeatureSnapshot.snapshot_type == "PORTFOLIO"
    ).all()
    if len(snapshots) < 100:
        raise ValueError("Insufficient data for training: at least 100 portfolio snapshots are required.")
    data = []
    for s in snapshots:
        row = [float(getattr(s, f) or 0.0) for f in FEATURES]
        data.append(row)
    x = np.array(data)
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(x)
    preds = model.predict(x)
    anomaly_count = int(np.sum(preds == -1))
    anomaly_rate = float(anomaly_count / len(snapshots))
    means = {}
    stds = {}
    for i, f in enumerate(FEATURES):
        means[f] = float(np.mean(x[:, i]))
        if len(x) > 1:
            stds[f] = float(np.std(x[:, i], ddof=1))
        else:
            stds[f] = 0.0
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    training_time_ms = float((time.perf_counter() - start_time) * 1000.0)
    payload = {
        "model": model,
        "means": means,
        "stds": stds,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "contamination": contamination,
        "training_rows": len(snapshots),
        "features": FEATURES,
        "anomaly_count": anomaly_count,
        "anomaly_rate": anomaly_rate,
        "training_time_ms": training_time_ms
    }
    joblib.dump(payload, MODEL_PATH)
    return payload

def score_anomaly(db: Session, snapshot: FeatureSnapshot) -> None:
    if snapshot.snapshot_type != "PORTFOLIO":
        return
    if not os.path.exists(MODEL_PATH):
        snapshot.anomaly_score = None
        snapshot.is_anomaly = None
        snapshot.anomaly_explanation = None
        return
    load_start = time.perf_counter()
    payload = joblib.load(MODEL_PATH)
    model_load_time_ms = float((time.perf_counter() - load_start) * 1000.0)
    saved_features = payload.get("features", [])
    if saved_features != FEATURES:
        raise ValueError("Model features mismatch: expected " + str(saved_features) + " but got " + str(FEATURES))
    means = payload["means"]
    stds = payload["stds"]
    model = payload["model"]
    inference_start = time.perf_counter()
    row = [float(getattr(snapshot, f) or 0.0) for f in FEATURES]
    x = np.array([row])
    score = float(model.decision_function(x)[0])
    is_anomaly = bool(model.predict(x)[0] == -1)
    inference_time_ms = float((time.perf_counter() - inference_start) * 1000.0)
    logger.info(f"Anomaly scoring performance: load_time={model_load_time_ms:.2f}ms inference_time={inference_time_ms:.2f}ms")
    snapshot.anomaly_score = score
    snapshot.is_anomaly = is_anomaly
    if is_anomaly:
        z_scores = []
        for f in FEATURES:
            mean = means[f]
            std = stds[f]
            val = float(getattr(snapshot, f) or 0.0)
            if std > 0.0:
                z = abs((val - mean) / std)
            else:
                z = 0.0
            z_scores.append((f, z))
        z_scores.sort(key=lambda item: item[1], reverse=True)
        top_3 = [item[0] for item in z_scores[:3]]
        snapshot.anomaly_explanation = json.dumps(top_3)
        alert = Alert(
            portfolio_id=snapshot.portfolio_id,
            alert_type="ANOMALY_DETECTED",
            message=f"Portfolio anomaly detected with score {score:.4f}. Top contributors: {', '.join(top_3)}"
        )
        db.add(alert)
    else:
        snapshot.anomaly_explanation = None
