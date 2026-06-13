import os
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from sqlalchemy.orm import Session
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from app.models.features import FeatureSnapshot
from app.models.alert import Alert

warnings.filterwarnings("ignore", category=ConvergenceWarning)

STATIC_FEATURES = [
    "exp_net_exposure",
    "exp_gross_exposure",
    "beh_trade_count_1h",
    "beh_avg_trade_size",
    "beh_volume_1h",
    "risk_var_95",
    "risk_hhi_concentration",
    "vol_rolling_volatility_5t",
    "vol_rolling_volatility_30t",
    "anomaly_score",
    "is_anomaly",
]

DYNAMIC_FEATURES = [
    "delta_var_1",
    "delta_var_5",
    "var_pct_change_1",
    "var_pct_change_5",
    "delta_net_exposure_1",
    "delta_net_exposure_5",
    "delta_gross_exposure_1",
    "delta_gross_exposure_5",
    "delta_hhi_1",
    "delta_hhi_5",
    "rolling_mean_var_5",
    "rolling_std_var_5",
    "rolling_mean_var_20",
    "rolling_std_var_20",
    "exposure_momentum",
    "concentration_momentum",
]

FEATURES = STATIC_FEATURES + DYNAMIC_FEATURES
MODEL_PATH = "models/risk_classifier_v1.joblib"
REGIMES = {0: "Low", 1: "Moderate", 2: "High", 3: "Critical"}

def compute_dynamic_features(snapshots: list, idx: int) -> dict:
    curr = snapshots[idx]
    
    def get_val(s, attr):
        return float(getattr(s, attr) or 0.0)
        
    var_curr = get_val(curr, "risk_var_95")
    net_curr = get_val(curr, "exp_net_exposure")
    gross_curr = get_val(curr, "exp_gross_exposure")
    hhi_curr = get_val(curr, "risk_hhi_concentration")
    
    prev1 = snapshots[idx - 1] if idx >= 1 else curr
    prev5 = snapshots[idx - 5] if idx >= 5 else curr
    
    var_prev1 = get_val(prev1, "risk_var_95")
    var_prev5 = get_val(prev5, "risk_var_95")
    net_prev1 = get_val(prev1, "exp_net_exposure")
    net_prev5 = get_val(prev5, "exp_net_exposure")
    gross_prev1 = get_val(prev1, "exp_gross_exposure")
    gross_prev5 = get_val(prev5, "exp_gross_exposure")
    hhi_prev1 = get_val(prev1, "risk_hhi_concentration")
    hhi_prev5 = get_val(prev5, "risk_hhi_concentration")
    
    delta_var_1 = var_curr - var_prev1
    delta_var_5 = var_curr - var_prev5
    var_pct_change_1 = delta_var_1 / max(abs(var_prev1), 1.0)
    var_pct_change_5 = delta_var_5 / max(abs(var_prev5), 1.0)
    
    delta_net_exposure_1 = net_curr - net_prev1
    delta_net_exposure_5 = net_curr - net_prev5
    delta_gross_exposure_1 = gross_curr - gross_prev1
    delta_gross_exposure_5 = gross_curr - gross_prev5
    
    delta_hhi_1 = hhi_curr - hhi_prev1
    delta_hhi_5 = hhi_curr - hhi_prev5
    
    var_vals_5 = [get_val(snapshots[i], "risk_var_95") for i in range(max(0, idx - 4), idx + 1)]
    rolling_mean_var_5 = float(np.mean(var_vals_5))
    rolling_std_var_5 = float(np.std(var_vals_5)) if len(var_vals_5) > 1 else 0.0
    
    var_vals_20 = [get_val(snapshots[i], "risk_var_95") for i in range(max(0, idx - 19), idx + 1)]
    rolling_mean_var_20 = float(np.mean(var_vals_20))
    rolling_std_var_20 = float(np.std(var_vals_20)) if len(var_vals_20) > 1 else 0.0
    
    exposure_momentum = (net_curr - net_prev5) / 5.0
    concentration_momentum = (hhi_curr - hhi_prev5) / 5.0
    
    return {
        "delta_var_1": delta_var_1,
        "delta_var_5": delta_var_5,
        "var_pct_change_1": var_pct_change_1,
        "var_pct_change_5": var_pct_change_5,
        "delta_net_exposure_1": delta_net_exposure_1,
        "delta_net_exposure_5": delta_net_exposure_5,
        "delta_gross_exposure_1": delta_gross_exposure_1,
        "delta_gross_exposure_5": delta_gross_exposure_5,
        "delta_hhi_1": delta_hhi_1,
        "delta_hhi_5": delta_hhi_5,
        "rolling_mean_var_5": rolling_mean_var_5,
        "rolling_std_var_5": rolling_std_var_5,
        "rolling_mean_var_20": rolling_mean_var_20,
        "rolling_std_var_20": rolling_std_var_20,
        "exposure_momentum": exposure_momentum,
        "concentration_momentum": concentration_momentum
    }

def digitize_var(v, q25, q50, q75):
    if v < q25:
        return 0
    elif v < q50:
        return 1
    elif v < q75:
        return 2
    else:
        return 3

def compute_classifier_metrics(y_true, y_pred):
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "macro_f1": float(f1),
        "weighted_f1": float(w_f1)
    }

def train_risk_classifier(db: Session) -> dict:
    start_time = time.perf_counter()
    snapshots = db.query(FeatureSnapshot).filter(
        FeatureSnapshot.snapshot_type == "PORTFOLIO"
    ).order_by(FeatureSnapshot.timestamp.asc()).all()
    if len(snapshots) < 300:
        raise ValueError("Insufficient data for training: at least 300 portfolio snapshots are required.")
    
    X_data = []
    y_raw = []
    for i in range(len(snapshots) - 1):
        row = []
        for f in STATIC_FEATURES:
            if f == "is_anomaly":
                val = 1.0 if getattr(snapshots[i], f) else 0.0
            else:
                val = float(getattr(snapshots[i], f) or 0.0)
            row.append(val)
        dyn = compute_dynamic_features(snapshots, i)
        for f in DYNAMIC_FEATURES:
            row.append(dyn[f])
        X_data.append(row)
        y_raw.append(float(snapshots[i+1].risk_var_95 or 0.0))
        
    X = np.array(X_data)
    y_raw = np.array(y_raw)
    train_size = int(len(X) * 0.8)
    
    y_raw_train = y_raw[:train_size]
    q25 = float(np.percentile(y_raw_train, 25))
    q50 = float(np.percentile(y_raw_train, 50))
    q75 = float(np.percentile(y_raw_train, 75))
    
    y = np.array([digitize_var(val, q25, q50, q75) for val in y_raw])
    y_train, y_test = y[:train_size], y[train_size:]
    X_train, X_test = X[:train_size], X[train_size:]
    
    counts = Counter(y_train)
    for cls in [0, 1, 2, 3]:
        if counts[cls] < 30:
            raise ValueError("Insufficient class diversity for classification training")
            
    lr = LogisticRegression(max_iter=2000, random_state=42)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_metrics = compute_classifier_metrics(y_test, lr_preds)
    
    rf = RandomForestClassifier(random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_metrics = compute_classifier_metrics(y_test, rf_preds)
    
    gb = GradientBoostingClassifier(random_state=42, n_estimators=100)
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_test)
    gb_metrics = compute_classifier_metrics(y_test, gb_preds)
    
    var_index = FEATURES.index("risk_var_95")
    baseline_preds = np.array([digitize_var(val, q25, q50, q75) for val in X_test[:, var_index]])
    baseline_metrics = compute_classifier_metrics(y_test, baseline_preds)
    
    winner_type = "GradientBoosting"
    winner_f1 = gb_metrics["macro_f1"]
    winner_model = gb
    winner_preds = gb_preds
    winner_metrics = gb_metrics
    
    if rf_metrics["macro_f1"] > winner_f1:
        winner_type = "RandomForest"
        winner_f1 = rf_metrics["macro_f1"]
        winner_model = rf
        winner_preds = rf_preds
        winner_metrics = rf_metrics
        
    if lr_metrics["macro_f1"] > winner_f1:
        winner_type = "LogisticRegression"
        winner_f1 = lr_metrics["macro_f1"]
        winner_model = lr
        winner_preds = lr_preds
        winner_metrics = lr_metrics
        
    cm = confusion_matrix(y_test, winner_preds, labels=[0, 1, 2, 3]).tolist()
    
    if hasattr(winner_model, "feature_importances_"):
        importances = winner_model.feature_importances_
        importance_items = [(FEATURES[i], float(importances[i])) for i in range(len(FEATURES))]
        importance_items.sort(key=lambda x: x[1], reverse=True)
        sorted_importances = {item[0]: item[1] for item in importance_items}
        top_features = [item[0] for item in importance_items[:10]]
    else:
        sorted_importances = {f: 0.0 for f in FEATURES}
        top_features = FEATURES[:10]
        
    class_counts = {
        "Low": int(counts[0]),
        "Moderate": int(counts[1]),
        "High": int(counts[2]),
        "Critical": int(counts[3])
    }
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    training_time_ms = float((time.perf_counter() - start_time) * 1000.0)
    payload = {
        "model": winner_model,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURES,
        "selected_model": winner_type,
        "logistic_regression": lr_metrics,
        "random_forest": rf_metrics,
        "gradient_boosting": gb_metrics,
        "baseline_metrics": baseline_metrics,
        "confusion_matrix": cm,
        "class_counts": class_counts,
        "feature_importances": sorted_importances,
        "top_features": top_features,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "training_time_ms": training_time_ms,
        "var_quantiles": [q25, q50, q75]
    }
    joblib.dump(payload, MODEL_PATH)
    return payload

def predict_risk_regime(db: Session, snapshot: FeatureSnapshot) -> None:
    if snapshot.snapshot_type != "PORTFOLIO":
        return
    if not os.path.exists(MODEL_PATH):
        snapshot.risk_regime = None
        snapshot.risk_regime_probability = None
        return
    payload = joblib.load(MODEL_PATH)
    saved_features = payload.get("features", [])
    if saved_features != FEATURES:
        raise ValueError("Model features mismatch: expected " + str(saved_features) + " but got " + str(FEATURES))
    
    hist = db.query(FeatureSnapshot).filter(
        FeatureSnapshot.portfolio_id == snapshot.portfolio_id,
        FeatureSnapshot.snapshot_type == "PORTFOLIO",
        FeatureSnapshot.timestamp < snapshot.timestamp
    ).order_by(FeatureSnapshot.timestamp.asc()).all()
    
    sequence = hist + [snapshot]
    idx = len(sequence) - 1
    
    row = []
    for f in STATIC_FEATURES:
        if f == "is_anomaly":
            val = 1.0 if getattr(snapshot, f) else 0.0
        else:
            val = float(getattr(snapshot, f) or 0.0)
        row.append(val)
    dyn = compute_dynamic_features(sequence, idx)
    for f in DYNAMIC_FEATURES:
        row.append(dyn[f])
        
    model = payload["model"]
    x = np.array([row])
    pred = int(model.predict(x)[0])
    probs = model.predict_proba(x)[0]
    prob = float(probs[pred])
    
    regime_name = REGIMES[pred]
    snapshot.risk_regime = regime_name
    snapshot.risk_regime_probability = prob
    
    prev_snapshot = db.query(FeatureSnapshot).filter(
        FeatureSnapshot.portfolio_id == snapshot.portfolio_id,
        FeatureSnapshot.snapshot_type == "PORTFOLIO",
        FeatureSnapshot.timestamp < snapshot.timestamp
    ).order_by(FeatureSnapshot.timestamp.desc(), FeatureSnapshot.id.desc()).first()
    
    if prev_snapshot and prev_snapshot.risk_regime != regime_name:
        transition = (prev_snapshot.risk_regime, regime_name)
        allowed_transitions = [
            ("Moderate", "High"),
            ("High", "Critical"),
            ("Low", "Critical")
        ]
        if transition in allowed_transitions:
            alert = Alert(
                portfolio_id=snapshot.portfolio_id,
                alert_type="RISK_REGIME_CHANGE",
                message=f"Risk regime transitioned from {prev_snapshot.risk_regime} to {regime_name}"
            )
            db.add(alert)
