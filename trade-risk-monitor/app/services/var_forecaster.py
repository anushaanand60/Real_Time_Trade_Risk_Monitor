import os
import time
from datetime import datetime, timezone
from sqlalchemy.orm import Session
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from app.models.features import FeatureSnapshot

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
MODEL_PATH = "models/var_forecaster_v1.joblib"

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

def train_var_forecaster(db: Session) -> dict:
    start_time = time.perf_counter()
    snapshots = db.query(FeatureSnapshot).filter(
        FeatureSnapshot.snapshot_type == "PORTFOLIO"
    ).order_by(FeatureSnapshot.timestamp.asc()).all()
    if len(snapshots) < 300:
        raise ValueError("Insufficient data for training: at least 300 portfolio snapshots are required.")
    
    X_data = []
    y_data = []
    for i in range(len(snapshots) - 1):
        row = []
        for f in STATIC_FEATURES:
            row.append(float(getattr(snapshots[i], f) or 0.0))
        dyn = compute_dynamic_features(snapshots, i)
        for f in DYNAMIC_FEATURES:
            row.append(dyn[f])
        X_data.append(row)
        var_curr = float(snapshots[i].risk_var_95 or 0.0)
        var_next = float(snapshots[i+1].risk_var_95 or 0.0)
        y_data.append(var_next - var_curr)
        
    X = np.array(X_data)
    y = np.array(y_data)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    var_index = FEATURES.index("risk_var_95")
    baseline_preds = np.zeros(len(y_test))
    baseline_mae = float(mean_absolute_error(y_test, baseline_preds))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_preds)))
    baseline_r2 = float(r2_score(y_test, baseline_preds))
    
    rf = RandomForestRegressor(random_state=42, n_estimators=100)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_mae = float(mean_absolute_error(y_test, rf_preds))
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_preds)))
    rf_r2 = float(r2_score(y_test, rf_preds))
    
    gb = GradientBoostingRegressor(random_state=42, n_estimators=100)
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_test)
    gb_mae = float(mean_absolute_error(y_test, gb_preds))
    gb_rmse = float(np.sqrt(mean_squared_error(y_test, gb_preds)))
    gb_r2 = float(r2_score(y_test, gb_preds))
    
    if rf_rmse <= gb_rmse:
        winner = rf
        winner_type = "RandomForest"
        winner_mae = rf_mae
        winner_rmse = rf_rmse
        winner_r2 = rf_r2
    else:
        winner = gb
        winner_type = "GradientBoosting"
        winner_mae = gb_mae
        winner_rmse = gb_rmse
        winner_r2 = gb_r2
        
    outperformed_baseline = bool(winner_rmse < baseline_rmse)
    importances = winner.feature_importances_
    importance_items = [(FEATURES[i], float(importances[i])) for i in range(len(FEATURES))]
    importance_items.sort(key=lambda item: item[1], reverse=True)
    sorted_importances = {item[0]: item[1] for item in importance_items}
    top_features = [item[0] for item in importance_items]
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    training_time_ms = float((time.perf_counter() - start_time) * 1000.0)
    payload = {
        "model": winner,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURES,
        "target_type": "delta_var",
        "model_type": winner_type,
        "selected_model": winner_type,
        "mae": winner_mae,
        "rmse": winner_rmse,
        "r2": winner_r2,
        "random_forest": {"mae": rf_mae, "rmse": rf_rmse, "r2": rf_r2},
        "gradient_boosting": {"mae": gb_mae, "rmse": gb_rmse, "r2": gb_r2},
        "baseline_metrics": {"mae": baseline_mae, "rmse": baseline_rmse, "r2": baseline_r2},
        "outperformed_baseline": outperformed_baseline,
        "feature_importances": sorted_importances,
        "top_features": top_features,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "training_time_ms": training_time_ms
    }
    joblib.dump(payload, MODEL_PATH)
    return payload

def predict_var_forecast(db: Session, snapshot: FeatureSnapshot) -> None:
    if snapshot.snapshot_type != "PORTFOLIO":
        return
    if not os.path.exists(MODEL_PATH):
        snapshot.risk_var_forecast = None
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
        row.append(float(getattr(snapshot, f) or 0.0))
    dyn = compute_dynamic_features(sequence, idx)
    for f in DYNAMIC_FEATURES:
        row.append(dyn[f])
        
    model = payload["model"]
    x = np.array([row])
    predicted_delta = float(model.predict(x)[0])
    current_var = float(snapshot.risk_var_95 or 0.0)
    snapshot.risk_var_forecast = current_var + predicted_delta
