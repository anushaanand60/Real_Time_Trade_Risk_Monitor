import os
import sys
import time
import json
import random
import argparse
from decimal import Decimal
from datetime import datetime, timezone, timedelta

os.environ["TESTING"] = "True"

import numpy as np

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.portfolio import Portfolio
from app.models.trade import Trade
from app.models.position import Position
from app.models.alert import Alert
from app.models.features import FeatureSnapshot
from app.services.position import update_position_logic
from app.services.alerts import run_post_trade_alerts
from app.services.feature_generation import generate_features_for_trade
from app.services.anomaly_detector import (
    train_global_anomaly_model, score_anomaly,
    FEATURES as ANOMALY_FEATURES, MODEL_PATH as ANOMALY_MODEL_PATH,
)
from app.services.var_forecaster import (
    train_var_forecaster, predict_var_forecast,
    FEATURES as VAR_FEATURES, MODEL_PATH as VAR_MODEL_PATH,
)
from app.services.risk_classifier import (
    train_risk_classifier, predict_risk_regime,
    FEATURES as CLF_FEATURES, MODEL_PATH as CLF_MODEL_PATH,
    STATIC_FEATURES as CLF_STATIC_FEATURES,
    DYNAMIC_FEATURES as CLF_DYNAMIC_FEATURES,
)
from app.services.market_simulator import (
    generate_simulation_data, REGIME_SCHEDULE,
)

BENCH_DB = "./benchmark_run.db"
REPORT_PATH = "benchmark_report.json"
TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

SEP = "=" * 65
SUBSEP = "-" * 65


def make_session():
    engine = create_engine(BENCH_DB.replace("./", "sqlite:///./"),
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SM = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, SM()


def model_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0.0


def fmt(label: str, value) -> str:
    return f"  {label:<42} {value}"

def generate_data(db, n_trades: int):
    print(f"\n{SEP}")
    print(f"  DATA GENERATION  ({n_trades} trades, regime-aware simulator)")
    print(SEP)
    print("  Volatility regime schedule:")
    for start, end, name, sigma, beta in REGIME_SCHEDULE:
        trades_start = int(start * n_trades)
        trades_end = int(end * n_trades)
        extra = f"  crisis_beta={beta}" if beta > 0 else ""
        print(f"    [{trades_start:>4}-{trades_end:>4}]  {name:<12}  sigma={sigma:.3f}{extra}")
    print()

    t0 = time.perf_counter()
    generate_simulation_data(db, portfolio_id=1, num_trades=n_trades)
    sim_ms = (time.perf_counter() - t0) * 1000

    n_snap = db.query(FeatureSnapshot).filter(
        FeatureSnapshot.snapshot_type == "PORTFOLIO"
    ).count()
    print(fmt("Trades inserted:", n_trades))
    print(fmt("Portfolio snapshots:", n_snap))
    print(fmt("Simulation time:", f"{sim_ms:,.1f} ms"))
    return n_trades, n_snap, sim_ms


# Phase 2: Anomaly Detection
def bench_anomaly(db, latency_runs: int):
    print(f"\n{SEP}")
    print("  PHASE 2 — ANOMALY DETECTION  (IsolationForest)")
    print(SEP)

    res = train_global_anomaly_model(db, contamination=0.01)
    size_mb = model_size_mb(ANOMALY_MODEL_PATH)
    print(fmt("Training rows:", res["training_rows"]))
    print(fmt("Contamination:", res["contamination"]))
    print(fmt("Anomalies flagged:", res["anomaly_count"]))
    print(fmt("Anomaly rate:", f"{res['anomaly_rate']:.4f}  ({res['anomaly_rate']*100:.2f} %)"))
    print(fmt("Training time:", f"{res['training_time_ms']:,.1f} ms"))
    print(fmt("Model size:", f"{size_mb:.3f} MB"))

    sample = (
        db.query(FeatureSnapshot)
        .filter(FeatureSnapshot.snapshot_type == "PORTFOLIO")
        .order_by(FeatureSnapshot.id.desc()).first()
    )
    lats = []
    for _ in range(latency_runs):
        t0 = time.perf_counter()
        score_anomaly(db, sample)
        lats.append((time.perf_counter() - t0) * 1000)
    lats = np.array(lats)
    print(fmt("Inference latency (median):", f"{np.median(lats):.3f} ms"))
    print(fmt("Inference latency (p95):", f"{np.percentile(lats, 95):.3f} ms"))
    print(fmt("Inference latency (p99):", f"{np.percentile(lats, 99):.3f} ms"))

    return {
        "algorithm": "IsolationForest",
        "contamination": res["contamination"],
        "training_rows": res["training_rows"],
        "anomaly_count": res["anomaly_count"],
        "anomaly_rate": round(res["anomaly_rate"], 4),
        "training_time_ms": round(res["training_time_ms"], 2),
        "model_size_mb": round(size_mb, 4),
        "features": ANOMALY_FEATURES,
        "feature_means": {k: round(v, 4) for k, v in res["means"].items()},
        "feature_stds":  {k: round(v, 4) for k, v in res["stds"].items()},
        "latency": {
            "median_ms": round(float(np.median(lats)), 3),
            "p95_ms": round(float(np.percentile(lats, 95)), 3),
            "p99_ms": round(float(np.percentile(lats, 99)), 3),
            "runs": latency_runs,
        },
    }


# Phase 3: VaR Forecasting 

def bench_var(db, latency_runs: int):
    print(f"\n{SEP}")
    print("  PHASE 3 — VAR FORECASTING  (RF vs GBM)")
    print(SEP)

    res = train_var_forecaster(db)
    size_mb = model_size_mb(VAR_MODEL_PATH)
    bl = res["baseline_metrics"]
    rf = res["random_forest"]
    gb = res["gradient_boosting"]

    print(fmt("Selected model:", res["selected_model"]))
    print(fmt("Training rows:", res["training_rows"]))
    print(fmt("Test rows:", res["test_rows"]))
    print(fmt("Training time:", f"{res['training_time_ms']:,.1f} ms"))
    print(fmt("Model size:", f"{size_mb:.3f} MB"))
    print()
    print(f"  {'Model':<22} {'MAE':>10} {'RMSE':>10} {'R²':>8}")
    print(f"  {SUBSEP[:54]}")
    print(f"  {'Persistence (baseline)':<22} {bl['mae']:>10.4f} {bl['rmse']:>10.4f} {bl['r2']:>8.4f}")
    print(f"  {'RandomForest':<22} {rf['mae']:>10.4f} {rf['rmse']:>10.4f} {rf['r2']:>8.4f}")
    print(f"  {'GradientBoosting':<22} {gb['mae']:>10.4f} {gb['rmse']:>10.4f} {gb['r2']:>8.4f}")
    print()
    print(fmt("Outperformed baseline:", str(res["outperformed_baseline"])))
    print()
    print("  Top-10 Feature Importances:")
    for rank, (feat, imp) in enumerate(list(res["feature_importances"].items())[:10], 1):
        print(f"    {rank:>2}. {feat:<38} {imp:.4f}")

    sample = (
        db.query(FeatureSnapshot)
        .filter(FeatureSnapshot.snapshot_type == "PORTFOLIO")
        .order_by(FeatureSnapshot.id.desc()).first()
    )
    lats = []
    for _ in range(latency_runs):
        t0 = time.perf_counter()
        predict_var_forecast(db, sample)
        lats.append((time.perf_counter() - t0) * 1000)
    lats = np.array(lats)
    print(fmt("Inference latency (median):", f"{np.median(lats):.3f} ms"))
    print(fmt("Inference latency (p95):", f"{np.percentile(lats, 95):.3f} ms"))

    return {
        "selected_model": res["selected_model"],
        "training_rows": res["training_rows"],
        "test_rows": res["test_rows"],
        "training_time_ms": round(res["training_time_ms"], 2),
        "model_size_mb": round(size_mb, 4),
        "winner": {"mae": round(res["mae"], 4), "rmse": round(res["rmse"], 4), "r2": round(res["r2"], 4)},
        "random_forest": {k: round(v, 4) for k, v in rf.items()},
        "gradient_boosting": {k: round(v, 4) for k, v in gb.items()},
        "baseline": {k: round(v, 4) for k, v in bl.items()},
        "outperformed_baseline": res["outperformed_baseline"],
        "feature_importances": {k: round(v, 4) for k, v in res["feature_importances"].items()},
        "top_10_features": res["top_features"][:10],
        "static_features": VAR_FEATURES[:9],
        "temporal_features": VAR_FEATURES[9:],
        "latency": {
            "median_ms": round(float(np.median(lats)), 3),
            "p95_ms": round(float(np.percentile(lats, 95)), 3),
            "p99_ms": round(float(np.percentile(lats, 99)), 3),
            "runs": latency_runs,
        },
    }

# Phase 4: Risk Classifier
def bench_classifier(db, latency_runs: int):
    print(f"\n{SEP}")
    print("  PHASE 4 — RISK REGIME CLASSIFICATION  (LR / RF / GBM)")
    print(SEP)

    res = train_risk_classifier(db)
    size_mb = model_size_mb(CLF_MODEL_PATH)
    lr = res["logistic_regression"]
    rf = res["random_forest"]
    gb = res["gradient_boosting"]
    bl = res["baseline_metrics"]
    q = res["var_quantiles"]

    print(fmt("Selected model:", res["selected_model"]))
    print(fmt("Training rows:", res["training_rows"]))
    print(fmt("Test rows:", res["test_rows"]))
    print(fmt("Training time:", f"{res['training_time_ms']:,.1f} ms"))
    print(fmt("Model size:", f"{size_mb:.3f} MB"))
    print()
    print(fmt("VaR quantile thresholds (q25/q50/q75):", f"{q[0]:.4f} / {q[1]:.4f} / {q[2]:.4f}"))
    print()
    cc = res["class_counts"]
    print(fmt("Class distribution (train):",
              f"Low={cc['Low']}  Moderate={cc['Moderate']}  High={cc['High']}  Critical={cc['Critical']}"))
    print()
    print(f"  {'Model':<26} {'Accuracy':>9} {'Macro F1':>9} {'Weighted F1':>12}")
    print(f"  {SUBSEP[:60]}")
    print(f"  {'Persistence (baseline)':<26} {bl['accuracy']:>9.4f} {bl['macro_f1']:>9.4f} {bl['weighted_f1']:>12.4f}")
    print(f"  {'LogisticRegression':<26} {lr['accuracy']:>9.4f} {lr['macro_f1']:>9.4f} {lr['weighted_f1']:>12.4f}")
    print(f"  {'RandomForest':<26} {rf['accuracy']:>9.4f} {rf['macro_f1']:>9.4f} {rf['weighted_f1']:>12.4f}")
    print(f"  {'GradientBoosting':<26} {gb['accuracy']:>9.4f} {gb['macro_f1']:>9.4f} {gb['weighted_f1']:>12.4f}")
    print()
    print("  Confusion Matrix  [rows=actual, cols=predicted]  Low/Mod/High/Crit:")
    labels = ["Low ", "Mod ", "High", "Crit"]
    for i, row in enumerate(res["confusion_matrix"]):
        print(f"    {labels[i]} | " + "  ".join(f"{v:>4}" for v in row))
    print()
    print("  Top-10 Feature Importances:")
    for rank, feat in enumerate(res["top_features"][:10], 1):
        imp = res["feature_importances"].get(feat, 0.0)
        print(f"    {rank:>2}. {feat:<38} {imp:.4f}")

    sample = (
        db.query(FeatureSnapshot)
        .filter(FeatureSnapshot.snapshot_type == "PORTFOLIO")
        .order_by(FeatureSnapshot.id.desc()).first()
    )
    lats = []
    for _ in range(latency_runs):
        t0 = time.perf_counter()
        predict_risk_regime(db, sample)
        lats.append((time.perf_counter() - t0) * 1000)
    lats = np.array(lats)
    print(fmt("Inference latency (median):", f"{np.median(lats):.3f} ms"))
    print(fmt("Inference latency (p95):", f"{np.percentile(lats, 95):.3f} ms"))

    return {
        "selected_model": res["selected_model"],
        "training_rows": res["training_rows"],
        "test_rows": res["test_rows"],
        "training_time_ms": round(res["training_time_ms"], 2),
        "model_size_mb": round(size_mb, 4),
        "var_quantiles": {"q25": round(q[0], 4), "q50": round(q[1], 4), "q75": round(q[2], 4)},
        "class_counts_train": cc,
        "regimes": ["Low", "Moderate", "High", "Critical"],
        "logistic_regression": {k: round(v, 4) for k, v in lr.items()},
        "random_forest": {k: round(v, 4) for k, v in rf.items()},
        "gradient_boosting": {k: round(v, 4) for k, v in gb.items()},
        "baseline": {k: round(v, 4) for k, v in bl.items()},
        "confusion_matrix": res["confusion_matrix"],
        "feature_importances": {k: round(v, 4) for k, v in res["feature_importances"].items()},
        "top_10_features": res["top_features"][:10],
        "static_features": CLF_STATIC_FEATURES,
        "temporal_features": CLF_DYNAMIC_FEATURES,
        "latency": {
            "median_ms": round(float(np.median(lats)), 3),
            "p95_ms": round(float(np.percentile(lats, 95)), 3),
            "p99_ms": round(float(np.percentile(lats, 99)), 3),
            "runs": latency_runs,
        },
    }

def bench_e2e(db, latency_runs: int):
    print(f"\n{SEP}")
    print("  END-TO-END PIPELINE LATENCY")
    print(SEP)

    portfolio_id = 1
    _= db.query(Position).filter(Position.portfolio_id == portfolio_id).first()

    feat_lats, anomaly_lats, var_lats, clf_lats, total_lats = [], [], [], [], []

    for _ in range(latency_runs):
        trade = Trade(
            portfolio_id=portfolio_id,
            ticker=random.choice(TICKERS),
            quantity=Decimal(str(random.randint(10, 50))),
            price=Decimal(str(round(random.uniform(100.0, 400.0), 4))),
            side="BUY",
        )
        db.add(trade)
        db.flush()
        update_position_logic(db, trade)

        t_total = time.perf_counter()

        t0 = time.perf_counter()
        snapshot = generate_features_for_trade(db, trade)
        feat_lats.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        score_anomaly(db, snapshot)
        anomaly_lats.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        predict_var_forecast(db, snapshot)
        var_lats.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        predict_risk_regime(db, snapshot)
        clf_lats.append((time.perf_counter() - t0) * 1000)

        total_lats.append((time.perf_counter() - t_total) * 1000)

        db.rollback()

    def stats(arr):
        a = np.array(arr)
        return {
            "mean_ms":   round(float(np.mean(a)),   3),
            "median_ms": round(float(np.median(a)),  3),
            "p95_ms":    round(float(np.percentile(a, 95)), 3),
            "p99_ms":    round(float(np.percentile(a, 99)), 3),
        }

    rows = [
        ("Feature generation",    feat_lats),
        ("Anomaly scoring",       anomaly_lats),
        ("VaR forecast",          var_lats),
        ("Risk classification",   clf_lats),
        ("Full pipeline",         total_lats),
    ]
    print(f"  {'Stage':<30} {'Mean':>8} {'Median':>8} {'P95':>8} {'P99':>8}")
    print(f"  {SUBSEP[:66]}")
    for label, arr in rows:
        a = np.array(arr)
        print(f"  {label:<30} {np.mean(a):>7.3f}  {np.median(a):>7.3f}  "
              f"{np.percentile(a,95):>7.3f}  {np.percentile(a,99):>7.3f}  ms")

    return {
        "runs": latency_runs,
        "feature_generation": stats(feat_lats),
        "anomaly_scoring":    stats(anomaly_lats),
        "var_forecast":       stats(var_lats),
        "risk_classification":stats(clf_lats),
        "full_pipeline":      stats(total_lats),
    }


def main():
    parser = argparse.ArgumentParser(description="Trade Risk Monitor — Benchmark")
    parser.add_argument("--trades", type=int, default=350,
                        help="Number of trades to simulate (default: 350)")
    parser.add_argument("--latency-runs", type=int, default=100,
                        help="Inference runs per stage (default: 100)")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    if os.path.exists(BENCH_DB.lstrip("./")):
        os.remove(BENCH_DB.lstrip("./"))

    engine, db = make_session()

    print(f"\n{'#'*65}")
    print("  REAL-TIME TRADE RISK MONITOR — BENCHMARK")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  trades={args.trades}  latency_runs={args.latency_runs}")
    print(f"{'#'*65}")

    n_trades, n_snap, sim_ms = generate_data(db, args.trades)
    anomaly_metrics = bench_anomaly(db, args.latency_runs)
    var_metrics = bench_var(db, args.latency_runs)
    clf_metrics = bench_classifier(db, args.latency_runs)
    e2e_metrics = bench_e2e(db, args.latency_runs)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "n_trades": n_trades,
            "n_portfolio_snapshots": n_snap,
            "tickers": TICKERS,
            "simulation_time_ms": round(sim_ms, 2),
        },
        "training_thresholds": {
            "anomaly_min_snapshots": 100,
            "var_min_snapshots": 300,
            "classifier_min_snapshots": 300,
            "classifier_min_class_count": 30,
            "var_window_days": 30,
            "var_confidence_level": 0.95,
            "concentration_alert_threshold": 0.40,
            "var_alert_threshold": 1000000.00,
            "train_test_split": "80 / 20  (chronological)",
        },
        "api_inventory": {
            "portfolios": [
                "POST /portfolios",
                "GET  /portfolios/{id}/positions",
                "GET  /portfolios/{id}/var",
            ],
            "trades": [
                "POST /trades",
                "GET  /portfolios/{id}/trades",
            ],
            "simulation": [
                "POST /portfolios/{id}/simulate-data",
                "POST /simulate",
            ],
            "features": [
                "GET  /portfolios/{id}/features",
            ],
            "anomaly_detection": [
                "POST /anomaly/train",
                "GET  /portfolios/{id}/anomaly/score",
                "GET  /portfolios/{id}/anomaly/history",
            ],
            "var_forecasting": [
                "POST /var/train",
                "GET  /portfolios/{id}/var/forecast",
            ],
            "risk_classification": [
                "POST /risk-classifier/train",
                "GET  /portfolios/{id}/risk-regime",
                "GET  /portfolios/{id}/risk-regime/history",
            ],
            "system": [
                "GET  /health",
            ],
        },
        "phase2_anomaly_detection": anomaly_metrics,
        "phase3_var_forecasting": var_metrics,
        "phase4_risk_classification": clf_metrics,
        "end_to_end_latency": e2e_metrics,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{SEP}")
    print(f"  Report written -> {REPORT_PATH}")
    print(SEP)
    print(f"\n  End-to-end median pipeline latency: "
          f"{e2e_metrics['full_pipeline']['median_ms']:.2f} ms")
    print(f"  End-to-end p99 pipeline latency:    "
          f"{e2e_metrics['full_pipeline']['p99_ms']:.2f} ms\n")

    db.close()
    engine.dispose()
    for path in [BENCH_DB.lstrip("./"),
                 ANOMALY_MODEL_PATH, VAR_MODEL_PATH, CLF_MODEL_PATH]:
        if os.path.exists(path):
            os.remove(path)

if __name__ == "__main__":
    main()
