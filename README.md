# Intelligent Trade Risk Monitor

Real-time portfolio risk monitoring system built with FastAPI, PostgreSQL, Redis, and a 4-stage ML pipeline. Ingests trades, tracks positions in real time, generates rolling features, detects anomalies, forecasts VaR changes, and classifies risk regimes to trigger alerts.

---

## 1. System Architecture

```mermaid
graph TD
    Ingestion[Trade Ingestion] --> FeatureStore[Feature Store]
    FeatureStore --> Anomaly[Anomaly Detection <br> Isolation Forest]
    FeatureStore --> VaR[VaR Forecasting <br> RandomForest / GBM]
    Anomaly --> Classification[Risk Regime Classification <br> RandomForest / GBM]
    VaR --> Classification
    Classification --> AlertEngine[Alert Engine]
    AlertEngine --> FastAPI[FastAPI APIs]
```

**Data flow**: Live trade → FastAPI router → PostgreSQL logging + feature store update → model pipeline execution → cache eviction → Redis in-memory serving

---

## 2. Repository Structure

```
Real_Time_Trade_Risk_Monitor/
├── trade-risk-monitor/
│   ├── app/
│   │   ├── core/
│   │   │   ├── config.py          # app settings, alert thresholds
│   │   │   └── redis.py           # redis client, cache utilities
│   │   ├── models/                # SQLAlchemy DB schemas
│   │   │   ├── alert.py           # triggered alerts
│   │   │   ├── features.py        # rolling feature snapshots
│   │   │   ├── portfolio.py       # portfolios
│   │   │   ├── position.py        # holdings per ticker
│   │   │   └── trade.py           # trade transaction logs
│   │   ├── routers/               # API endpoints
│   │   │   ├── anomaly.py         # anomaly training/scoring
│   │   │   ├── features.py        # feature store queries
│   │   │   ├── portfolio.py       # portfolio/position retrieval
│   │   │   ├── risk_classifier.py # risk regime routes
│   │   │   ├── simulate.py        # simulation runner
│   │   │   ├── trade.py           # trade ingestion
│   │   │   └── var_forecast.py    # VaR forecast routes
│   │   ├── schemas/               # Pydantic validation
│   │   ├── services/              # business logic
│   │   │   ├── alerts.py          # threshold checking
│   │   │   ├── anomaly_detector.py # Isolation Forest
│   │   │   ├── feature_generation.py # 25 statistical features
│   │   │   ├── market_simulator.py # synthetic trade generator
│   │   │   ├── position.py        # position compounding
│   │   │   ├── risk.py            # historical VaR (NumPy)
│   │   │   ├── risk_classifier.py # Random Forest classifier
│   │   │   └── var_forecaster.py  # VaR change forecasting
│   │   ├── database.py            # DB connection helper
│   │   └── main.py                # FastAPI bootstrap
│   ├── tests/
│   │   ├── conftest.py            # pytest fixtures
│   │   ├── test_anomaly.py
│   │   ├── test_features.py
│   │   ├── test_risk_classifier.py
│   │   ├── test_risk_monitor.py
│   │   └── test_var_forecast.py
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── benchmark.py
│   └── benchmark_report.json
├── .gitignore
└── README.md
```

---

## 3. Results Summary

**VaR has strong temporal persistence**. Next-day VaR is heavily dependent on current VaR, which makes predicting absolute values trivial but not useful.

**Forecasting risk changes is hard**. Because VaR is persistent, a simple baseline predicting zero change (ΔVaR = 0) actually outperformed our ML models:
- Random Forest RMSE: 179,034
- Gradient Boosting RMSE: 185,893
- Baseline (zero change) RMSE: 145,964

**We had target leakage initially**. The risk classifier was getting the VaR forecast as an input and just learned to map it to quantile thresholds. Got 100% accuracy but learned nothing. Removing that feature forced it to actually learn from exposures, concentration levels, and rolling volatilities.

**After fixing leakage, classification works**. Random Forest achieved 93% test accuracy on Low/Moderate/High/Critical regimes using chronological split (no lookahead).

**Latency is fine for production**. Median end-to-end is 259ms. Could get under 35ms by pre-loading models in memory instead of reading from disk each time.

---

## 4. Performance Metrics

**VaR Forecasting (predicting ΔVaR)**

| Model | MAE | RMSE | R² |
| :--- | ---: | ---: | ---: |
| Random Forest Regressor | 79,218 | 179,034 | -0.5044 |
| Gradient Boosting Regressor | 92,638 | 185,893 | -0.6219 |
| Persistence Baseline (zero change) | 23,387 | 145,965 | 0.0000 |

**Risk Regime Classification (Low, Moderate, High, Critical)**

| Model | Accuracy | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: |
| Random Forest Classifier | 93.0% | 0.5829 | 0.9131 |
| Gradient Boosting Classifier | 92.0% | 0.5375 | 0.9086 |
| Logistic Regression | 85.0% | 0.2933 | 0.8542 |
| Persistence Baseline (previous regime) | 96.0% | 0.7705 | 0.9600 |

*Note: Persistence baseline looks good here because the test set has the portfolio at max size, staying in High/Critical regimes for many consecutive periods.*

---

## 5. Latency Profile

Measured over 150 simulation runs:

| Stage | Mean | Median | P95 | P99 |
| :--- | ---: | ---: | ---: | ---: |
| Feature Generation | 41.4 ms | 40.9 ms | 44.4 ms | 48.7 ms |
| Anomaly Detection | 66.8 ms | 65.7 ms | 73.6 ms | 83.4 ms |
| VaR Forecasting | 74.5 ms | 71.6 ms | 89.8 ms | 132.4 ms |
| Risk Classification | 80.8 ms | 79.7 ms | 85.4 ms | 100.4 ms |
| **Full Pipeline** | **263.6 ms** | **259.1 ms** | **291.9 ms** | **326.0 ms** |

---

## 6. What Went Wrong (And How We Fixed It)

### 6.1 The target formulation problem

Initially we tried to predict absolute next-day VaR. But consecutive VaR values are almost identical in a growing portfolio, so the model looked good without actually learning anything useful.

**Fix**: Redesigned the market simulator to inject volatile regimes, correlation crashes, and exposure shocks. Changed the target to predict the change in VaR (ΔVaR = VaR(t+1) - VaR(t)). Now the baseline is "predict zero change" (R² ≈ 0), so the models actually have to learn something to beat it.

### 6.2 Feature leakage in classification

The classifier was getting the VaR forecast as an input. It just memorized the quantile bounds and hit 100% accuracy without using any exposure or volatility features.

**Fix**: Removed the VaR forecast from classifier inputs. Now it has to learn risk regimes from exposures, rolling volatilities, and concentration metrics (HHI).

### 6.3 Temporal validation requirements

Random k-fold cross-validation on time series causes lookahead bias because rolling windows make adjacent snapshots share overlapping data.

**Fix**: Enforced chronological split (first 80% train, last 20% test). This matches real production where models only see past data to predict future states.

### 6.4 Test set label sparsity

With chronological split, the test set is at the end of the timeline where the portfolio is largest. Most labels are "High" or "Critical", so scikit-learn's confusion_matrix would return 2x2 instead of 4x4 and break our evaluation loops.

**Fix**: Explicitly set `labels=[0, 1, 2, 3]` in confusion matrix calls to guarantee 4x4 shape even if some risk classes don't appear in the test slice.

---

## 7. Key Engineering Decisions

**Redis cache with write-through invalidation**: Positions are read frequently. Instead of hitting PostgreSQL every time, we serve from Redis. When a trade commits to PostgreSQL, we delete the corresponding Redis key. Next read misses cache, fetches from DB, and repopulates Redis (TTL 30s).

**Historical VaR with NumPy**: Instead of assuming normal distributions, we compute VaR directly from actual P&L distributions using NumPy percentiles. Rolling 30-day window at 95% confidence. Returns `insufficient_data` flag if fewer than 10 history points.

**Decimal precision for financial data**: Python floats cause cumulative rounding errors. Used `decimal.Decimal` mapped to PostgreSQL `Numeric(18,4)` with `ROUND_HALF_UP` quantization.

---

## 8. Running the System

### 8.1 Local setup

Python 3.10+ required.

```bash
cd trade-risk-monitor
pip install -r requirements.txt
```

### 8.2 Docker stack

Spins up PostgreSQL, Redis, and FastAPI:

```bash
docker-compose up --build -d
```

- FastAPI: `8000`
- PostgreSQL: `5432`
- Redis: `6379`

Shut down:
```bash
docker-compose down
```

### 8.3 Running tests

```bash
# Windows
$env:PYTHONPATH="."
pytest

# macOS/Linux
PYTHONPATH=. pytest
```

### 8.4 Running benchmark

Simulates trades under volatile regimes, trains all models, computes metrics, measures latency:

```bash
python benchmark.py --trades 500 --latency-runs 150
```

---

## 9. API Endpoints

| Method | Route | Description |
| :---: | :--- | :--- |
| `POST` | `/portfolios` | Create portfolio |
| `GET` | `/portfolios/{id}/positions` | Get positions (Redis cache with SQL fallback) |
| `GET` | `/portfolios/{id}/var` | Get historical VaR profile |
| `POST` | `/trades` | Ingest trade, update position, run models |
| `GET` | `/portfolios/{id}/trades` | Get trade history |
| `GET` | `/portfolios/{id}/features/latest` | Latest feature snapshot |
| `GET` | `/portfolios/{id}/features/history` | Paginated feature history |
| `POST` | `/anomaly/train` | Train Isolation Forest |
| `GET` | `/portfolios/{id}/anomaly/score` | Latest anomaly score |
| `POST` | `/var/train` | Train VaR forecasting models |
| `GET` | `/portfolios/{id}/var/forecast` | Latest VaR forecast |
| `POST` | `/risk-classifier/train` | Train risk regime classifiers |
| `GET` | `/portfolios/{id}/risk-regime` | Current risk regime |
| `POST` | `/portfolios/{id}/simulate` | Run mock trade sequence |
| `POST` | `/portfolios/{id}/simulate-data` | Generate synthetic training data |
| `GET` | `/health` | Service health check |

---

## 10. API Examples (curl)

With Docker running or local uvicorn (`uvicorn app.main:app --port 8000` from `trade-risk-monitor/`):

**Create portfolio:**
```bash
curl -X POST http://localhost:8000/portfolios \
  -H "Content-Type: application/json" \
  -d '{"name": "QuantFund Alpha"}'
```

**Generate synthetic training data (500 trades):**
```bash
curl -X POST "http://localhost:8000/portfolios/1/simulate-data?num_trades=500"
```

**Train ML models:**
```bash
curl -X POST http://localhost:8000/anomaly/train -H "Content-Type: application/json" -d '{"contamination": 0.01}'
curl -X POST http://localhost:8000/var/train -H "Content-Type: application/json" -d '{}'
curl -X POST http://localhost:8000/risk-classifier/train -H "Content-Type: application/json" -d '{}'
```

**Ingest a trade:**
```bash
curl -X POST http://localhost:8000/trades \
  -H "Content-Type: application/json" \
  -d '{"portfolio_id": 1, "ticker": "AAPL", "quantity": 1500.0, "price": 175.50, "side": "BUY"}'
```

**Get positions (check cache header):**
```bash
curl -i http://localhost:8000/portfolios/1/positions
```
Look for `X-Cache: HIT` (cached) or `X-Cache: MISS` (DB fetch).

**Get risk assessments:**
```bash
curl http://localhost:8000/portfolios/1/anomaly/score
curl http://localhost:8000/portfolios/1/var/forecast
curl http://localhost:8000/portfolios/1/risk-regime
```

---

## 11. Production Deployment (Render)

**Provision services:**
1. Managed PostgreSQL on Render (save connection string)
2. Managed Redis on Render (save access URL)
3. Web service pointing to repo, root directory = `trade-risk-monitor`

**Environment variables:**

| Variable | Description | Example |
| :--- | :--- | :--- |
| `DATABASE_URL` | PostgreSQL connection | `postgresql://user:pass@host:5432/db` |
| `REDIS_URL` | Redis connection | `redis://user:pass@host:6379` |
| `VAR_THRESHOLD` | VaR alert threshold | `1000000.0000` |
| `CONCENTRATION_THRESHOLD` | Concentration alert threshold | `0.4000` |

**Database migrations:**
```bash
cd trade-risk-monitor
alembic init alembic
alembic revision --autogenerate -m "initial schema"
alembic upgrade head
```