# Real-Time Trade Risk Monitor

A high-throughput, low-latency portfolio risk monitoring infrastructure engineered for institutional-grade trade lifecycle management. Built on FastAPI, PostgreSQL, and Redis to deliver real-time trade ingestion, automated position aggregation, historical Value-at-Risk computation, concentration breach detection, and dynamic risk alerting across multi-asset equity portfolios.

The system processes live trade events through a deterministic pipeline that logs transactions, recalculates portfolio-level risk metrics on every state mutation, invalidates stale cache entries, and serves fresh position snapshots from an in-memory key-value store with sub-millisecond read latency.

---

## Architecture

![System Architecture](images/architecture.png)

**Request Flow:** Live Trade Event → FastAPI Router → Transaction Logging (PostgreSQL) + Real-Time Aggregation Engine → Cache Invalidation → In-Memory State Serving (Redis)

---

## Quick Start

### Prerequisites

- Docker and Docker Compose installed on the host machine.

### Launch the Full Stack

```bash
cd trade-risk-monitor
docker-compose up --build -d
```

This provisions three isolated containers: the FastAPI web server on port 8000, PostgreSQL 15 on port 5432, and Redis 7 on port 6379. The web container waits for both database and cache healthchecks to pass before accepting traffic.

### Create a Portfolio

```bash
curl -X POST http://localhost:8000/portfolios \
  -H "Content-Type: application/json" \
  -d '{"name": "QuantFund Alpha"}'
```

### Run the 100-Trade Bulk Simulation

```bash
curl -X POST http://localhost:8000/portfolios/1/simulate
```

This wipes existing portfolio state, executes 100 randomized trades across AAPL, GOOGL, MSFT, AMZN, and TSLA through the full ingestion pipeline, and returns a summary payload containing final positions, net worth valuation, VaR metrics, and all triggered risk alerts.

### Query Cached Positions

```bash
curl -i http://localhost:8000/portfolios/1/positions
```

Inspect the `X-Cache` response header: `HIT` indicates a Redis cache serve, `MISS` indicates a fresh PostgreSQL query with subsequent cache population.

### Check Portfolio Value-at-Risk

```bash
curl http://localhost:8000/portfolios/1/var
```

### Submit an Individual Trade

```bash
curl -X POST http://localhost:8000/trades \
  -H "Content-Type: application/json" \
  -d '{"portfolio_id": 1, "ticker": "AAPL", "quantity": "50", "price": "178.2300", "side": "BUY"}'
```

### Verify System Health

```bash
curl http://localhost:8000/health
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/portfolios` | Create a new portfolio |
| `POST` | `/trades` | Ingest a trade, update positions, evaluate alerts |
| `GET` | `/portfolios/{id}/positions` | Fetch cached positions (Redis → PostgreSQL fallback) |
| `GET` | `/portfolios/{id}/trades` | Paginated trade history |
| `GET` | `/portfolios/{id}/var` | Historical Value-at-Risk calculation |
| `POST` | `/portfolios/{id}/simulate` | Execute 100-trade bulk simulation |
| `GET` | `/health` | Container healthcheck probe |

---

## Architectural Design Rationale

### Redis for High-Throughput Position Read Isolation

Position queries represent the highest-frequency read pattern in a live trading system. Serving these directly from PostgreSQL under concurrent load introduces connection pool contention and query execution latency that compounds at scale. Redis provides deterministic O(1) key lookups with sub-millisecond response times, decoupling read-heavy position queries from the transactional write path entirely.

The cache invalidation strategy is write-through: every trade ingestion event atomically deletes the corresponding portfolio position cache key after the database transaction commits. Subsequent reads trigger a cache miss, repopulate from PostgreSQL, and serve all following requests from memory until the 30-second TTL expires or another trade invalidates the entry. This guarantees eventual consistency within a bounded staleness window while eliminating redundant database round-trips for repeated position lookups.

The `X-Cache: HIT/MISS` response header provides operational visibility into cache utilization rates without requiring separate monitoring infrastructure.

### Empirical Historical VaR Over Statistical Assumptions

The Value-at-Risk implementation uses the Historical Simulation method rather than parametric (variance-covariance) or Monte Carlo approaches. Historical VaR computes risk directly from observed P&L distributions using numpy percentile calculations, making zero assumptions about the normality of return distributions.

This is a deliberate engineering decision. Parametric VaR assumes normally distributed returns, which systematically underestimates tail risk during market stress events where fat tails and volatility clustering dominate. Historical VaR captures the actual empirical distribution of portfolio-level daily P&L changes, including any skewness, kurtosis, or regime-dependent behavior present in the observed data.

The 30-day rolling window with a 95% confidence level balances responsiveness to recent market conditions against statistical stability. The minimum threshold of 10 daily P&L observations prevents spurious VaR estimates from insufficient sample sizes by returning an explicit `insufficient_data` flag rather than a misleading numeric value.

### Financial-Grade Decimal Precision Enforcement

All monetary calculations use Python's `decimal.Decimal` type with explicit 4-decimal-place precision (`Numeric(18, 4)` at the database layer). IEEE 754 floating-point arithmetic introduces representation errors that compound through sequential position averaging, P&L aggregation, and concentration ratio calculations. A single trade priced at `178.23` can accumulate rounding drift across hundreds of position updates, producing portfolio valuations that diverge measurably from the mathematically correct result.

Decimal arithmetic eliminates this class of errors entirely. The `ROUND_HALF_UP` quantization policy ensures deterministic rounding behavior consistent with financial accounting standards. Every value crossing a system boundary — from trade ingestion through position compounding to VaR output — maintains exact decimal representation without silent precision loss.

---

## Production Deployment on Render

### Infrastructure Setup

1. **PostgreSQL Database**: Create a managed PostgreSQL instance on Render. Copy the internal connection string provided by the dashboard.

2. **Redis Instance**: Create a managed Redis instance on Render. Copy the internal Redis URL.

3. **Web Service**: Create a new Web Service pointing to this repository. Set the root directory to `trade-risk-monitor` and configure the build and start commands:

   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Environment Variables

Configure the following environment variables on the Render web service dashboard:

| Variable | Value |
|----------|-------|
| `DATABASE_URL` | `postgresql://user:password@host:5432/dbname` (from Render PostgreSQL) |
| `REDIS_URL` | `redis://user:password@host:6379` (from Render Redis) |
| `VAR_THRESHOLD` | `1000000.0000` |
| `CONCENTRATION_THRESHOLD` | `0.4000` |

### Database Migration with Alembic

After configuring the production database, run the initial schema migration:

```bash
cd trade-risk-monitor
alembic init alembic
```

Update `alembic/env.py` to import your SQLAlchemy `Base` metadata and `DATABASE_URL` from the application config. Then generate and apply the migration:

```bash
alembic revision --autogenerate -m "initial schema"
alembic upgrade head
```

For subsequent deployments, run `alembic upgrade head` as part of the build or release phase to apply any pending schema changes to the production database safely.

---

## Future Engineering Enhancements

### Real-Time WebSocket Streaming

Replace the current request-response polling model with persistent WebSocket connections that push position updates, VaR recalculations, and alert notifications to connected clients the instant a trade is processed. This eliminates polling latency entirely and enables live dashboard rendering where portfolio metrics update in-place as the order book evolves. FastAPI's native WebSocket support combined with Redis Pub/Sub as the broadcast backbone would deliver fan-out to multiple concurrent client sessions without coupling the trade ingestion path to the notification delivery path.

### Event-Driven Architecture with Kafka

Introduce Apache Kafka as an intermediate message broker between the trade ingestion layer and the downstream analytics pipeline. Each incoming trade publishes an event to a partitioned Kafka topic, decoupling the synchronous HTTP response from the asynchronous position recalculation, VaR computation, and alert evaluation. This architecture enables horizontal scaling of analytics consumers independently from the API layer, provides durable event replay for auditing and backtesting, and guarantees exactly-once processing semantics for trade lifecycle events through consumer group offset management.

### Advanced Risk Model Calculations

Extend the current Historical VaR implementation with two complementary methodologies. **Variance-Covariance VaR** would model the joint distribution of asset returns using a covariance matrix estimated from historical price data, enabling faster computation for large multi-asset portfolios where the full historical simulation becomes computationally expensive. **Monte Carlo VaR** would generate thousands of simulated portfolio return paths using stochastic processes calibrated to observed volatility and correlation structures, capturing non-linear risk exposures from options and derivatives that analytical methods cannot represent. Implementing all three approaches in parallel would allow risk managers to compare model outputs and identify divergences that signal regime changes or model specification errors.

---

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| API Framework | FastAPI | 0.115.0 |
| ASGI Server | Uvicorn | 0.30.6 |
| ORM | SQLAlchemy | 2.0.35 |
| Validation | Pydantic | 2.9.2 |
| Database | PostgreSQL | 15 |
| Cache | Redis | 7 |
| Numerical Computing | NumPy | 1.26.4 |
| Migrations | Alembic | 1.13.2 |
| Containerization | Docker + Compose | 3.9 |

---

## Project Structure

```
trade-risk-monitor/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── database.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── redis.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── portfolio.py
│   │   ├── trade.py
│   │   ├── position.py
│   │   └── alert.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── portfolio.py
│   │   ├── trade.py
│   │   └── position.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── portfolio.py
│   │   ├── trade.py
│   │   └── simulate.py
│   └── services/
│       ├── __init__.py
│       ├── position.py
│       ├── risk.py
│       └── alerts.py
├── tests/
│   └── test_risk_monitor.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .dockerignore
```
