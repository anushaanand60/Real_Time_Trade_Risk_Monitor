import math
import random
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
import numpy as np
from app.models.trade import Trade
from app.models.position import Position
from app.models.portfolio import Portfolio
from app.models.features import FeatureSnapshot
from app.models.alert import Alert
from app.services.position import update_position_logic
from app.services.alerts import run_post_trade_alerts
from app.services.feature_generation import generate_features_for_trade
from app.services.anomaly_detector import score_anomaly
from app.services.var_forecaster import predict_var_forecast
from app.services.risk_classifier import predict_risk_regime

TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

BASE_PRICES = {
    "AAPL": 150.0,
    "GOOGL": 140.0,
    "MSFT": 310.0,
    "AMZN": 180.0,
    "TSLA": 220.0,
}

REGIME_SCHEDULE = [
    (0.00, 0.23, "LOW_VOL",    0.007, 0.00),
    (0.23, 0.36, "CRISIS",     0.048, 0.88),
    (0.36, 0.62, "NORMAL_VOL", 0.018, 0.00),
    (0.62, 0.78, "HIGH_VOL",   0.032, 0.00),
    (0.78, 1.00, "NORMAL_VOL", 0.015, 0.00),
]

EXPOSURE_SHOCK_PROB = 0.03
NORMAL_QTY_RANGE = (10, 100)
SHOCK_QTY_RANGE = (3000, 12000)


def _get_regime(step: int, n_steps: int):
    frac = step / max(n_steps, 1)
    for start, end, name, sigma, beta in REGIME_SCHEDULE:
        if start <= frac < end:
            return name, sigma, beta
    return "NORMAL_VOL", 0.018, 0.00


def _generate_price_paths(n_steps: int) -> dict:
    paths = {t: [BASE_PRICES[t]] for t in TICKERS}
    for step in range(n_steps):
        _, sigma, crisis_beta = _get_regime(step, n_steps)
        if crisis_beta > 0:
            market_factor = np.random.normal(0, sigma)
            for ticker in TICKERS:
                idio_sigma = sigma * 0.25
                ret = crisis_beta * market_factor + np.random.normal(0, idio_sigma)
                new_price = max(paths[ticker][-1] * math.exp(ret), 2.0)
                paths[ticker].append(new_price)
        else:
            for ticker in TICKERS:
                ret = np.random.normal(0, sigma)
                new_price = max(paths[ticker][-1] * math.exp(ret), 2.0)
                paths[ticker].append(new_price)
    return paths


def generate_simulation_data(
    db: Session,
    portfolio_id: int,
    num_trades: int,
    start_date: datetime = None,
) -> None:
    db.query(Alert).filter(Alert.portfolio_id == portfolio_id).delete()
    db.query(Trade).filter(Trade.portfolio_id == portfolio_id).delete()
    db.query(Position).filter(Position.portfolio_id == portfolio_id).delete()
    db.query(FeatureSnapshot).filter(FeatureSnapshot.portfolio_id == portfolio_id).delete()
    db.flush()
    if not db.query(Portfolio).filter(Portfolio.id == portfolio_id).first():
        db.add(Portfolio(id=portfolio_id, name=f"SimPortfolio-{portfolio_id}"))
        db.commit()

    if start_date is None:
        start_date = datetime.now(timezone.utc) - timedelta(days=num_trades)

    price_paths = _generate_price_paths(num_trades)

    for step in range(num_trades):
        ticker = random.choice(TICKERS)
        side = "BUY" if random.random() < 0.70 else "SELL"

        is_exposure_shock = random.random() < EXPOSURE_SHOCK_PROB
        if is_exposure_shock:
            quantity = Decimal(str(random.randint(*SHOCK_QTY_RANGE)))
        else:
            quantity = Decimal(str(random.randint(*NORMAL_QTY_RANGE)))

        price_float = price_paths[ticker][step + 1]
        price = Decimal(str(round(price_float, 4)))

        if side == "SELL":
            existing = db.query(Position).filter(
                Position.portfolio_id == portfolio_id,
                Position.ticker == ticker,
            ).first()
            if existing is None or existing.net_quantity < quantity:
                side = "BUY"

        trade_time = start_date + timedelta(days=step, minutes=random.randint(0, 59))
        trade = Trade(
            portfolio_id=portfolio_id,
            ticker=ticker,
            quantity=quantity,
            price=price,
            side=side,
            timestamp=trade_time,
        )
        db.add(trade)
        db.flush()
        update_position_logic(db, trade)
        run_post_trade_alerts(db, trade)
        snapshot = generate_features_for_trade(db, trade)
        score_anomaly(db, snapshot)
        predict_var_forecast(db, snapshot)
        predict_risk_regime(db, snapshot)
    db.commit()
