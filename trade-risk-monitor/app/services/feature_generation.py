from decimal import Decimal
from datetime import datetime, timezone, timedelta
from sqlalchemy import func
from sqlalchemy.orm import Session
import numpy as np
from app.models.trade import Trade
from app.models.position import Position
from app.models.features import FeatureSnapshot
from app.services.risk import compute_portfolio_var

def get_latest_price(db: Session, portfolio_id: int, ticker: str, trade: Trade) -> Decimal:
    if trade.ticker == ticker:
        return trade.price
    last_trade = db.query(Trade).filter(
        Trade.portfolio_id == portfolio_id,
        Trade.ticker == ticker,
        Trade.id <= trade.id
    ).order_by(Trade.timestamp.desc(), Trade.id.desc()).first()
    return last_trade.price if last_trade else Decimal("0.0000")

def get_ticker_prices(db: Session, portfolio_id: int, ticker: str, trade: Trade, limit: int) -> list[float]:
    trades = db.query(Trade).filter(
        Trade.portfolio_id == portfolio_id,
        Trade.ticker == ticker,
        Trade.id <= trade.id
    ).order_by(Trade.timestamp.desc(), Trade.id.desc()).limit(limit).all()
    return [float(t.price) for t in reversed(trades)]

def get_portfolio_historical_values(db: Session, portfolio_id: int, trade: Trade, current_val: Decimal, limit: int) -> list[float]:
    snapshots = db.query(FeatureSnapshot).filter(
        FeatureSnapshot.portfolio_id == portfolio_id,
        FeatureSnapshot.snapshot_type == "PORTFOLIO",
        FeatureSnapshot.timestamp < trade.timestamp
    ).order_by(FeatureSnapshot.timestamp.desc()).limit(limit).all()
    if len(snapshots) >= limit - 1:
        return [float(s.exp_gross_exposure) for s in reversed(snapshots)] + [float(current_val)]
    trades = db.query(Trade).filter(
        Trade.portfolio_id == portfolio_id,
        Trade.timestamp <= trade.timestamp
    ).order_by(Trade.timestamp.asc()).all()
    pos_qtys = {}
    values = []
    latest_prices = {}
    for t in trades:
        qty_diff = t.quantity if t.side == "BUY" else -t.quantity
        pos_qtys[t.ticker] = pos_qtys.get(t.ticker, Decimal("0")) + qty_diff
        latest_prices[t.ticker] = t.price
        val = sum(pos_qtys[tick] * latest_prices[tick] for tick in pos_qtys)
        values.append(float(val))
    return values[-limit:]

def get_volatility(prices: list[float]) -> float:
    if len(prices) < 2:
        return 0.0
    returns = []
    for i in range(1, len(prices)):
        p_prev = prices[i-1]
        p_curr = prices[i]
        if p_prev > 0 and p_curr > 0:
            returns.append(np.log(p_curr / p_prev))
    if len(returns) < 1:
        return 0.0
    return float(np.std(returns))

def generate_features_for_trade(db: Session, trade: Trade):
    db.flush()
    portfolio_id = trade.portfolio_id
    positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
    mark_prices = {}
    position_values = {}
    total_portfolio_value = Decimal("0.0000")
    for pos in positions:
        price = get_latest_price(db, portfolio_id, pos.ticker, trade)
        mark_prices[pos.ticker] = price
        val = pos.net_quantity * price
        position_values[pos.ticker] = val
        total_portfolio_value += abs(val)
    one_hour_ago = trade.timestamp - timedelta(hours=1)
    port_trade_count = db.query(Trade).filter(
        Trade.portfolio_id == portfolio_id,
        Trade.timestamp >= one_hour_ago,
        Trade.timestamp <= trade.timestamp
    ).count()
    port_avg_size = db.query(func.avg(Trade.quantity)).filter(
        Trade.portfolio_id == portfolio_id,
        Trade.timestamp >= one_hour_ago,
        Trade.timestamp <= trade.timestamp
    ).scalar() or Decimal("0.0000")
    port_volume = db.query(func.sum(Trade.quantity * Trade.price)).filter(
        Trade.portfolio_id == portfolio_id,
        Trade.timestamp >= one_hour_ago,
        Trade.timestamp <= trade.timestamp
    ).scalar() or Decimal("0.0000")
    port_values_5t = get_portfolio_historical_values(db, portfolio_id, trade, total_portfolio_value, 6)
    port_vol_5t = get_volatility(port_values_5t)
    port_values_30t = get_portfolio_historical_values(db, portfolio_id, trade, total_portfolio_value, 31)
    port_vol_30t = get_volatility(port_values_30t)
    var_res = compute_portfolio_var(db, portfolio_id, as_of=trade.timestamp)
    var_95 = Decimal(str(var_res["var_value"])) if not var_res["insufficient_data"] else Decimal("0.0000")
    hhi = Decimal("0.0000")
    if total_portfolio_value > 0:
        for pos in positions:
            weight = abs(position_values[pos.ticker]) / total_portfolio_value
            hhi += weight * weight
    portfolio_snapshot = FeatureSnapshot(
        portfolio_id=portfolio_id,
        ticker=None,
        trade_id=trade.id,
        timestamp=trade.timestamp,
        snapshot_type="PORTFOLIO",
        exp_net_exposure=sum(position_values.values()),
        exp_gross_exposure=total_portfolio_value,
        exp_weight=Decimal("1.0000"),
        beh_trade_count_1h=port_trade_count,
        beh_avg_trade_size=Decimal(str(port_avg_size)),
        beh_volume_1h=Decimal(str(port_volume)),
        pos_net_quantity=None,
        pos_avg_price=None,
        pos_unrealized_pnl=None,
        risk_var_95=var_95,
        risk_hhi_concentration=hhi,
        vol_rolling_volatility_5t=Decimal(str(port_vol_5t)),
        vol_rolling_volatility_30t=Decimal(str(port_vol_30t))
    )
    db.add(portfolio_snapshot)
    for pos in positions:
        ticker = pos.ticker
        net_qty = pos.net_quantity
        avg_price = pos.avg_price
        unrealized_pnl = pos.unrealized_pnl
        pos_val = position_values[ticker]
        weight = abs(pos_val) / total_portfolio_value if total_portfolio_value > 0 else Decimal("0.0000")
        tick_trade_count = db.query(Trade).filter(
            Trade.portfolio_id == portfolio_id,
            Trade.ticker == ticker,
            Trade.timestamp >= one_hour_ago,
            Trade.timestamp <= trade.timestamp
        ).count()
        tick_avg_size = db.query(func.avg(Trade.quantity)).filter(
            Trade.portfolio_id == portfolio_id,
            Trade.ticker == ticker,
            Trade.timestamp >= one_hour_ago,
            Trade.timestamp <= trade.timestamp
        ).scalar() or Decimal("0.0000")
        tick_volume = db.query(func.sum(Trade.quantity * Trade.price)).filter(
            Trade.portfolio_id == portfolio_id,
            Trade.ticker == ticker,
            Trade.timestamp >= one_hour_ago,
            Trade.timestamp <= trade.timestamp
        ).scalar() or Decimal("0.0000")
        prices_5t = get_ticker_prices(db, portfolio_id, ticker, trade, 6)
        tick_vol_5t = get_volatility(prices_5t)
        prices_30t = get_ticker_prices(db, portfolio_id, ticker, trade, 31)
        tick_vol_30t = get_volatility(prices_30t)
        position_snapshot = FeatureSnapshot(
            portfolio_id=portfolio_id,
            ticker=ticker,
            trade_id=trade.id,
            timestamp=trade.timestamp,
            snapshot_type="POSITION",
            exp_net_exposure=pos_val,
            exp_gross_exposure=abs(pos_val),
            exp_weight=weight,
            beh_trade_count_1h=tick_trade_count,
            beh_avg_trade_size=Decimal(str(tick_avg_size)),
            beh_volume_1h=Decimal(str(tick_volume)),
            pos_net_quantity=net_qty,
            pos_avg_price=avg_price,
            pos_unrealized_pnl=unrealized_pnl,
            risk_var_95=None,
            risk_hhi_concentration=None,
            vol_rolling_volatility_5t=Decimal(str(tick_vol_5t)),
            vol_rolling_volatility_30t=Decimal(str(tick_vol_30t))
        )
        db.add(position_snapshot)
    return portfolio_snapshot
