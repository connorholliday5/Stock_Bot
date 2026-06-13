"""
tests/test_phase4.py
Phase 4 - Crypto 24/7 strategy tests.

Network (CCXT) is never touched: all frames are synthetic. The DB tests run against
an in-memory SQLite bound to the REAL db.Base metadata, so they exercise the actual
SQLAlchemy Enum binding and prove the SignalType -> SignalDirection fix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import database.db as db
import strategies.signals as signals
from database.db import Signal, Position, AssetType, OrderSide, OrderStatus, SignalDirection, MarketRegime
from data.fetcher import add_features
from strategies.signals import SignalEvent, SignalType, AssetClass, persist_signals
from strategies import crypto_24h as c24
from strategies.crypto_24h import (
    classify_regime, above_ema200, bullish_ema_cross, bearish_ema_cross,
    macd_confirms_long, half_kelly_fraction, compute_position_size,
    generate_24h_entries, generate_24h_exits, run_crypto_24h_pipeline,
    MAX_POSITION_NOTIONAL_PCT, MIN_POSITION_USD,
)
from execution.ccxt_crypto import CryptoExecutor


# ---------------------------------------------------------------------------
# Synthetic data builders (deterministic)
# ---------------------------------------------------------------------------

def make_trending_4h(n=360, seed=1):
    rng = np.random.default_rng(seed)
    base = np.linspace(200, 520, n)
    noise = rng.normal(0, 0.4, n).cumsum() * 0.10
    price = base + noise
    price[-17:-3] -= np.linspace(26 * 0.2, 26, 14)
    price[-3:] += np.array([8, 12, 16])
    close = pd.Series(price)
    high, low = close * 1.003, close * 0.997
    openp = close.shift(1).fillna(close.iloc[0])
    vol = pd.Series(rng.uniform(950, 1050, n))
    vol.iloc[-1] = 2000
    idx = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    return add_features(pd.DataFrame(
        {"open": openp.values, "high": high.values, "low": low.values,
         "close": close.values, "volume": vol.values}, index=idx))


def make_choppy_4h(n=320, seed=2):
    rng = np.random.default_rng(seed)
    price = 100 + rng.normal(0, 1, n).cumsum() * 0.05
    c = pd.Series(price)
    idx = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    return add_features(pd.DataFrame(
        {"open": c.values, "high": (c * 1.002).values, "low": (c * 0.998).values,
         "close": c.values, "volume": rng.uniform(900, 1100, n)}, index=idx))


def make_highvol_4h(n=320, seed=3):
    rng = np.random.default_rng(seed)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.06, n)))
    c = pd.Series(price)
    idx = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    return add_features(pd.DataFrame(
        {"open": c.values, "high": (c * 1.03).values, "low": (c * 0.97).values,
         "close": c.values, "volume": rng.uniform(900, 1100, n)}, index=idx))


def make_downtrend_4h(n=320, seed=4):
    rng = np.random.default_rng(seed)
    price = np.linspace(500, 200, n) + rng.normal(0, 0.4, n).cumsum() * 0.1
    c = pd.Series(price)
    idx = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    return add_features(pd.DataFrame(
        {"open": c.values, "high": (c * 1.003).values, "low": (c * 0.997).values,
         "close": c.values, "volume": rng.uniform(900, 1100, n)}, index=idx))


# ---------------------------------------------------------------------------
# DB fixture: real schema on in-memory SQLite, patched into both modules
# ---------------------------------------------------------------------------

@pytest.fixture
def sqlite_db(monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    db.Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    monkeypatch.setattr(db, "SessionLocal", TestSession)
    monkeypatch.setattr(signals, "SessionLocal", TestSession)
    yield TestSession
    db.Base.metadata.drop_all(engine)


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def test_indicator_columns_present():
    df = make_trending_4h()
    for col in ["ema_9", "ema_21", "ema_200", "macd", "macd_signal", "macd_hist", "adx", "realized_vol"]:
        assert col in df.columns, f"missing {col}"


def test_indicator_values_sane():
    df = make_trending_4h()
    adx = df["adx"].dropna()
    assert (adx >= 0).all() and (adx <= 100).all()
    assert (df["realized_vol"].dropna() >= 0).all()
    # ema_200 needs 200 obs; first 199 are NaN
    assert df["ema_200"].isna().sum() == 199
    last = df.iloc[-1]
    assert abs(last["macd_hist"] - (last["macd"] - last["macd_signal"])) < 1e-9


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def test_regime_trending():
    assert classify_regime(make_trending_4h()) == MarketRegime.TRENDING


def test_regime_mean_reverting():
    assert classify_regime(make_choppy_4h()) == MarketRegime.MEAN_REVERTING


def test_regime_high_vol():
    assert classify_regime(make_highvol_4h()) == MarketRegime.HIGH_VOL


def test_regime_empty_is_unknown():
    assert classify_regime(pd.DataFrame()) == MarketRegime.UNKNOWN


# ---------------------------------------------------------------------------
# Entry gates
# ---------------------------------------------------------------------------

def test_full_entry_triggers():
    df = make_trending_4h()
    entries = generate_24h_entries(
        universe={"BTC/USDT": df}, funding_rates={"BTC/USDT": 0.0001},
        open_positions=[], equity=10_000.0, available_cash=10_000.0, btc_only=True)
    assert len(entries) == 1
    e = entries[0]
    assert e.symbol == "BTC/USDT"
    assert e.stop_price < e.entry_price < e.take_profit
    assert e.units > 0


def test_ema200_gate_blocks_downtrend():
    df = make_downtrend_4h()
    assert above_ema200(df) is False
    entries = generate_24h_entries(
        universe={"BTC/USDT": df}, funding_rates={}, open_positions=[],
        equity=10_000.0, available_cash=10_000.0, btc_only=True)
    assert entries == []


def test_high_vol_regime_blocks_entry():
    df = make_highvol_4h()
    entries = generate_24h_entries(
        universe={"BTC/USDT": df}, funding_rates={}, open_positions=[],
        equity=10_000.0, available_cash=10_000.0, btc_only=True)
    assert entries == []


def test_btc_only_gate():
    df = make_trending_4h()
    blocked = generate_24h_entries(
        universe={"ETH/USDT": df}, funding_rates={"ETH/USDT": 0.0}, open_positions=[],
        equity=10_000.0, available_cash=10_000.0, btc_only=True)
    assert blocked == []
    allowed = generate_24h_entries(
        universe={"ETH/USDT": df}, funding_rates={"ETH/USDT": 0.0}, open_positions=[],
        equity=10_000.0, available_cash=10_000.0, btc_only=False)
    assert len(allowed) == 1 and allowed[0].symbol == "ETH/USDT"


def test_max_positions_cap():
    df = make_trending_4h()
    held = [{"symbol": "ETH/USDT"}, {"symbol": "SOL/USDT"}, {"symbol": "BNB/USDT"}]
    entries = generate_24h_entries(
        universe={"BTC/USDT": df}, funding_rates={"BTC/USDT": 0.0}, open_positions=held,
        equity=10_000.0, available_cash=10_000.0, btc_only=False, max_positions=3)
    assert entries == []


def test_already_held_skipped():
    df = make_trending_4h()
    entries = generate_24h_entries(
        universe={"BTC/USDT": df}, funding_rates={"BTC/USDT": 0.0},
        open_positions=[{"symbol": "BTC/USDT"}],
        equity=10_000.0, available_cash=10_000.0, btc_only=True)
    assert entries == []


# ---------------------------------------------------------------------------
# Sizing and Kelly
# ---------------------------------------------------------------------------

def test_half_kelly_positive_and_negative_edge():
    assert half_kelly_fraction(0.5, 1.5) == pytest.approx((0.5 - 0.5 / 1.5) / 2)
    assert half_kelly_fraction(0.4, 1.0) == 0.0   # negative edge -> no allocation
    assert half_kelly_fraction(0.5, 0.0) == 0.0   # guard


def test_sizing_risk_and_caps():
    # entry 100, atr 1 => stop_distance 2, stop 98. equity 10k.
    r = compute_position_size(equity=10_000, available_cash=10_000, entry_price=100, atr=1.0)
    # effective risk = min(1.5%, half_kelly(0.5,1.5)=8.3%) = 1.5% => risk $150 over $2 stop = 75 units
    # but 35% equity cap = $3500 => 35 units
    assert r.notional == pytest.approx(3500.0)
    assert r.units == pytest.approx(35.0)
    assert r.stop_price == pytest.approx(98.0)
    assert r.tradable


def test_sizing_no_margin_cash_cap():
    r = compute_position_size(equity=10_000, available_cash=500, entry_price=100, atr=1.0)
    assert r.notional <= 500 + 1e-9
    assert r.units == pytest.approx(5.0)


def test_sizing_below_min_position_not_tradable():
    r = compute_position_size(equity=10_000, available_cash=40, entry_price=100, atr=1.0)
    assert not r.tradable


def test_funding_negative_haircut_and_skip():
    full = compute_position_size(equity=10_000, available_cash=10_000, entry_price=100, atr=1.0, funding_rate=0.0)
    haircut = compute_position_size(equity=10_000, available_cash=10_000, entry_price=100, atr=1.0, funding_rate=-0.0005)
    assert haircut.effective_risk_pct == pytest.approx(full.effective_risk_pct * 0.5)
    skipped = compute_position_size(equity=10_000, available_cash=10_000, entry_price=100, atr=1.0, funding_rate=-0.01)
    assert not skipped.tradable and "funding" in skipped.reason


# ---------------------------------------------------------------------------
# Exits
# ---------------------------------------------------------------------------

def test_exit_below_ema200():
    df = make_downtrend_4h()
    exits = generate_24h_exits([{"symbol": "BTC/USDT", "entry_price": 600.0}], {"BTC/USDT": df})
    assert len(exits) == 1 and exits[0].reason == "below_ema200"


def test_exit_trailing_stop_breach():
    df = make_trending_4h()
    last_close = float(df["close"].iloc[-1])
    # stop just above current price forces a trailing-stop breach while trend intact
    exits = generate_24h_exits(
        [{"symbol": "BTC/USDT", "entry_price": last_close, "stop_loss": last_close + 5}],
        {"BTC/USDT": df})
    assert len(exits) == 1 and "trailing_stop_breach" in exits[0].reason


def test_no_exit_when_trend_intact():
    df = make_trending_4h()
    last_close = float(df["close"].iloc[-1])
    exits = generate_24h_exits(
        [{"symbol": "BTC/USDT", "entry_price": last_close, "stop_loss": last_close - 50}],
        {"BTC/USDT": df})
    assert exits == []


# ---------------------------------------------------------------------------
# Persistence: the SignalType -> SignalDirection fix (the core bug)
# ---------------------------------------------------------------------------

def _crypto_event(signal_type):
    return SignalEvent(
        ticker="BTC/USDT", signal_type=signal_type, asset_class=AssetClass.CRYPTO,
        composite_score=0.6, rsi=55.0, momentum=0.02, volume_ratio=1.8,
        trend_score=1.0, close_price=500.0, atr=5.0, adx=42.0)


def test_persist_buy_maps_to_long(sqlite_db):
    n = persist_signals([_crypto_event(SignalType.BUY)])
    assert n == 1
    with sqlite_db() as s:
        row = s.query(Signal).one()
        assert row.direction == SignalDirection.LONG
        assert row.asset_type == AssetType.CRYPTO
        assert row.adx == pytest.approx(42.0)


def test_persist_sell_maps_to_flat(sqlite_db):
    persist_signals([_crypto_event(SignalType.SELL)])
    with sqlite_db() as s:
        row = s.query(Signal).one()
        assert row.direction == SignalDirection.FLAT


def test_original_buy_string_would_fail(sqlite_db):
    # Documents the original bug: writing the raw SignalType value is rejected.
    from sqlalchemy.exc import StatementError
    with pytest.raises((StatementError, LookupError)):
        sess = sqlite_db()
        sess.add(Signal(symbol="BTC/USDT", direction="BUY", asset_type="crypto"))
        sess.commit()
        sess.query(Signal).all()   # read-back triggers the enum lookup that rejects "BUY"
        sess.close()


def test_persist_dedup_same_day(sqlite_db):
    persist_signals([_crypto_event(SignalType.BUY)])
    n2 = persist_signals([_crypto_event(SignalType.BUY)])
    assert n2 == 0
    with sqlite_db() as s:
        assert s.query(Signal).count() == 1


# ---------------------------------------------------------------------------
# Orchestrator + executor
# ---------------------------------------------------------------------------

def test_pipeline_persists_and_returns_plans(sqlite_db):
    df = make_trending_4h()
    result = run_crypto_24h_pipeline(
        universe={"BTC/USDT": df}, funding_rates={"BTC/USDT": 0.0001},
        open_positions=[], equity=10_000.0, available_cash=10_000.0, btc_only=True)
    assert len(result.entries) == 1
    assert result.persisted == 1
    with sqlite_db() as s:
        assert s.query(Signal).filter_by(direction=SignalDirection.LONG).count() == 1


def test_executor_paper_open_and_close(sqlite_db):
    ex = CryptoExecutor(paper=True, paper_cash=10_000.0)
    opened = ex.open_long("BTC/USDT", units=10.0, entry_price=100.0, stop_loss=96.0, take_profit=112.0)
    assert opened["status"] == "filled"
    with sqlite_db() as s:
        assert s.query(Position).filter_by(symbol="BTC/USDT").count() == 1
    closed = ex.close_long("BTC/USDT", exit_price=110.0, reason="take_profit")
    assert closed["status"] == "filled"
    assert closed["pnl"] > 0  # 10 units * $10 gain, net of fees
    with sqlite_db() as s:
        assert s.query(Position).filter_by(symbol="BTC/USDT").count() == 0


def test_executor_max_positions_backstop(sqlite_db):
    ex = CryptoExecutor(paper=True, paper_cash=10_000.0, max_positions=1)
    ex.open_long("BTC/USDT", units=10.0, entry_price=100.0, stop_loss=96.0, take_profit=112.0)
    rejected = ex.open_long("ETH/USDT", units=10.0, entry_price=100.0, stop_loss=96.0, take_profit=112.0)
    assert rejected["status"] == "rejected" and rejected["reason"] == "max_positions"


def test_executor_min_notional_backstop(sqlite_db):
    ex = CryptoExecutor(paper=True, paper_cash=10_000.0)
    rejected = ex.open_long("BTC/USDT", units=0.1, entry_price=100.0, stop_loss=96.0, take_profit=112.0)
    assert rejected["status"] == "rejected" and rejected["reason"] == "below_min_notional"