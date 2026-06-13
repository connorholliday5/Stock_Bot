"""
tests/test_phase5.py
Phase 5 - Risk management. Fully offline: synthetic frames, no network.

DB-backed tests run the REAL schema (database/db.py) on an in-memory SQLite database
via a StaticPool, so the actual Enum bindings and persistence paths are exercised.
"""

from __future__ import annotations

import os

# Dummy env so config.settings (required fields) imports cleanly before any project import.
for _k, _v in {
    "ALPACA_API_KEY": "test", "ALPACA_SECRET_KEY": "test",
    "POLYGON_API_KEY": "test", "BINANCE_API_KEY": "test", "BINANCE_SECRET_KEY": "test",
    "DB_PASSWORD": "test", "TELEGRAM_BOT_TOKEN": "test", "TELEGRAM_CHAT_ID": "test",
    "STARTING_CAPITAL": "10000",
}.items():
    os.environ.setdefault(_k, _v)

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Bind the real schema to a shared in-memory SQLite engine BEFORE importing modules
# that capture SessionLocal / get_db.
import database.db as dbmod

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_TestSession = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
dbmod.engine = _engine
dbmod.SessionLocal = _TestSession
dbmod.Base.metadata.create_all(_engine)

import strategies.signals as signals_mod
signals_mod.SessionLocal = _TestSession  # persist_signals binds SessionLocal by name

from database.db import (
    AssetType, MarketRegime, OrderSide, OrderStatus, Position, Signal, Trade,
)
from risk.manager import (
    RiskManager, half_kelly_fraction, size_position, stock_params, crypto_params,
    check_group_exposure,
)
from execution.alpaca import StockExecutor
from strategies import crypto_24h, stock_weekly

UTC = timezone.utc


@pytest.fixture(autouse=True)
def clean_db():
    """Wipe mutable tables before each test for isolation."""
    with dbmod.get_db() as db:
        db.query(Position).delete()
        db.query(Trade).delete()
        db.query(Signal).delete()
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_position(symbol, asset_type, entry, stop, qty, current=None):
    with dbmod.get_db() as db:
        db.add(Position(
            asset_type=asset_type, symbol=symbol, quantity=qty,
            entry_price=entry, current_price=current if current is not None else entry,
            stop_loss=stop, take_profit=entry * 1.1,
            opened_at=datetime.now(UTC),
        ))


def _add_closed_trade(symbol, asset_type, pnl, closed_at):
    with dbmod.get_db() as db:
        db.add(Trade(
            asset_type=asset_type, symbol=symbol, side=OrderSide.SELL,
            quantity=1.0, entry_price=100.0, exit_price=100.0 + pnl,
            status=OrderStatus.FILLED, pnl=pnl, pnl_pct=pnl / 100.0,
            opened_at=closed_at - timedelta(days=1), closed_at=closed_at,
        ))


def _stock_feature_df(close, rsi=50.0, vol=1.2, atr=2.0, above50=1):
    return pd.DataFrame({
        "close": [close], "rsi": [rsi], "volume_ratio": [vol], "atr": [atr],
        "above_sma50": [above50], "momentum": [0.05], "sma_50": [close * 0.98],
        "sma_200": [close * 0.9],
    })


# ===========================================================================
# half-Kelly
# ===========================================================================

def test_half_kelly_positive_edge():
    # f* = 0.5 - 0.5/1.5 = 0.16667; half = 0.08333
    assert half_kelly_fraction(0.5, 1.5) == pytest.approx(0.083333, abs=1e-5)


def test_half_kelly_negative_edge_floors_zero():
    assert half_kelly_fraction(0.2, 1.5) == 0.0


def test_half_kelly_bad_ratio():
    assert half_kelly_fraction(0.6, 0.0) == 0.0


def test_crypto_half_kelly_delegates_to_engine():
    assert crypto_24h.half_kelly_fraction(0.5, 1.5) == half_kelly_fraction(0.5, 1.5)


# ===========================================================================
# size_position - generic engine
# ===========================================================================

def test_size_stock_basic():
    ps = size_position(
        stock_params(), equity=10000, available_cash=10000,
        entry_price=100.0, stop_price=95.0, tp_r_mult=2.0,
    )
    assert ps.tradable
    assert ps.effective_risk_pct == pytest.approx(0.02)
    assert ps.units == pytest.approx(40.0, abs=1e-6)   # 200 risk / 5 stop
    assert ps.stop_price == pytest.approx(95.0)
    assert ps.take_profit == pytest.approx(110.0)


def test_size_heat_clamp():
    ps = size_position(
        stock_params(), equity=10000, available_cash=10000,
        entry_price=100.0, stop_price=95.0, tp_r_mult=2.0, current_heat=0.095,
    )
    # room = 0.10 - 0.095 = 0.005
    assert ps.effective_risk_pct == pytest.approx(0.005)
    assert ps.units == pytest.approx(10.0, abs=1e-6)


def test_size_heat_exceeded():
    ps = size_position(
        stock_params(), equity=10000, available_cash=10000,
        entry_price=100.0, stop_price=95.0, current_heat=0.10,
    )
    assert not ps.tradable
    assert ps.reason == "heat_exceeded"


def test_size_min_floor():
    ps = size_position(
        stock_params(), equity=10000, available_cash=30.0,
        entry_price=100.0, stop_price=95.0,
    )
    assert not ps.tradable
    assert ps.reason == "below_min_position"


def test_size_regime_high_vol_halves():
    full = size_position(stock_params(), 10000, 10000, 100.0, stop_price=95.0,
                         regime=MarketRegime.TRENDING)
    hv = size_position(stock_params(), 10000, 10000, 100.0, stop_price=95.0,
                       regime=MarketRegime.HIGH_VOL)
    assert hv.effective_risk_pct == pytest.approx(full.effective_risk_pct * 0.5)


def test_size_invalid_inputs():
    assert not size_position(stock_params(), 0, 1000, 100.0, stop_price=95.0).tradable
    assert not size_position(stock_params(), 1000, 1000, 0.0, stop_price=95.0).tradable
    assert size_position(stock_params(), 1000, 1000, 100.0).reason == "invalid_inputs"  # no stop/atr


def test_size_cash_ceiling_no_margin():
    ps = size_position(
        crypto_params(), equity=100000, available_cash=500.0,
        entry_price=100.0, atr=1.0,
    )
    # huge equity would size large, but cash caps notional at <= 500
    assert ps.notional <= 500.0 + 1e-6


# ===========================================================================
# crypto compute_position_size - integration preserves Phase 4 math
# ===========================================================================

def test_crypto_sizing_concentration_cap():
    s = crypto_24h.compute_position_size(
        equity=10000, available_cash=10000, entry_price=100.0, atr=1.0,
    )
    # 1.5% of 10k = 150 risk / 2 stop = 75 units -> $7500, capped at 35% = $3500 -> 35 units
    assert s.units == pytest.approx(35.0, abs=1e-6)
    assert s.notional == pytest.approx(3500.0, abs=1e-3)
    assert s.stop_price == pytest.approx(98.0)
    assert s.take_profit == pytest.approx(106.0)
    assert s.reason == "ok"


def test_crypto_sizing_uncapped():
    s = crypto_24h.compute_position_size(
        equity=1000, available_cash=1000, entry_price=100.0, atr=5.0,
    )
    assert s.units == pytest.approx(1.5, abs=1e-6)
    assert s.notional == pytest.approx(150.0, abs=1e-6)


def test_crypto_funding_skip():
    s = crypto_24h.compute_position_size(10000, 10000, 100.0, 1.0, funding_rate=-0.002)
    assert s.units == 0.0
    assert s.reason.startswith("funding_too_negative")


def test_crypto_funding_haircut():
    s = crypto_24h.compute_position_size(1000, 1000, 100.0, 5.0, funding_rate=-0.0005)
    # risk halved: 0.0075 * 1000 = 7.5 / 10 = 0.75 units
    assert s.units == pytest.approx(0.75, abs=1e-6)


def test_crypto_no_edge():
    s = crypto_24h.compute_position_size(1000, 1000, 100.0, 5.0, win_rate=0.2)
    assert s.units == 0.0
    assert s.reason == "no_edge_zero_risk"


def test_crypto_below_min():
    s = crypto_24h.compute_position_size(1000, 40.0, 100.0, 5.0)
    assert s.units == 0.0
    assert s.reason.startswith("below_min_position")


# ===========================================================================
# RiskManager - NAV / drawdown / halt
# ===========================================================================

def test_monthly_drawdown_triggers_halt():
    now = datetime.now(UTC)
    month_start = datetime(now.year, now.month, 1, tzinfo=UTC)
    _add_closed_trade("AAA", AssetType.STOCK, +500.0, month_start - timedelta(days=1))
    _add_closed_trade("BBB", AssetType.STOCK, -2000.0, now)
    rm = RiskManager(starting_capital=10000)
    # baseline = 10500, current = 8500 -> -19.05%
    assert rm.month_start_nav(now) == pytest.approx(10500.0)
    assert rm.current_nav() == pytest.approx(8500.0)
    assert rm.monthly_drawdown(now) == pytest.approx(-2000 / 10500, abs=1e-4)
    assert rm.is_halted(now) is True


def test_monthly_drawdown_within_limit():
    now = datetime.now(UTC)
    month_start = datetime(now.year, now.month, 1, tzinfo=UTC)
    _add_closed_trade("AAA", AssetType.STOCK, +500.0, month_start - timedelta(days=1))
    _add_closed_trade("BBB", AssetType.STOCK, -1000.0, now)
    rm = RiskManager(starting_capital=10000)
    assert rm.is_halted(now) is False


def test_portfolio_heat():
    _add_position("AAA", AssetType.STOCK, entry=100.0, stop=95.0, qty=10)   # risk 50
    _add_position("BBB", AssetType.STOCK, entry=50.0, stop=47.5, qty=20)    # risk 50
    rm = RiskManager(starting_capital=10000)
    assert rm.portfolio_heat(AssetType.STOCK, equity=10000) == pytest.approx(0.01)


def test_pre_trade_gate_blocks_on_halt():
    now = datetime.now(UTC)
    month_start = datetime(now.year, now.month, 1, tzinfo=UTC)
    _add_closed_trade("AAA", AssetType.STOCK, +500.0, month_start - timedelta(days=1))
    _add_closed_trade("BBB", AssetType.STOCK, -2000.0, now)
    rm = RiskManager(starting_capital=10000)
    ok, reason = rm.pre_trade_ok(stock_params(), "MSFT", now=now)
    assert ok is False and reason == "drawdown_halt"


def test_pre_trade_gate_blocks_duplicate():
    _add_position("MSFT", AssetType.STOCK, entry=100.0, stop=95.0, qty=5)
    rm = RiskManager(starting_capital=10000)
    ok, reason = rm.pre_trade_ok(stock_params(), "MSFT")
    assert ok is False and reason == "already_open"


# ===========================================================================
# Sector / correlation guard
# ===========================================================================

def test_group_exposure_guard():
    smap = {"AAA": "tech", "BBB": "tech", "CCC": "energy"}
    ok, _ = check_group_exposure(["AAA"], "BBB", smap, max_per_group=2)
    assert ok is True
    blocked, reason = check_group_exposure(["AAA", "BBB"], "CCC", {"AAA": "tech", "BBB": "tech", "CCC": "tech"}, 2)
    assert blocked is False and reason.startswith("group_full")


def test_group_exposure_noop_without_map():
    ok, reason = check_group_exposure(["AAA"], "BBB", None, 2)
    assert ok is True and reason == "no_group_data"


# ===========================================================================
# StockExecutor (paper) on real schema
# ===========================================================================

def test_stock_executor_open_and_close():
    ex = StockExecutor(paper=True, paper_cash=10000, max_positions=8)
    res = ex.open_long("AAPL", units=10, entry_price=100.0, stop_loss=95.0, take_profit=110.0, week_number=22)
    assert res["status"] == "filled"
    assert ex.count_open_stock_positions() == 1

    close = ex.close_long("AAPL", exit_price=110.0, reason="weekly_exit")
    assert close["status"] == "filled"
    assert ex.count_open_stock_positions() == 0
    # bought ~100.05 (slippage), sold ~109.945 -> positive pnl
    assert close["pnl"] > 0


def test_stock_executor_backstops():
    ex = StockExecutor(paper=True, paper_cash=10000, max_positions=1)
    assert ex.open_long("AAPL", 10, 100.0, 95.0, 110.0)["status"] == "filled"
    # duplicate
    assert ex.open_long("AAPL", 10, 100.0, 95.0, 110.0)["reason"] == "already_open"
    # max positions
    assert ex.open_long("MSFT", 10, 100.0, 95.0, 110.0)["reason"] == "max_positions"
    # below min notional
    ex2 = StockExecutor(paper=True, paper_cash=10000, max_positions=8)
    assert ex2.open_long("NVDA", 0.1, 100.0, 95.0, 110.0)["reason"] == "below_min_notional"


# ===========================================================================
# stock_weekly pipeline
# ===========================================================================

def _scored_and_universe():
    scored = pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "composite": [0.80, 0.70],
        "trend_score": [0.9, 0.8],
        "rsi_score": [1.0, 1.0],
        "momentum_score": [0.6, 0.5],
        "volume_score": [0.6, 0.6],
        "volatility_score": [0.5, 0.5],
    })
    universe = {
        "AAA": _stock_feature_df(close=100.0, rsi=55.0),
        "BBB": _stock_feature_df(close=50.0, rsi=50.0),
    }
    return scored, universe


def test_stock_weekly_entries_sized_and_gated():
    scored, universe = _scored_and_universe()
    rm = RiskManager(starting_capital=10000)
    plans = stock_weekly.generate_week_entries(
        scored_df=scored, universe=universe,
        equity=10000, available_cash=10000, risk_manager=rm,
    )
    assert len(plans) == 2
    for p in plans:
        assert p.units > 0
        assert p.notional >= 50.0
        assert p.stop_price == pytest.approx(p.entry_price * 0.95)


def test_stock_weekly_buys_execute_and_persist():
    scored, universe = _scored_and_universe()
    rm = RiskManager(starting_capital=10000)
    ex = StockExecutor(paper=True, paper_cash=10000, max_positions=8)
    result = stock_weekly.run_stock_weekly_buys(
        scored_df=scored, universe=universe,
        equity=10000, available_cash=10000, executor=ex, risk_manager=rm,
    )
    assert len(result.entries) == 2
    assert all(r["status"] == "filled" for r in result.executed)
    assert result.persisted == 2
    assert ex.count_open_stock_positions() == 2


def test_stock_weekly_sells_close_all():
    ex = StockExecutor(paper=True, paper_cash=10000, max_positions=8)
    ex.open_long("AAA", 10, 100.0, 95.0, 110.0, week_number=22)
    open_positions = [{"ticker": "AAA", "entry_price": 100.0}]
    universe = {"AAA": _stock_feature_df(close=105.0)}
    result = stock_weekly.run_stock_weekly_sells(open_positions, universe, executor=ex)
    assert len(result.exits) == 1
    assert result.executed[0]["status"] == "filled"
    assert ex.count_open_stock_positions() == 0


def test_stock_weekly_halt_suppresses_entries():
    now = datetime.now(UTC)
    month_start = datetime(now.year, now.month, 1, tzinfo=UTC)
    _add_closed_trade("X", AssetType.STOCK, +500.0, month_start - timedelta(days=1))
    _add_closed_trade("Y", AssetType.STOCK, -2000.0, now)
    rm = RiskManager(starting_capital=10000)
    scored, universe = _scored_and_universe()
    plans = stock_weekly.generate_week_entries(
        scored_df=scored, universe=universe, equity=10000, available_cash=10000, risk_manager=rm,
    )
    assert plans == []


# ===========================================================================
# crypto pipeline honors the shared halt
# ===========================================================================

def test_crypto_pipeline_halt_suppresses_entries():
    now = datetime.now(UTC)
    month_start = datetime(now.year, now.month, 1, tzinfo=UTC)
    _add_closed_trade("X", AssetType.CRYPTO, +500.0, month_start - timedelta(days=1))
    _add_closed_trade("Y", AssetType.CRYPTO, -2000.0, now)
    rm = RiskManager(starting_capital=10000)
    universe = {"BTC/USDT": _stock_feature_df(close=60000.0)}
    result = crypto_24h.run_crypto_24h_pipeline(
        universe=universe, equity=10000, available_cash=10000,
        risk_manager=rm, persist=False,
    )
    assert result.entries == []


def test_persist_signals_writes_real_schema():
    from strategies.signals import SignalEvent, SignalType, AssetClass, persist_signals
    ev = SignalEvent(
        ticker="AAA", signal_type=SignalType.BUY, asset_class=AssetClass.STOCK,
        composite_score=0.8, rsi=55.0, momentum=0.05, volume_ratio=1.2,
        trend_score=0.9, close_price=100.0, atr=2.0,
    )
    n = persist_signals([ev])
    assert n == 1
    with dbmod.get_db() as db:
        row = db.query(Signal).filter_by(symbol="AAA").first()
        assert row is not None
        assert row.direction == "long"
        assert row.asset_type == "stock"
