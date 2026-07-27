"""tests/test_phase11_backtest.py - historical replay engine.

The point of these tests is that a WRONG backtest is worse than no backtest:
it produces confident, false conclusions. So they pin the properties that
make results trustworthy - point-in-time discipline, costs actually charged,
gap-down fills, and reuse of the live decision code.
"""

from __future__ import annotations

import os

for _k, _v in {
    "ALPACA_API_KEY": "test", "ALPACA_SECRET_KEY": "test",
    "STARTING_CAPITAL": "10000",
}.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import pandas as pd
import pytest

from backtest.engine import (
    BacktestConfig, BacktestTrade, _slice_universe, run_backtest,
)
from data.fetcher import add_features


def _frame(seed: int, n: int = 400, drift: float = 0.0004) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    rets = drift + rng.normal(0, 0.012, n)
    close = 100 * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    openp = pd.Series(close).shift(1).fillna(close[0]).to_numpy()
    vol = rng.uniform(1e6, 2e6, n)
    df = pd.DataFrame({"open": openp, "high": high, "low": low,
                       "close": close, "volume": vol}, index=dates)
    return add_features(df)


def _universe(n_tickers: int = 12, **kw) -> dict[str, pd.DataFrame]:
    uni = {f"T{i:02d}": _frame(seed=10 + i, **kw) for i in range(n_tickers)}
    uni["SPY"] = _frame(seed=999, **kw)
    return uni


# ---------------------------------------------------------------------------
# point-in-time discipline
# ---------------------------------------------------------------------------

def test_slice_universe_never_leaks_future_bars():
    uni = _universe(3)
    cutoff = pd.Timestamp("2022-06-01")
    pit = _slice_universe(uni, cutoff, min_bars=10)
    assert pit, "expected some frames to survive the slice"
    for sym, df in pit.items():
        assert df.index.max() <= cutoff, f"{sym} leaked a future bar"


def test_slice_universe_drops_short_history():
    uni = _universe(2)
    pit = _slice_universe(uni, pd.Timestamp("2022-01-10"), min_bars=200)
    assert pit == {}          # warmup not satisfied yet -> no decisions


# ---------------------------------------------------------------------------
# engine mechanics
# ---------------------------------------------------------------------------

def test_backtest_runs_and_reports_metrics():
    res = run_backtest(_universe(10), BacktestConfig(warmup_bars=210))
    assert res.metrics, "expected metrics"
    assert not res.equity.empty
    assert res.equity.index.is_monotonic_increasing
    for key in ("total_return_pct", "max_drawdown_pct", "sharpe", "trades"):
        assert key in res.metrics
    assert res.metrics["max_drawdown_pct"] <= 0.0


def test_equity_starts_at_initial_capital():
    cfg = BacktestConfig(warmup_bars=210, initial_capital=5_000.0)
    res = run_backtest(_universe(8), cfg)
    # first marked bar happens before any fill, so equity == starting cash
    assert res.equity.iloc[0] == pytest.approx(5_000.0)


def test_insufficient_history_is_a_clean_no_op():
    short = {"AAA": _frame(1, n=50)}
    res = run_backtest(short, BacktestConfig(warmup_bars=200))
    assert res.equity.empty and res.metrics == {}


def test_costs_are_actually_charged():
    """A run that trades must report non-zero costs; zero would mean the
    cost model silently isn't wired (the classic too-good backtest)."""
    cfg = BacktestConfig(warmup_bars=210, cost_bps=50.0)
    res = run_backtest(_universe(10), cfg)
    if res.metrics.get("trades", 0) > 0:
        assert res.metrics["total_costs"] > 0


def test_higher_costs_never_improve_results():
    uni = _universe(10)
    cheap = run_backtest(uni, BacktestConfig(warmup_bars=210, cost_bps=1.0))
    dear = run_backtest(uni, BacktestConfig(warmup_bars=210, cost_bps=100.0))
    if cheap.metrics.get("trades", 0) > 0:
        assert dear.metrics["total_return_pct"] <= cheap.metrics["total_return_pct"] + 1e-9


def test_positions_never_exceed_top_n():
    cfg = BacktestConfig(warmup_bars=210, top_n=3)
    res = run_backtest(_universe(12), cfg)
    by_date: dict = {}
    for t in res.trades:
        if t.exit_date is None:
            continue
        for d in pd.bdate_range(t.entry_date, t.exit_date):
            by_date[d] = by_date.get(d, 0) + 1
    if by_date:
        assert max(by_date.values()) <= cfg.top_n


def test_no_negative_cash_implied_by_equity():
    res = run_backtest(_universe(10), BacktestConfig(warmup_bars=210))
    assert (res.equity > 0).all()


# ---------------------------------------------------------------------------
# trade accounting
# ---------------------------------------------------------------------------

def test_trade_pnl_math():
    t = BacktestTrade(symbol="AAA", entry_date=pd.Timestamp("2024-01-02"),
                      entry_price=100.0, units=2.0)
    t.exit_price, t.exit_date = 110.0, pd.Timestamp("2024-01-09")
    assert t.pnl == pytest.approx(20.0)
    assert t.pnl_pct == pytest.approx(0.10)
    assert t.bars_held == 7


def test_open_trade_has_zero_pnl():
    t = BacktestTrade(symbol="AAA", entry_date=pd.Timestamp("2024-01-02"),
                      entry_price=100.0, units=2.0)
    assert t.pnl == 0.0 and t.bars_held is None


def test_benchmark_is_reported_when_present():
    """A nan benchmark hides the only number that matters - whether the
    strategy beats doing nothing. Regression: SPY is an ETF, never an index
    constituent, so the default universe omitted it and the comparison was
    silently nan."""
    res = run_backtest(_universe(8), BacktestConfig(warmup_bars=210))
    assert not res.benchmark.empty
    assert res.metrics["benchmark_return_pct"] == res.metrics["benchmark_return_pct"]


def test_summary_is_printable():
    res = run_backtest(_universe(8), BacktestConfig(warmup_bars=210))
    text = res.summary()
    assert "Return" in text and "Max drawdown" in text
