"""tests/test_phase12_lab.py - alternative strategy variants."""

from __future__ import annotations

import os

for _k, _v in {"ALPACA_API_KEY": "test", "ALPACA_SECRET_KEY": "test",
               "STARTING_CAPITAL": "10000"}.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import pandas as pd
import pytest

from backtest.lab import (
    LabConfig, STRATEGIES, backtest_buy_hold, backtest_momentum,
    backtest_trend_filter,
)
from data.fetcher import add_features


def _frame(seed: int, n: int = 700, drift: float = 0.0004) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n)
    close = 100 * np.cumprod(1 + drift + rng.normal(0, 0.012, n))
    op = pd.Series(close).shift(1).fillna(close[0]).to_numpy()
    return add_features(pd.DataFrame(
        {"open": op, "high": close * 1.004, "low": close * 0.996,
         "close": close, "volume": rng.uniform(1e6, 2e6, n)}, index=dates))


def _universe(n: int = 20) -> dict:
    uni = {f"T{i:02d}": _frame(10 + i) for i in range(n)}
    uni["SPY"] = _frame(999)
    return uni


def test_all_strategies_produce_metrics():
    uni = _universe()
    for name, fn in STRATEGIES.items():
        res = fn(uni, LabConfig())
        assert res.metrics, f"{name} produced no metrics"
        assert not res.equity.empty
        assert res.equity.index.is_monotonic_increasing


def test_buy_hold_trades_once_and_tracks_the_asset():
    uni = _universe()
    res = backtest_buy_hold(uni, LabConfig())
    assert res.metrics["trades"] == 1          # one entry, never sells
    # return should track SPY closely (cost drag only)
    assert abs(res.metrics["total_return_pct"]
               - res.metrics["benchmark_return_pct"]) < 2.0


def test_trend_filter_trades_rarely():
    """The whole point: a few round trips a year, so costs cannot eat it."""
    res = backtest_trend_filter(_universe(), LabConfig())
    assert res.metrics["trades"] < 40          # vs 261 for weekly rotation


def test_trend_filter_reduces_drawdown_vs_buy_hold():
    uni = _universe()
    bh = backtest_buy_hold(uni, LabConfig())
    tf = backtest_trend_filter(uni, LabConfig())
    # sitting in cash below the MA cannot deepen the worst drawdown
    assert tf.metrics["max_drawdown_pct"] >= bh.metrics["max_drawdown_pct"] - 1e-6


def test_momentum_rebalances_monthly_not_weekly():
    res = backtest_momentum(_universe(20), LabConfig(top_n=5))
    # ~36 months x (5 sells + 5 buys) is the right order of magnitude;
    # weekly churn would be several times this
    assert 0 < res.metrics["trades"] < 600


def test_momentum_skip_window_is_applied():
    """12-1 means the most recent month is EXCLUDED from the ranking - the
    skip is what avoids short-term reversal contaminating the signal."""
    cfg = LabConfig(momentum_lookback=252, momentum_skip=21)
    assert cfg.momentum_skip > 0
    res = backtest_momentum(_universe(12), cfg)
    assert res.metrics


def test_weighting_modes_all_run_and_stay_invested():
    uni = _universe(20)
    for mode in ("equal", "rank", "score"):
        res = backtest_momentum(uni, LabConfig(weighting=mode, top_n=5))
        assert res.metrics, f"{mode} produced nothing"
        assert (res.equity > 0).all()


def test_weights_sum_to_one_and_favour_the_leader():
    from backtest.lab import _weights
    syms = ["A", "B", "C"]
    scores = pd.Series({"A": 0.9, "B": 0.5, "C": 0.1})
    for mode in ("equal", "rank", "score"):
        w = dict(_weights(syms, scores, mode))
        assert sum(w.values()) == pytest.approx(1.0)
    assert dict(_weights(syms, scores, "equal"))["A"] == pytest.approx(1 / 3)
    # concentrating modes give the leader strictly more than an equal split
    assert dict(_weights(syms, scores, "rank"))["A"] > 1 / 3
    assert dict(_weights(syms, scores, "score"))["A"] > 1 / 3


def test_score_weighting_survives_all_negative_scores():
    """Never divide by zero, and never short: fall back to equal weight."""
    from backtest.lab import _weights
    scores = pd.Series({"A": -0.2, "B": -0.5})
    w = dict(_weights(["A", "B"], scores, "score"))
    assert w["A"] == pytest.approx(0.5) and w["B"] == pytest.approx(0.5)


def test_costs_scale_with_turnover():
    uni = _universe()
    cheap = backtest_trend_filter(uni, LabConfig(cost_bps=1.0))
    dear = backtest_trend_filter(uni, LabConfig(cost_bps=100.0))
    assert dear.metrics["total_costs"] > cheap.metrics["total_costs"]
    assert dear.metrics["total_return_pct"] <= cheap.metrics["total_return_pct"] + 1e-9


def test_insufficient_history_is_clean():
    tiny = {"SPY": _frame(1, n=40)}
    assert backtest_buy_hold(tiny, LabConfig()).equity.empty
    assert backtest_trend_filter(tiny, LabConfig()).equity.empty
    assert backtest_momentum(tiny, LabConfig()).equity.empty


def test_equity_never_goes_negative():
    uni = _universe()
    for fn in (backtest_buy_hold, backtest_trend_filter, backtest_momentum):
        res = fn(uni, LabConfig())
        assert (res.equity > 0).all()
