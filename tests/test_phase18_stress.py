"""tests/test_phase18_stress.py - robustness checks.

Each of these encodes a way a good-looking backtest turns out to be nothing:
costs we guessed too low, a P&L that lives in five fills, one exceptional
year, or a parameter peak that is really just where the noise lined up.
"""

from __future__ import annotations

import os

for _k, _v in {"ALPACA_API_KEY": "test", "ALPACA_SECRET_KEY": "test",
               "STARTING_CAPITAL": "10000"}.items():
    os.environ.setdefault(_k, _v)

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestResult
from backtest.stress import (
    Plateau, cost_stress, drop_best_trades, drop_best_year, parameter_plateau,
    stress_strategy,
)


def _equity(seed: int = 0, n: int = 800, drift: float = 0.0005,
            start: str = "2021-01-04") -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(10_000 * np.cumprod(1 + drift + rng.normal(0, 0.01, n)), index=idx)


@dataclass
class FakeTrade:
    pnl: float
    exit_price: float = 1.0


# --------------------------- concentration ---------------------------

def test_removing_the_best_days_lowers_the_return():
    res = BacktestResult(equity=_equity(1), trades=[], benchmark=pd.Series(dtype=float),
                         metrics={"total_return_pct": 0.0})
    out, label = drop_best_trades(res, n=5)
    assert "DAYS" in label, "no trade objects available, so this must say days"
    base = (res.equity.iloc[-1] / res.equity.iloc[0] - 1) * 100
    assert out["total_return_pct"] < base


def test_a_lottery_ticket_strategy_is_exposed():
    """All the profit in five fills. The headline return looks great and the
    strategy is worthless - removing them must take the whole gain."""
    idx = pd.bdate_range("2021-01-04", periods=300)
    rets = np.full(300, -0.0002)          # bleeding almost every day
    rets[[10, 50, 120, 200, 260]] = 0.35   # five enormous winners
    eq = pd.Series(10_000 * np.cumprod(1 + rets), index=idx)
    res = BacktestResult(equity=eq, trades=[], benchmark=pd.Series(dtype=float),
                         metrics={})
    assert eq.iloc[-1] > eq.iloc[0], "setup should look profitable"
    out, _ = drop_best_trades(res, n=5)
    assert out["total_return_pct"] < 0, "removing the 5 winners must sink it"


def test_trade_objects_are_preferred_over_days_when_available():
    res = BacktestResult(equity=_equity(2), trades=[FakeTrade(500.0), FakeTrade(400.0),
                                                    FakeTrade(-50.0)],
                         benchmark=pd.Series(dtype=float), metrics={})
    out, label = drop_best_trades(res, n=2)
    assert "trades removed" in label and "DAYS" not in label
    assert out["total_return_pct"] < (res.equity.iloc[-1] / res.equity.iloc[0] - 1) * 100


def test_wiping_out_the_account_is_reported_not_crashed():
    eq = pd.Series(10_000 * np.linspace(1.0, 1.05, 100),
                   index=pd.bdate_range("2022-01-03", periods=100))
    res = BacktestResult(equity=eq, trades=[FakeTrade(50_000.0)],
                         benchmark=pd.Series(dtype=float), metrics={})
    out, label = drop_best_trades(res, n=1)
    assert out["total_return_pct"] == -100.0
    assert "wipes out" in label


# --------------------------- regime ---------------------------

def test_removing_the_best_year_lowers_the_return():
    eq = _equity(3, n=1000, start="2020-01-02")
    res = BacktestResult(equity=eq, trades=[], benchmark=pd.Series(dtype=float),
                         metrics={})
    out, label = drop_best_year(res)
    assert out, "multi-year curve must produce a result"
    assert out["total_return_pct"] < (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    assert "removed" in label


def test_single_year_history_is_refused_not_guessed():
    eq = _equity(4, n=200, start="2022-01-03")
    res = BacktestResult(equity=eq, trades=[], benchmark=pd.Series(dtype=float),
                         metrics={})
    out, label = drop_best_year(res)
    assert out == {} and "two calendar years" in label


# --------------------------- cost stress ---------------------------

def _lab_universe(n: int = 12, bars: int = 800) -> dict:
    from data.fetcher import add_features
    uni = {}
    for i in range(n):
        rng = np.random.default_rng(20 + i)
        idx = pd.bdate_range("2021-01-04", periods=bars)
        close = 100 * np.cumprod(1 + 0.0004 + rng.normal(0, 0.012, bars))
        op = pd.Series(close).shift(1).fillna(close[0]).to_numpy()
        uni[f"T{i:02d}"] = add_features(pd.DataFrame(
            {"open": op, "high": close * 1.004, "low": close * 0.996,
             "close": close, "volume": rng.uniform(1e6, 2e6, bars)}, index=idx))
    uni["SPY"] = uni["T00"].copy()
    return uni


def test_higher_costs_never_improve_a_result():
    from backtest.lab import LabConfig, backtest_momentum
    cfg = LabConfig(top_n=4, point_in_time_membership=False)
    curve = cost_stress(backtest_momentum, _lab_universe(), cfg, (1.0, 2.0, 4.0))
    assert len(curve) == 3
    rets = [m["total_return_pct"] for _, m in curve]
    assert rets == sorted(rets, reverse=True), "cost is monotone; this is a wiring check"


def test_cost_stress_reports_the_2x_gate():
    from backtest.lab import LabConfig, backtest_momentum
    cfg = LabConfig(top_n=4, point_in_time_membership=False)
    rep = stress_strategy(backtest_momentum, _lab_universe(), cfg, name="momentum")
    assert rep.baseline
    assert rep.survives_2x_costs() in (True, False)


# --------------------------- parameter plateau ---------------------------

def test_a_smooth_surface_is_called_a_plateau():
    @dataclass
    class Cfg:
        knob: int = 10

    def smooth(_uni, cfg):
        # gently peaked: neighbours retain most of the peak
        score = 1.0 - abs(cfg.knob - 10) * 0.02
        return BacktestResult(equity=pd.Series(dtype=float), trades=[],
                              benchmark=pd.Series(dtype=float),
                              metrics={"sharpe": score})

    p = parameter_plateau(smooth, {}, Cfg(), "knob", span=0.25, points=5)
    assert p.is_plateau
    assert p.scores


def test_a_spike_is_called_out_as_overfit():
    """The classic overfit signature: one parameter value works and its
    immediate neighbours collapse. Shipping that peak is shipping noise."""
    @dataclass
    class Cfg:
        knob: int = 10

    def spiky(_uni, cfg):
        score = 2.0 if cfg.knob == 10 else 0.05
        return BacktestResult(equity=pd.Series(dtype=float), trades=[],
                              benchmark=pd.Series(dtype=float),
                              metrics={"sharpe": score})

    p = parameter_plateau(spiky, {}, Cfg(), "knob", span=0.25, points=5)
    assert not p.is_plateau
    assert p.peak_value == 10
    assert "SPIKE" in p.summary()


def test_shipped_value_is_the_plateau_centre_not_the_peak():
    """The true optimum is as likely to sit either side of the observed
    maximum, so the middle of a good region is the safer bet."""
    @dataclass
    class Cfg:
        knob: int = 20

    # scores rise across the sweep with a lucky single-point spike at the low end
    table = {16: 1.9, 17: 0.5, 18: 1.0, 19: 1.05, 20: 1.1, 21: 1.05, 22: 1.0,
             23: 0.95, 24: 0.9}

    def bumpy(_uni, cfg):
        return BacktestResult(equity=pd.Series(dtype=float), trades=[],
                              benchmark=pd.Series(dtype=float),
                              metrics={"sharpe": table.get(cfg.knob, 0.5)})

    p = parameter_plateau(bumpy, {}, Cfg(), "knob", span=0.25, points=9)
    assert p.peak_value == 16                    # the lucky spike
    assert p.centre_value != p.peak_value        # but not what we ship
    assert not p.is_plateau


def test_a_losing_strategy_is_never_a_plateau():
    """A flat surface of negative Sharpes is smooth, and worthless. Plateau
    must mean 'robustly good', not merely 'robustly consistent'."""
    @dataclass
    class Cfg:
        knob: int = 10

    def losing(_uni, cfg):
        return BacktestResult(equity=pd.Series(dtype=float), trades=[],
                              benchmark=pd.Series(dtype=float),
                              metrics={"sharpe": -0.4})

    assert not parameter_plateau(losing, {}, Cfg(), "knob", points=5).is_plateau


def test_unknown_parameter_is_a_clean_noop():
    @dataclass
    class Cfg:
        knob: int = 10

    p = parameter_plateau(lambda u, c: None, {}, Cfg(), "nonexistent")
    assert p.scores == [] and not p.is_plateau


# --------------------------- end to end ---------------------------

def test_full_report_runs_and_grades_every_gate():
    from backtest.lab import LabConfig, backtest_momentum
    cfg = LabConfig(top_n=4, point_in_time_membership=False)
    rep = stress_strategy(backtest_momentum, _lab_universe(10, 800), cfg,
                          name="momentum", params=("top_n",), points=3)
    assert rep.baseline and rep.costs
    assert rep.no_top_trades_label
    for gate in (rep.survives_2x_costs(), rep.survives_trade_removal(),
                 rep.all_plateaus()):
        assert gate in (True, False, None)


# --------------------------- research log / DSR wiring ---------------------------

def test_trial_count_is_read_from_the_research_log():
    """The deflated Sharpe is only meaningful if N is honest, and N lives in
    a tracked file rather than a constant someone can quietly lower."""
    from backtest.compare import RESEARCH_LOG, count_trials
    n, source = count_trials()
    assert source == RESEARCH_LOG.name
    assert n >= 15, "the log should already record our whole search history"


def test_missing_research_log_does_not_flatter_the_result():
    """Falling back to N=1 would assume no search at all - the single most
    flattering possible assumption, and it would inflate every DSR shown."""
    from pathlib import Path

    from backtest.compare import count_trials
    n, source = count_trials(Path("/nonexistent/RESEARCH_LOG.md"), default=20)
    assert n == 20 and source == "assumed"


def test_our_best_result_does_not_clear_its_own_noise_floor():
    """The finding this whole phase exists because of: Sharpe 0.81, selected
    as the winner of the logged trials, sits BELOW what luck alone produces."""
    from backtest.compare import count_trials
    from backtest.validate import expected_max_sharpe
    n, _ = count_trials()
    assert expected_max_sharpe(n, years=5) > 0.81
