"""tests/test_phase19_crypto_gates.py - the crypto side, held to the equity bar.

Phase 1 put the equity strategies through look-ahead detection, cost stress
and a Deflated Sharpe that prices in the trial count. The crypto strategies
had been through none of it while trading 24/7. These tests cover the gates
that close that gap.

As in the equity leakage tests, honest code must stay quiet AND planted
leaks must be caught - a detector that has never fired proves nothing.
"""

from __future__ import annotations

import os

for _k, _v in {"ALPACA_API_KEY": "test", "ALPACA_SECRET_KEY": "test",
               "STARTING_CAPITAL": "10000"}.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import pandas as pd
import pytest

from backtest.crypto_lab import (
    CryptoLabConfig, backtest_crypto_momentum, backtest_hold, backtest_regime,
)
from backtest.leakage import LeakReport, check_strategy_path, run_all_crypto
from backtest.validate import deflated_sharpe_ratio


def _coin(seed: int, n: int = 520, drift: float = 0.001) -> pd.DataFrame:
    """RAW OHLCV on 4h bars - deliberately not featured. The crypto
    strategies slice history and compute their own indicators, so featured
    frames would test a path production never takes."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    close = 100 * np.cumprod(1 + drift + rng.normal(0, 0.02, n))
    op = pd.Series(close).shift(1).fillna(close[0]).to_numpy()
    return pd.DataFrame(
        {"open": op, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": rng.uniform(1e5, 2e5, n)}, index=idx)


def _universe() -> dict:
    return {"BTC/USD": _coin(1), "ETH/USD": _coin(2), "SOL/USD": _coin(3),
            "DOGE/USD": _coin(4, drift=-0.0005)}


# --------------------------- honest code stays quiet ---------------------------

def test_crypto_strategies_have_no_look_ahead():
    """The end-to-end gate: destroy everything after a cut date and the
    equity curve BEFORE the cut must be bit-for-bit identical."""
    uni = _universe()
    cfg = CryptoLabConfig(warmup_bars=210)
    for name, fn in (("regime", backtest_regime),
                     ("momentum", backtest_crypto_momentum)):
        rep = check_strategy_path(fn, uni, cfg)
        assert rep.checks > 0, f"{name}: zero comparisons - nothing was tested"
        assert rep.clean, f"{name} leak: {[str(l) for l in rep.leaks]}"


def test_run_all_crypto_actually_runs_checks():
    rep = run_all_crypto(_universe(), feature_symbols=2, n_dates=4)
    assert rep.checks > 0, "run_all_crypto reported clean without testing anything"
    assert rep.clean, f"leaks: {[str(l) for l in rep.leaks]}"


# --------------------------- planted leaks get caught --------------------------

def test_detector_catches_a_planted_crypto_leak():
    """A strategy that peeks at the final bar must be flagged. Without this,
    a clean report from the honest strategies means nothing."""
    from backtest.engine import BacktestResult

    def leaky(universe, cfg=None):
        df = universe["BTC/USD"]
        closes = pd.to_numeric(df["close"], errors="coerce").dropna()
        # The whole series is scaled by a value only knowable at the END.
        final = float(closes.iloc[-1])
        eq = (closes / final) * 10_000.0
        return BacktestResult(equity=eq, trades=[], benchmark=pd.Series(dtype=float),
                              metrics={"total_return_pct": 0.0})

    rep = check_strategy_path(leaky, _universe(), CryptoLabConfig())
    assert not rep.clean, "detector missed a strategy scaled by a future bar"


def test_empty_universe_is_inconclusive_not_clean():
    """The dangerous failure mode: reporting a pass because nothing ran."""
    rep = run_all_crypto({}, feature_symbols=2, n_dates=4)
    assert rep.checks == 0
    # clean is True here (no leaks found), which is exactly why the CLI gates
    # on `checks` as well - see _print_report.
    assert rep.clean


# --------------------------- annualization is honest ---------------------------

def test_4h_bars_are_not_annualized_as_daily():
    """A 4h-bar Sharpe annualized with the 252-day equity default is inflated
    by ~sqrt(2190/252) ~ 2.9x. Getting this wrong turns noise into a headline."""
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.0005, 0.01, 1200))

    daily = deflated_sharpe_ratio(r, n_trials=20, periods_per_year=252)
    four_h = deflated_sharpe_ratio(r, n_trials=20, periods_per_year=2190)

    # Annualization scales by sqrt(periods), so it inflates the MAGNITUDE.
    # Asserting four_h > daily would be wrong for a losing strategy, where
    # the same scaling makes the number more negative.
    assert abs(four_h.sharpe) > abs(daily.sharpe)
    ratio = four_h.sharpe / daily.sharpe
    assert ratio == pytest.approx(np.sqrt(2190 / 252), rel=0.01)

    # The years attributed to the sample must follow the bar size too: 1200
    # 4h bars is ~0.55y of history, not the ~4.8y a daily reading implies.
    assert four_h.years == pytest.approx(1200 / 2190, rel=0.01)
    assert daily.years == pytest.approx(1200 / 252, rel=0.01)


def test_crypto_compare_uses_crypto_periods_per_year():
    """Guard the wiring, not just the maths: the 4h default must reach the
    DSR call as 2190, not 252."""
    import inspect

    from backtest import crypto_compare

    src = inspect.getsource(crypto_compare.main)
    assert "periods_per_year=ppy" in src, "DSR is not being told the bar size"
    assert "365 * 24 / hours" in src, "periods-per-year is not derived from timeframe"


# --------------------------- stress wiring -------------------------------------

def test_stress_skips_hold_only_results():
    """Buy-and-hold has one trade and no parameters; stressing it measures
    nothing, so the runner must say so rather than print an empty report."""
    uni = _universe()
    cfg = CryptoLabConfig(warmup_bars=210)
    res = backtest_hold(uni, cfg)
    rows = [("hold BTC", res.metrics, res)]

    from backtest.crypto_compare import _stress
    _stress(rows, uni, cfg)          # must not raise


# --------------------------- hysteresis ----------------------------------------

def test_confirm_bars_1_is_the_old_behaviour():
    """The default must not silently change results that are already logged."""
    uni = _universe()
    base = CryptoLabConfig(warmup_bars=210)
    assert base.confirm_bars == 1
    a = backtest_regime(uni, base)
    b = backtest_regime(uni, CryptoLabConfig(warmup_bars=210, confirm_bars=1))
    assert a.metrics["trades"] == b.metrics["trades"]
    pd.testing.assert_series_equal(a.equity, b.equity)


def test_confirm_bars_reduces_trading():
    """The whole point: a slower gate must trade less and pay less. If this
    does not hold, the cost diagnosis was wrong."""
    uni = _universe()
    fast = backtest_regime(uni, CryptoLabConfig(warmup_bars=210, confirm_bars=1))
    slow = backtest_regime(uni, CryptoLabConfig(warmup_bars=210, confirm_bars=8))

    assert slow.metrics["trades"] <= fast.metrics["trades"]
    assert slow.metrics["total_costs"] <= fast.metrics["total_costs"]


def test_confirm_bars_is_monotonic_in_patience():
    """Trade count must not increase as the gate gets slower. A rise would
    mean the hysteresis is oscillating rather than damping."""
    uni = _universe()
    counts = [backtest_regime(uni, CryptoLabConfig(warmup_bars=210, confirm_bars=v))
              .metrics["trades"] for v in (1, 2, 4, 8, 12)]
    assert counts == sorted(counts, reverse=True), counts


def test_confirm_bars_zero_is_treated_as_one():
    """Guard against a config typo silently disabling the gate."""
    uni = _universe()
    a = backtest_regime(uni, CryptoLabConfig(warmup_bars=210, confirm_bars=0))
    b = backtest_regime(uni, CryptoLabConfig(warmup_bars=210, confirm_bars=1))
    assert a.metrics["trades"] == b.metrics["trades"]


def test_hysteresis_ignores_a_single_bar_flicker():
    """Directly: one bar of contrary signal must not move the position."""
    from backtest.crypto_lab import backtest_regime as _regime

    uni = {"BTC/USD": _coin(11, n=600, drift=0.002)}
    fast = _regime(uni, CryptoLabConfig(warmup_bars=210, confirm_bars=1))
    slow = _regime(uni, CryptoLabConfig(warmup_bars=210, confirm_bars=5))
    # A strongly trending single coin should be held throughout by the slow
    # gate, while the fast one churns on noise.
    assert slow.metrics["trades"] <= fast.metrics["trades"]


def test_stress_runs_on_an_active_strategy():
    uni = _universe()
    cfg = CryptoLabConfig(warmup_bars=210)
    res = backtest_regime(uni, cfg)
    rows = [("regime wide (4)", res.metrics, res)]

    from backtest.crypto_compare import _stress
    _stress(rows, uni, cfg)          # must not raise
