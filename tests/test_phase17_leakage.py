"""tests/test_phase17_leakage.py - the look-ahead detector.

A leak detector that has never caught a leak is not evidence of anything.
Half of these tests deliberately plant look-ahead and require the detector
to find it; the rest require it to stay quiet on honest code.
"""

from __future__ import annotations

import os

for _k, _v in {"ALPACA_API_KEY": "test", "ALPACA_SECRET_KEY": "test",
               "STARTING_CAPITAL": "10000"}.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import pandas as pd
import pytest

from backtest.leakage import (
    LeakReport, check_features, check_scores, check_strategy_path,
    perturb_after, warmup_sensitivity,
)
from data.fetcher import add_features


def _ohlcv(seed: int = 0, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n)
    close = 100 * np.cumprod(1 + 0.0004 + rng.normal(0, 0.012, n))
    op = pd.Series(close).shift(1).fillna(close[0]).to_numpy()
    return pd.DataFrame({"open": op, "high": close * 1.004, "low": close * 0.996,
                         "close": close, "volume": rng.uniform(1e6, 2e6, n)},
                        index=dates)


def _universe(n: int = 8, bars: int = 600) -> dict:
    uni = {f"T{i:02d}": add_features(_ohlcv(10 + i, bars)) for i in range(n)}
    uni["SPY"] = add_features(_ohlcv(999, bars))
    return uni


# --------------------------- honest code stays quiet ---------------------------

def test_real_indicators_have_no_look_ahead():
    """add_features must be causal: truncating the future cannot change a
    value the indicator already produced."""
    rep = check_features(_ohlcv(1), "T", n_dates=5, warmup=250)
    assert rep.checks > 0, "check ran zero comparisons - it is not testing anything"
    assert rep.clean, f"leak in production indicators: {[str(l) for l in rep.leaks]}"


def test_the_composite_ranking_has_no_look_ahead():
    """The score the scheduler actually ranks on, end to end through
    add_features -> score_universe."""
    raw = {f"T{i:02d}": _ohlcv(30 + i) for i in range(4)}
    rep = check_scores(raw, n_dates=3, warmup=250)
    assert rep.checks > 0, "check_scores ran zero comparisons"
    assert rep.clean, f"composite score leaked: {[str(l) for l in rep.leaks][:3]}"


def test_momentum_backtest_ignores_the_future():
    from backtest.lab import LabConfig, backtest_momentum
    cfg = LabConfig(top_n=4, point_in_time_membership=False)
    rep = check_strategy_path(backtest_momentum, _universe(8), cfg)
    assert rep.checks > 0
    assert rep.clean, f"momentum leaked: {[str(l) for l in rep.leaks][:3]}"


def test_trend_filter_backtest_ignores_the_future():
    from backtest.lab import LabConfig, backtest_trend_filter
    cfg = LabConfig(point_in_time_membership=False)
    rep = check_strategy_path(backtest_trend_filter, _universe(4), cfg)
    assert rep.checks > 0
    assert rep.clean, f"trend filter leaked: {[str(l) for l in rep.leaks][:3]}"


def test_weekly_rotation_engine_ignores_the_future():
    from backtest.engine import BacktestConfig, run_backtest
    cfg = BacktestConfig(warmup_bars=210, point_in_time_membership=False)
    rep = check_strategy_path(run_backtest, _universe(8), cfg)
    assert rep.checks > 0
    assert rep.clean, f"rotation engine leaked: {[str(l) for l in rep.leaks][:3]}"


# --------------------------- planted leaks must be caught ---------------------

def test_detector_catches_a_centered_rolling_window(monkeypatch):
    """`center=True` is the classic accident: the average at bar T is built
    from bars on both sides of T."""
    import data.fetcher as fetcher
    real = fetcher.add_features

    def leaky(df):
        out = real(df)
        out["sma_50"] = df["close"].rolling(50, center=True).mean()
        return out

    monkeypatch.setattr(fetcher, "add_features", leaky)
    rep = check_features(_ohlcv(2), "T", n_dates=5, warmup=250)
    assert not rep.clean
    assert any("sma_50" in l.subject for l in rep.leaks)


def test_detector_catches_full_history_normalization(monkeypatch):
    """Scaling by the series max leaks the eventual peak into every bar -
    the mistake that makes an ML backtest look clairvoyant."""
    import data.fetcher as fetcher
    real = fetcher.add_features

    def leaky(df):
        out = real(df)
        out["rsi"] = out["rsi"] / max(float(out["rsi"].max()), 1e-9)
        return out

    monkeypatch.setattr(fetcher, "add_features", leaky)
    rep = check_features(_ohlcv(3), "T", n_dates=5, warmup=250)
    assert not rep.clean
    assert any("rsi" in l.subject for l in rep.leaks)


def test_detector_catches_a_negative_shift(monkeypatch):
    """shift(-3) writes a LATER bar into an earlier row - the same mechanism
    as a backfilled gap, in its most blatant form."""
    import data.fetcher as fetcher
    real = fetcher.add_features

    def leaky(df):
        out = real(df)
        out["momentum"] = df["close"].shift(-3)      # tomorrow's price, plainly
        return out

    monkeypatch.setattr(fetcher, "add_features", leaky)
    rep = check_features(_ohlcv(4), "T", n_dates=5, warmup=250)
    assert not rep.clean


def test_detector_catches_a_strategy_that_peeks():
    """A strategy that ranks on the NEXT bar's return is pure look-ahead;
    the perturbation test must notice even though every indicator is clean."""
    from backtest.engine import BacktestResult

    def cheating(universe, cfg=None):
        # "buy whatever goes up tomorrow" - equity built from future bars
        spy = universe["SPY"]["close"]
        future = spy.shift(-5).ffill()
        return BacktestResult(equity=future / future.iloc[0] * 10_000.0,
                              trades=[], benchmark=spy, metrics={"trades": 1})

    rep = check_strategy_path(cheating, _universe(4))
    assert not rep.clean, "a strategy reading 5 bars ahead went undetected"


def test_empty_strategy_result_is_a_note_not_a_crash():
    """The first real-data run crashed here: an empty BacktestResult carries
    a RangeIndex, and slicing it against a Timestamp raises. Nothing-to-test
    must surface as a note - never a traceback, never a silent pass."""
    from backtest.engine import BacktestResult

    def produces_nothing(universe, cfg=None):
        return BacktestResult()

    rep = check_strategy_path(produces_nothing, _universe(3))
    assert rep.checks == 0
    assert rep.clean                       # no leaks - but also no evidence
    assert any("NOTHING was tested" in n for n in rep.notes)


def test_trend_filter_without_benchmark_is_a_note_not_a_crash():
    """The exact real-world trigger: the first 12 S&P tickers are A..ADSK,
    no SPY, so trend_filter has no benchmark and returns an empty result."""
    from backtest.lab import LabConfig, backtest_trend_filter
    uni = {f"T{i:02d}": add_features(_ohlcv(10 + i)) for i in range(4)}  # no SPY
    cfg = LabConfig(point_in_time_membership=False)
    rep = check_strategy_path(backtest_trend_filter, uni, cfg)
    assert rep.checks == 0 and rep.notes


# --------------------------- the perturbation itself ---------------------------

def test_perturbation_only_touches_bars_after_the_cut():
    uni = _universe(3)
    cut = uni["SPY"].index[300]
    mangled = perturb_after(uni, cut)
    for sym, df in mangled.items():
        before = df.loc[df.index <= cut, "close"]
        assert before.equals(uni[sym].loc[uni[sym].index <= cut, "close"]), \
            f"{sym}: perturbation corrupted the past"
        after = df.loc[df.index > cut, "close"]
        assert not np.allclose(after.to_numpy(),
                               uni[sym].loc[uni[sym].index > cut, "close"].to_numpy()), \
            f"{sym}: perturbation left the future unchanged - the test is inert"


def test_perturbation_reaches_derived_indicators_not_just_prices():
    """Mangling only OHLCV would leave sma_50, rsi, macd intact after the
    cut, so a strategy that peeked at a future INDICATOR rather than a
    future price would go undetected."""
    uni = _universe(2)
    cut = uni["SPY"].index[300]
    mangled = perturb_after(uni, cut)
    df, orig = mangled["T00"], uni["T00"]
    future = df.index > cut
    for col in ("sma_50", "rsi", "macd", "atr", "above_sma50"):
        assert col in df.columns, f"{col} missing - test universe changed"
        a = df.loc[future, col].to_numpy(dtype="float64", na_value=0.0)
        b = orig.loc[future, col].to_numpy(dtype="float64", na_value=0.0)
        assert not np.allclose(a, b), f"{col} survived the perturbation untouched"


def test_perturbation_leaves_pre_cut_indicators_bit_identical():
    """The other half of the contract: if the past moved at all, every leak
    this reports would be an artifact of the test itself."""
    uni = _universe(2)
    cut = uni["SPY"].index[300]
    mangled = perturb_after(uni, cut)
    df, orig = mangled["T00"], uni["T00"]
    past = df.index <= cut
    for col in ("sma_50", "rsi", "macd", "atr", "close", "volume"):
        a = df.loc[past, col].to_numpy(dtype="float64", na_value=0.0)
        b = orig.loc[past, col].to_numpy(dtype="float64", na_value=0.0)
        assert np.array_equal(a, b), f"{col}: perturbation corrupted the past"


# --------------------------- warm-up (not leakage) ---------------------------

def test_recursive_indicators_converge_with_enough_warmup():
    """EMA and Wilder RSI never fully forget their seed, so a small drift is
    correct behaviour. It must still be small enough that the backtest and
    the live bot agree on the same bar."""
    drift = warmup_sensitivity(_ohlcv(5, n=900))
    assert drift, "expected some columns to be measured"
    for col in ("sma_50", "sma_200"):
        if col in drift:
            assert drift[col] < 1e-9, f"{col} is a plain window; it must match exactly"
    assert max(drift.values()) < 0.05, "warm-up drift is large enough to matter"


def test_warmup_check_is_clean_on_short_history():
    assert warmup_sensitivity(_ohlcv(6, n=100)) == {}


# --------------------------- report plumbing ---------------------------

def test_report_merges_and_summarizes():
    a, b = LeakReport(checks=3), LeakReport(checks=4)
    a.extend(b)
    assert a.checks == 7 and a.clean
    assert "ZERO leaks" in a.summary()
