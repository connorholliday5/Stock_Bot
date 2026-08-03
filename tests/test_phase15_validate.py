"""tests/test_phase15_validate.py - statistical validation tools."""

from __future__ import annotations

import os

for _k, _v in {"ALPACA_API_KEY": "test", "ALPACA_SECRET_KEY": "test",
               "STARTING_CAPITAL": "10000"}.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import pandas as pd
import pytest

from backtest.validate import (
    bootstrap_drawdown, deflated_sharpe_ratio, expected_max_sharpe,
    permutation_test,
)


def test_noise_floor_rises_with_trial_count():
    """Testing more variants raises the Sharpe that pure luck produces."""
    f5 = expected_max_sharpe(10, years=5)
    f20 = expected_max_sharpe(20, years=5)
    f100 = expected_max_sharpe(100, years=5)
    assert f5 < f20 < f100
    # published reference points (Bailey & Lopez de Prado), 5y of data
    assert 0.6 < f5 < 0.8
    assert 0.75 < f20 < 0.95


def test_noise_floor_falls_with_longer_history():
    assert expected_max_sharpe(20, years=20) < expected_max_sharpe(20, years=2)


def test_single_trial_has_no_selection_bias():
    assert expected_max_sharpe(1, years=5) == 0.0


def test_our_momentum_result_is_inside_the_noise_band():
    """The finding that decided the plan: a 5y Sharpe of 0.81 selected from
    ~15 variants is what a zero-edge strategy produces by luck."""
    floor = expected_max_sharpe(15, years=5)
    assert floor > 0.75, "noise floor for 15 trials on 5y should exceed 0.75"
    assert 0.81 < floor + 0.15, "0.81 must not clear the floor by much"


def test_strong_edge_survives_deflation():
    rng = np.random.default_rng(0)
    # daily returns with a genuinely large Sharpe (~2 annualized)
    r = pd.Series(rng.normal(0.002 / 252 * 252 / 252, 0.01, 1260) + 0.0012)
    res = deflated_sharpe_ratio(r, n_trials=5)
    assert res.sharpe > 1.0
    assert res.deflated_sharpe > 0.9


def test_noise_returns_fail_deflation():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0, 0.01, 1260))       # no edge at all
    res = deflated_sharpe_ratio(r, n_trials=20)
    assert res.deflated_sharpe < 0.8
    assert "NOISE" in res.verdict.upper() or "borderline" in res.verdict


def test_more_trials_lowers_confidence_in_the_same_returns():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0004, 0.01, 1260))
    few = deflated_sharpe_ratio(r, n_trials=2)
    many = deflated_sharpe_ratio(r, n_trials=200)
    assert many.deflated_sharpe < few.deflated_sharpe


def test_deflation_handles_degenerate_input():
    res = deflated_sharpe_ratio(pd.Series([0.0] * 5), n_trials=10)
    assert res.verdict == "insufficient data"


def test_permutation_test_flags_a_drift_harvesting_rule():
    """A rule that just holds a rising asset should NOT beat shuffled data:
    shuffling preserves drift, so buy-and-hold scores the same."""
    rng = np.random.default_rng(3)
    close = 100 * np.cumprod(1 + 0.0005 + rng.normal(0, 0.01, 400))
    uni = {"A": pd.DataFrame({"close": close, "open": close,
                              "high": close, "low": close})}

    def buy_and_hold(u) -> float:
        c = u["A"]["close"]
        return float(c.iloc[-1] / c.iloc[0] - 1.0)

    out = permutation_test(buy_and_hold, uni, n_permutations=60)
    assert out["n"] > 0
    assert out["p_value"] > 0.05          # indistinguishable, correctly


def test_bootstrap_reports_worse_tail_than_the_single_observed_path():
    rng = np.random.default_rng(4)
    eq = pd.Series(10_000 * np.cumprod(1 + rng.normal(0.0004, 0.011, 800)))
    out = bootstrap_drawdown(eq, n_samples=200)
    assert out["p95_max_dd_pct"] <= out["median_max_dd_pct"]
    assert out["worst_max_dd_pct"] <= out["p95_max_dd_pct"]


def test_bootstrap_needs_history():
    assert bootstrap_drawdown(pd.Series([1.0, 2.0])) == {}
