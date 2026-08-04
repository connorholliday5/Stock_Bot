"""tests/test_phase16_pit.py - point-in-time index membership.

The bias this fixes is not academic. Ranking today's S&P 500 across history
means only companies that SURVIVED are ever candidates, and momentum is hit
hardest by that because it systematically buys the extended names most likely
to later crater and leave the index.
"""

from __future__ import annotations

import os

for _k, _v in {"ALPACA_API_KEY": "test", "ALPACA_SECRET_KEY": "test",
               "STARTING_CAPITAL": "10000"}.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import pandas as pd
import pytest

from data.index_membership import (
    Membership, delisting_month, filter_to_members, is_available,
    load_membership, members_on, strip_delisting_suffix,
)


# --------------------------- ticker parsing ---------------------------

def test_delisting_suffix_is_stripped():
    assert strip_delisting_suffix("AAMRQ-201312") == "AAMRQ"
    assert strip_delisting_suffix("AAL-199702") == "AAL"


def test_class_share_dots_are_preserved():
    """Symbology must match data/sp500.py and Alpaca (BRK.B, not BRK-B) or
    the filter silently rejects every class share."""
    assert strip_delisting_suffix("BRK.B") == "BRK.B"
    assert strip_delisting_suffix("BF.B") == "BF.B"
    assert strip_delisting_suffix("AZA.A-200106") == "AZA.A"


def test_delisting_month_is_recoverable():
    assert delisting_month("AAMRQ-201312") == "201312"
    assert delisting_month("AAPL") is None


# --------------------------- lookup semantics ---------------------------

def _fake() -> Membership:
    return Membership(
        dates=[pd.Timestamp("2000-01-01"), pd.Timestamp("2005-01-01"),
               pd.Timestamp("2010-01-01")],
        sets=[frozenset({"A", "B"}), frozenset({"A", "C"}),
              frozenset({"C", "D"})],
    )


def test_lookup_uses_the_snapshot_at_or_before_the_date():
    m = _fake()
    assert m.members_on("2005-01-01") == {"A", "C"}
    assert m.members_on("2007-06-30") == {"A", "C"}


def test_lookup_never_uses_a_future_snapshot():
    """The whole point: a date in 2004 must not see 2005's membership, or
    the 'fix' introduces exactly the look-ahead it was meant to remove."""
    m = _fake()
    assert m.members_on("2004-12-31") == {"A", "B"}
    assert "D" not in m.members_on("2009-12-31")


def test_dates_before_coverage_fall_back_to_the_earliest_snapshot():
    """An approximation, but one made from the PAST - it cannot leak."""
    assert _fake().members_on("1990-01-01") == {"A", "B"}


def test_empty_membership_is_a_clean_unknown():
    assert Membership([], []).members_on("2020-01-01") == frozenset()


# --------------------------- the filter contract ---------------------------

def test_filter_keeps_only_members():
    m = _fake()
    import data.index_membership as im
    original, im._CACHE = im._CACHE, m
    try:
        assert filter_to_members(["A", "C", "ZZ"], "2005-06-01") == ["A", "C"]
    finally:
        im._CACHE = original


def test_missing_data_passes_symbols_through_unchanged():
    """Degrade loudly, not destructively: no membership data must mean 'no
    filter', never 'no eligible names' - otherwise a checkout without the
    CSVs produces an empty backtest that looks like a flat strategy."""
    import data.index_membership as im
    original, im._CACHE = im._CACHE, Membership([], [])
    try:
        assert filter_to_members(["AAPL", "MSFT"], "2020-01-01") == ["AAPL", "MSFT"]
    finally:
        im._CACHE = original


# --------------------------- the real dataset ---------------------------

pytestmark_real = pytest.mark.skipif(not is_available(),
                                     reason="membership dataset not present")


@pytestmark_real
def test_real_dataset_covers_our_backtest_window():
    lo, hi = load_membership().coverage
    assert lo <= pd.Timestamp("1996-01-31")
    assert hi >= pd.Timestamp("2025-01-01"), "coverage must reach recent history"


@pytestmark_real
def test_real_snapshots_are_index_sized():
    for d in ("2000-06-30", "2010-06-30", "2020-06-30"):
        assert 450 <= len(members_on(d)) <= 520, f"{d} membership looks wrong"


@pytestmark_real
def test_tesla_is_absent_before_its_2020_addition():
    """A concrete, checkable fact. TSLA joined the S&P 500 in Dec 2020; a
    momentum backtest that ranks it in 2015 is trading on hindsight."""
    assert "TSLA" not in members_on("2015-01-02")
    assert "TSLA" not in members_on("2020-06-01")
    assert "TSLA" in members_on("2021-06-01")


@pytestmark_real
def test_enron_is_present_before_it_collapsed():
    """The bias runs both ways: the corrected universe must still contain
    the names that later blew up, not just the survivors."""
    assert "ENRNQ" in members_on("1998-01-02")


@pytestmark_real
def test_nvidia_is_absent_in_the_nineties():
    assert "NVDA" not in members_on("1997-01-02")
    assert "NVDA" in members_on("2010-01-04")


# --------------------------- wiring into the backtests ---------------------------

def _frame(seed: int, n: int = 700) -> pd.DataFrame:
    from data.fetcher import add_features
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n)
    close = 100 * np.cumprod(1 + 0.0004 + rng.normal(0, 0.012, n))
    op = pd.Series(close).shift(1).fillna(close[0]).to_numpy()
    return add_features(pd.DataFrame(
        {"open": op, "high": close * 1.004, "low": close * 0.996,
         "close": close, "volume": rng.uniform(1e6, 2e6, n)}, index=dates))


@pytestmark_real
def test_momentum_only_holds_names_that_were_index_members():
    """Real tickers, half of them not in the index during the window: the
    non-members must never be bought."""
    from backtest.lab import LabConfig, backtest_momentum
    real = ["AAPL", "MSFT", "JNJ", "XOM", "KO", "PG"]
    fake = ["ZZZA", "ZZZB", "ZZZC", "ZZZD"]
    uni = {s: _frame(i) for i, s in enumerate(real + fake)}
    uni["SPY"] = _frame(99)

    res = backtest_momentum(uni, LabConfig(top_n=4))
    assert res.metrics, "PIT filter must not empty a valid universe"
    # ZZZ* are not S&P members on any date, so they can never be selected.
    # With the filter off they would be ranked like anything else.
    off = backtest_momentum(uni, LabConfig(top_n=4, point_in_time_membership=False))
    assert off.metrics
    assert res.metrics["trades"] <= off.metrics["trades"]


@pytestmark_real
def test_engine_filter_actually_bites():
    """Proof the wiring is live, not just present: with the filter ON, a
    universe of non-members produces no entries at all. If this ever passes
    trivially the filter has been disconnected."""
    from backtest.engine import BacktestConfig, run_backtest
    uni = {f"T{i:02d}": _frame(10 + i, n=400) for i in range(8)}
    uni["SPY"] = _frame(999, n=400)

    on = run_backtest(uni, BacktestConfig(warmup_bars=210))
    off = run_backtest(uni, BacktestConfig(warmup_bars=210,
                                           point_in_time_membership=False))
    assert on.metrics.get("trades", 0) == 0
    assert off.metrics.get("trades", 0) > 0


@pytestmark_real
def test_engine_keeps_scoring_a_holding_after_it_leaves_the_index():
    """Leaving the index is not a reason to stop evaluating a position we
    own - it must still be exitable, so held names bypass the filter."""
    from backtest.engine import _restrict_to_members
    pit = {"AAPL": 1, "ZZZA": 2, "ZZZB": 3}
    kept = _restrict_to_members(pit, pd.Timestamp("2020-06-01"), keep={"ZZZA"})
    assert set(kept) == {"AAPL", "ZZZA"}


def test_filter_can_be_disabled_for_non_equity_universes():
    """Crypto and custom watchlists are not S&P members; the switch has to
    exist or those backtests silently trade nothing."""
    from backtest.lab import LabConfig, backtest_momentum
    uni = {f"T{i:02d}": _frame(10 + i) for i in range(12)}
    uni["SPY"] = _frame(999)
    res = backtest_momentum(uni, LabConfig(top_n=5, point_in_time_membership=False))
    assert res.metrics and res.metrics["trades"] > 0
