"""tests/test_phase13_momentum.py - the live momentum strategy."""

from __future__ import annotations

import os

for _k, _v in {"ALPACA_API_KEY": "test", "ALPACA_SECRET_KEY": "test",
               "STARTING_CAPITAL": "10000"}.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import pandas as pd
import pytest

from strategies.momentum_monthly import (
    build_rebalance_plan, momentum_scores, run_monthly_rebalance,
)


def _series(total_return: float, n: int = 300) -> pd.DataFrame:
    """A frame whose 12-1 momentum is exactly `total_return`-ish."""
    close = np.linspace(100.0, 100.0 * (1 + total_return), n)
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({"close": close}, index=idx)


def _universe() -> dict:
    return {
        "WINNER": _series(1.00),     # +100%
        "GOOD": _series(0.40),
        "FLAT": _series(0.00),
        "LOSER": _series(-0.30),
    }


class FakeExecutor:
    def __init__(self):
        self.opened, self.closed = [], []
        self.paper_cash = 0.0
        self.min_position_usd = 10.0

    def open_long(self, symbol, units, entry_price, stop_loss, take_profit,
                  week_number=None):
        self.opened.append((symbol, units, stop_loss))
        return {"status": "filled", "symbol": symbol}

    def close_long(self, symbol, exit_price=0.0, units=None, reason=""):
        self.closed.append((symbol, reason))
        return {"status": "filled", "symbol": symbol}


# --------------------------- ranking ---------------------------

def test_momentum_ranks_by_12_1_return():
    ranked = momentum_scores(_universe())
    assert list(ranked.index) == ["WINNER", "GOOD", "FLAT", "LOSER"]


def test_short_history_is_excluded_not_guessed():
    uni = _universe()
    uni["NEWCO"] = _series(5.0, n=30)      # huge return, too little history
    assert "NEWCO" not in momentum_scores(uni).index


def test_skip_window_excludes_recent_month():
    """12-1 must ignore the most recent ~month; a spike that happens only in
    the skip window must not lift the score."""
    base = _series(0.0, n=300)
    spiked = base.copy()
    spiked.iloc[-10:] = spiked.iloc[-10:] * 3.0     # only inside the skip
    plain = momentum_scores({"A": base})["A"]
    with_spike = momentum_scores({"A": spiked})["A"]
    assert with_spike == pytest.approx(plain)


# --------------------------- planning ---------------------------

def test_plan_targets_top_n_and_diffs_the_book():
    plan = build_rebalance_plan(_universe(), open_symbols=["LOSER", "GOOD"],
                                equity=10_000.0, top_n=2)
    assert plan.targets == ["WINNER", "GOOD"]
    assert plan.sells == ["LOSER"]        # fell out of the top N
    assert plan.buys == ["WINNER"]        # GOOD is already held, not re-bought
    assert plan.target_value == pytest.approx(5_000.0)


def test_equal_weight_makes_top_n_real():
    """Momentum sizes equity/top_n, so the configured count is achievable -
    unlike risk-based sizing, where 2% risk / 5% stop was a 40% position."""
    plan = build_rebalance_plan(_universe(), [], equity=10_000.0, top_n=4)
    assert plan.target_value == pytest.approx(2_500.0)
    assert len(plan.targets) == 4


def test_empty_universe_is_a_clean_noop():
    plan = build_rebalance_plan({}, [], equity=1000.0, top_n=5)
    assert plan.targets == [] and plan.sells == [] and plan.buys == []


# --------------------------- execution ---------------------------

def test_rebalance_sells_dropouts_then_buys_entrants():
    ex = FakeExecutor()
    held = [{"symbol": "LOSER", "quantity": 10.0, "current_price": 70.0}]
    result = run_monthly_rebalance(_universe(), held, equity=10_000.0,
                                   available_cash=10_000.0, executor=ex, top_n=2)
    assert result["status"] == "ok"
    assert [c[0] for c in ex.closed] == ["LOSER"]
    assert ex.closed[0][1] == "momentum_rotation"
    assert [o[0] for o in ex.opened] == ["WINNER", "GOOD"]


def test_sells_fund_buys_within_the_same_rebalance():
    """Sells run first so their proceeds are available to the buys - with a
    fully invested book there is otherwise no cash to rotate with."""
    ex = FakeExecutor()
    held = [{"symbol": "LOSER", "quantity": 100.0, "current_price": 70.0}]
    run_monthly_rebalance(_universe(), held, equity=10_000.0,
                          available_cash=0.0, executor=ex, top_n=2)
    assert ex.closed and ex.opened      # $0 starting cash still funded a buy


def test_no_stop_loss_is_set():
    """Momentum exits by falling out of the ranking. A tight stop on equity
    noise sells dips into recoveries - that is what sank the rotation."""
    ex = FakeExecutor()
    run_monthly_rebalance(_universe(), [], equity=10_000.0,
                          available_cash=10_000.0, executor=ex, top_n=2)
    assert all(stop == 0.0 for _, _, stop in ex.opened)


def test_rebalance_respects_available_cash():
    ex = FakeExecutor()
    run_monthly_rebalance(_universe(), [], equity=10_000.0,
                          available_cash=100.0, executor=ex, top_n=4,
                          min_position_usd=50.0)
    # only ~$100 of cash: cannot fund four $2,500 slots
    assert len(ex.opened) <= 2


def test_rebalance_without_executor_still_plans():
    result = run_monthly_rebalance(_universe(), [], equity=1_000.0,
                                   available_cash=1_000.0, executor=None, top_n=2)
    assert result["targets"] == ["WINNER", "GOOD"]
    assert result["sold"] == 0 and result["bought"] == 0
