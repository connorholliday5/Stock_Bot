"""
strategies/momentum_monthly.py
Cross-sectional momentum, monthly rebalance - the live version of the
variant that survived backtesting (backtest/lab.py::backtest_momentum).

Rank every name by its return over the last ~12 months SKIPPING the most
recent ~1 month, hold the top N equal-weight, rebalance monthly. The skip is
not decoration: last month's return is dominated by short-term reversal,
which is what drowned the weekly-rotation signal.

Deliberate differences from the weekly rotation this replaces:
  - Monthly, not weekly. Turnover was the rotation's dominant cost (19% of
    capital over 3 years) and it bought nothing.
  - NO stop-loss. A 5% stop on normal equity volatility fires on noise and
    sells dips into recoveries; momentum's exit is "fell out of the ranking",
    which the monthly rebalance already handles. The monthly drawdown halt
    in RiskManager remains the portfolio-level backstop.
  - Equal weight, not risk-scaled. Every name gets equity/top_n, so the
    configured position count is what actually happens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from config import settings

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK = 252      # ~12 months of trading days
DEFAULT_SKIP = 21           # ~1 month, excluded from the ranking
DEFAULT_TOP_N = 10
MIN_BARS = DEFAULT_LOOKBACK + DEFAULT_SKIP + 5


@dataclass
class RebalancePlan:
    targets: list[str] = field(default_factory=list)
    sells: list[str] = field(default_factory=list)
    buys: list[str] = field(default_factory=list)
    target_value: float = 0.0
    ranked: Optional[pd.Series] = None

    def as_dict(self) -> dict:
        return {"targets": self.targets, "sells": self.sells,
                "buys": self.buys, "target_value": round(self.target_value, 2)}


def momentum_scores(
    universe: dict[str, pd.DataFrame],
    lookback: int = DEFAULT_LOOKBACK,
    skip: int = DEFAULT_SKIP,
) -> pd.Series:
    """symbol -> 12-1 momentum, highest first. Names with too little history
    are excluded rather than guessed at."""
    scores: dict[str, float] = {}
    need = lookback + skip + 1
    for symbol, df in (universe or {}).items():
        if df is None or "close" not in getattr(df, "columns", []):
            continue
        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(close) < need:
            continue
        past = float(close.iloc[-need])
        recent = float(close.iloc[-(skip + 1)])
        if past <= 0:
            continue
        scores[symbol] = recent / past - 1.0
    if not scores:
        return pd.Series(dtype="float64")
    return pd.Series(scores).sort_values(ascending=False)


def build_rebalance_plan(
    universe: dict[str, pd.DataFrame],
    open_symbols: list[str],
    equity: float,
    top_n: int = DEFAULT_TOP_N,
    lookback: int = DEFAULT_LOOKBACK,
    skip: int = DEFAULT_SKIP,
) -> RebalancePlan:
    """Target book = top N by momentum. Sell what fell out, buy what came in."""
    ranked = momentum_scores(universe, lookback, skip)
    if ranked.empty:
        logger.warning("momentum: no rankable symbols (insufficient history)")
        return RebalancePlan()

    targets = list(ranked.head(max(int(top_n), 1)).index)
    held = [s for s in (open_symbols or []) if s]
    sells = [s for s in held if s not in targets]
    buys = [s for s in targets if s not in held]
    target_value = (float(equity) / len(targets)) if targets else 0.0

    logger.info("momentum plan: hold=%d sell=%d buy=%d target=$%.2f each",
                len(targets) - len(buys), len(sells), len(buys), target_value)
    return RebalancePlan(targets=targets, sells=sells, buys=buys,
                         target_value=target_value, ranked=ranked)


def _last_close(universe: dict, symbol: str) -> float:
    df = universe.get(symbol)
    if df is None or df.empty or "close" not in df.columns:
        return 0.0
    try:
        return float(pd.to_numeric(df["close"], errors="coerce").dropna().iloc[-1])
    except Exception:
        return 0.0


def run_monthly_rebalance(
    universe: dict[str, pd.DataFrame],
    open_positions: list[dict],
    equity: float,
    available_cash: float,
    executor=None,
    top_n: int = DEFAULT_TOP_N,
    lookback: int = DEFAULT_LOOKBACK,
    skip: int = DEFAULT_SKIP,
    min_position_usd: float = 10.0,
    topup_tolerance: float = 0.25,
) -> dict:
    """Sell dropouts (frees cash), top up underweight holds, then buy entrants.

    topup_tolerance is the drift a held position may run below its target
    weight before capital is added - it lets deposits flow in without
    rebalancing the whole book every month.
    """
    held_map = {(p.get("symbol") or p.get("ticker")): p for p in (open_positions or [])}
    plan = build_rebalance_plan(universe, list(held_map), equity, top_n, lookback, skip)
    if not plan.targets:
        return {"status": "no_targets", **plan.as_dict()}

    sold = bought = topped = 0
    cash = float(available_cash)

    for symbol in plan.sells:
        price = _last_close(universe, symbol)
        pos = held_map.get(symbol) or {}
        if price <= 0:
            price = float(pos.get("current_price") or pos.get("entry_price") or 0.0)
        if price <= 0 or executor is None:
            continue
        result = executor.close_long(symbol, exit_price=price, reason="momentum_rotation")
        if result.get("status") == "filled":
            sold += 1
            cash += price * float(pos.get("quantity", 0.0) or 0.0)

    # Top up positions that are already held but sit below target weight.
    # Without this, deposited cash NEVER gets invested when the rankings do
    # not change: only new entrants were ever bought, so contributions would
    # pile up as idle cash forever. A tolerance band keeps this from churning
    # the book on small drifts.
    for symbol in plan.targets:
        if symbol in plan.buys or executor is None:
            continue
        pos = held_map.get(symbol) or {}
        price = _last_close(universe, symbol) or float(
            pos.get("current_price") or pos.get("entry_price") or 0.0)
        if price <= 0:
            continue
        current_value = float(pos.get("quantity", 0.0) or 0.0) * price
        shortfall = plan.target_value - current_value
        if (shortfall <= 0 or plan.target_value <= 0
                or shortfall / plan.target_value < topup_tolerance):
            continue
        spend = min(shortfall, cash)
        if spend < min_position_usd:
            continue
        result = executor.open_long(symbol=symbol, units=spend / price,
                                    entry_price=price, stop_loss=0.0,
                                    take_profit=0.0)
        if result.get("status") == "filled":
            topped += 1
            cash -= spend

    for symbol in plan.buys:
        price = _last_close(universe, symbol)
        if price <= 0 or executor is None:
            continue
        spend = min(plan.target_value, cash)
        if spend < min_position_usd:
            logger.info("momentum: %s skipped, only $%.2f of budget left", symbol, spend)
            continue
        units = spend / price
        # No stop: momentum exits by falling out of the ranking, and a tight
        # stop on equity noise sells dips into recoveries. Fields are kept
        # populated for the DB/dashboard contract.
        result = executor.open_long(symbol=symbol, units=units, entry_price=price,
                                    stop_loss=0.0, take_profit=0.0)
        if result.get("status") == "filled":
            bought += 1
            cash -= spend

    logger.info("momentum rebalance: sold=%d bought=%d topped_up=%d cash_left=$%.2f",
                sold, bought, topped, cash)
    return {"status": "ok", "sold": sold, "bought": bought, "topped_up": topped,
            **plan.as_dict()}


def top_n_from_settings() -> int:
    return int(getattr(settings, "momentum_top_n", DEFAULT_TOP_N))
