"""
backtest/stress.py
Robustness: is the result an edge, or one lucky path through one history?

A single backtest number answers almost nothing. Four questions decide
whether a result is worth acting on, and each one has killed strategies that
looked fine on the headline figure:

1. COST STRESS. Our modeled costs are a guess. If the edge only exists at
   exactly the spread we assumed, it does not exist - real fills are worse
   than modeled roughly always. Require survival at 2x.

2. CONCENTRATION. If the entire P&L lives in a handful of fills, there is no
   strategy, there is a lottery ticket. Remove the best few and look again.

3. REGIME. A strategy that made all its money in one exceptional year is a
   bet on that year repeating. Remove the best calendar year.

4. PARAMETER PLATEAU. The most-cited overfit tell. A real effect degrades
   gently as you move a parameter; a fitted one collapses off its peak. The
   value you ship should be the CENTRE of a plateau, never the peak of a
   spike - the peak is where the noise happened to line up, and the true
   optimum is as likely to be on either side of it.

None of this rescues a strategy with no edge. It catches results that look
like an edge and are not.

    python -m backtest.stress --strategy momentum --years 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _annualized(equity: pd.Series) -> dict:
    """Return/CAGR/Sharpe/maxDD from an equity curve, so every stressed
    variant is scored the same way as the baseline."""
    eq = pd.Series(equity).dropna()
    if len(eq) < 3 or eq.iloc[0] <= 0:
        return {}
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    daily = eq.pct_change().dropna()
    sharpe = (float(daily.mean() / daily.std() * np.sqrt(252))
              if len(daily) > 2 and daily.std() > 0 else 0.0)
    dd = float((eq / eq.cummax() - 1.0).min())
    return {
        "total_return_pct": total * 100,
        "cagr_pct": (float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1.0)) * 100,
        "sharpe": sharpe,
        "max_drawdown_pct": dd * 100,
    }


def _recompound(returns: pd.Series, start: float) -> pd.Series:
    return start * (1.0 + returns).cumprod()


# ---------------------------------------------------------------------------
# 1. cost stress
# ---------------------------------------------------------------------------

def cost_stress(strategy_fn: Callable, universe: dict, cfg,
                multiples: tuple[float, ...] = (1.0, 2.0, 3.0)) -> list[tuple[float, dict]]:
    """Re-run at multiples of the modeled cost.

    Costs are the one input we know is optimistic: modeled bps ignore
    queue position, partial fills, and the fact that the size we want is
    exactly the size everyone else wants at the same moment.
    """
    out = []
    base = float(getattr(cfg, "cost_bps", 10.0))
    for m in multiples:
        try:
            res = strategy_fn(universe, replace(cfg, cost_bps=base * m))
        except Exception as exc:
            logger.warning("cost stress x%.1f failed: %s", m, exc)
            continue
        if res.metrics:
            out.append((m, dict(res.metrics)))
    return out


# ---------------------------------------------------------------------------
# 2. concentration
# ---------------------------------------------------------------------------

def drop_best_trades(result, n: int = 5) -> tuple[dict, str]:
    """Remove the n most profitable contributions and recompound.

    Uses individual trades when the strategy reports them (the rotation
    engine does). The lab strategies report only a trade count, so it falls
    back to removing the n best DAYS - a slightly different question with
    the same purpose, and the returned label says which one ran rather than
    letting the two be confused.
    """
    eq = pd.Series(getattr(result, "equity", pd.Series(dtype=float))).dropna()
    if len(eq) < 10:
        return {}, "no equity curve"

    trades = [t for t in (getattr(result, "trades", None) or [])
              if getattr(t, "exit_price", None) is not None]
    if trades:
        winners = sorted(trades, key=lambda t: -float(getattr(t, "pnl", 0.0)))[:n]
        removed = sum(float(getattr(t, "pnl", 0.0)) for t in winners)
        start, end = float(eq.iloc[0]), float(eq.iloc[-1])
        adjusted = end - removed
        years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
        if adjusted <= 0:
            return ({"total_return_pct": -100.0, "cagr_pct": -100.0,
                     "sharpe": float("nan"), "max_drawdown_pct": -100.0},
                    f"{len(winners)} best trades removed (wipes out the account)")
        return ({"total_return_pct": (adjusted / start - 1.0) * 100,
                 "cagr_pct": ((adjusted / start) ** (1 / years) - 1.0) * 100,
                 "sharpe": float("nan"),      # path is not reconstructible
                 "max_drawdown_pct": float("nan")},
                f"{len(winners)} best trades removed")

    rets = eq.pct_change().dropna()
    if len(rets) <= n:
        return {}, "not enough return history"
    keep = rets.drop(rets.nlargest(n).index)
    return _annualized(_recompound(keep, float(eq.iloc[0]))), f"{n} best DAYS removed"


# ---------------------------------------------------------------------------
# 3. regime
# ---------------------------------------------------------------------------

def drop_best_year(result) -> tuple[dict, str]:
    """Recompound with the single best calendar year's returns removed."""
    eq = pd.Series(getattr(result, "equity", pd.Series(dtype=float))).dropna()
    rets = eq.pct_change().dropna()
    if len(rets) < 60:
        return {}, "not enough history"
    by_year = (1.0 + rets).groupby(rets.index.year).prod() - 1.0
    if len(by_year) < 2:
        return {}, "needs at least two calendar years"
    best = int(by_year.idxmax())
    keep = rets[rets.index.year != best]
    if keep.empty:
        return {}, "nothing left after removing the best year"
    return (_annualized(_recompound(keep, float(eq.iloc[0]))),
            f"{best} removed (it returned {by_year.max() * 100:.1f}%)")


# ---------------------------------------------------------------------------
# 4. parameter plateau
# ---------------------------------------------------------------------------

@dataclass
class Plateau:
    param: str
    values: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    peak_value: float = 0.0
    peak_score: float = 0.0
    centre_value: float = 0.0        # ship THIS, not the peak
    centre_score: float = 0.0
    is_plateau: bool = False

    def summary(self) -> str:
        shape = "plateau" if self.is_plateau else "SPIKE (overfit tell)"
        return (f"{self.param:<20} peak={self.peak_value:g} ({self.peak_score:.2f})  "
                f"ship={self.centre_value:g} ({self.centre_score:.2f})  {shape}")


def parameter_plateau(strategy_fn: Callable, universe: dict, cfg, param: str,
                      span: float = 0.25, points: int = 5,
                      metric: str = "sharpe", ratio: float = 0.6) -> Plateau:
    """Sweep one parameter +/- `span` and decide plateau vs spike.

    `ratio` is how much of the peak the peak's NEIGHBOURS must retain. A
    real effect is smooth: nudging a parameter 12% should not halve the
    result. If it does, the peak is noise that happened to line up, and the
    live value will land somewhere on the collapse.

    The shipped value is the centre of the best 3-point neighbourhood, not
    the peak - the true optimum is as likely to sit either side of the
    observed maximum, so the middle of a good region is the safer bet.
    """
    base = getattr(cfg, param, None)
    if base is None:
        return Plateau(param=param)
    is_int = isinstance(base, int)

    lo, hi = float(base) * (1 - span), float(base) * (1 + span)
    grid = np.linspace(lo, hi, points)
    values: list[float] = []
    scores: list[float] = []
    for v in grid:
        val = int(round(v)) if is_int else float(v)
        if val <= 0 or (values and val == values[-1]):
            continue
        try:
            res = strategy_fn(universe, replace(cfg, **{param: val}))
        except Exception as exc:
            logger.warning("plateau %s=%s failed: %s", param, val, exc)
            continue
        if not res.metrics:
            continue
        values.append(val)
        scores.append(float(res.metrics.get(metric, 0.0)))

    p = Plateau(param=param, values=values, scores=scores)
    if not scores:
        return p

    arr = np.array(scores)
    i_peak = int(arr.argmax())
    p.peak_value, p.peak_score = values[i_peak], float(arr[i_peak])

    # neighbours of the peak must retain `ratio` of it (sign-aware: a
    # negative peak means the strategy loses money everywhere, which is not
    # a plateau worth shipping)
    neighbours = [arr[j] for j in (i_peak - 1, i_peak + 1) if 0 <= j < len(arr)]
    p.is_plateau = bool(
        p.peak_score > 0 and neighbours
        and all(nb >= p.peak_score * ratio for nb in neighbours))

    # centre = best 3-point neighbourhood average
    if len(arr) >= 3:
        smoothed = [(float(arr[j - 1:j + 2].mean()), j) for j in range(1, len(arr) - 1)]
        best_avg, j = max(smoothed)
        p.centre_value, p.centre_score = values[j], best_avg
    else:
        p.centre_value, p.centre_score = p.peak_value, p.peak_score
    return p


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

@dataclass
class StressReport:
    name: str
    baseline: dict = field(default_factory=dict)
    costs: list[tuple[float, dict]] = field(default_factory=list)
    no_top_trades: dict = field(default_factory=dict)
    no_top_trades_label: str = ""
    no_best_year: dict = field(default_factory=dict)
    no_best_year_label: str = ""
    plateaus: list[Plateau] = field(default_factory=list)

    def survives_2x_costs(self) -> Optional[bool]:
        for m, met in self.costs:
            if abs(m - 2.0) < 1e-9:
                return met.get("cagr_pct", 0.0) > 0 and met.get("sharpe", 0.0) > 0
        return None

    def survives_trade_removal(self) -> Optional[bool]:
        if not self.no_top_trades:
            return None
        return self.no_top_trades.get("cagr_pct", 0.0) > 0

    def survives_year_removal(self) -> Optional[bool]:
        if not self.no_best_year:
            return None
        return self.no_best_year.get("cagr_pct", 0.0) > 0

    def all_plateaus(self) -> Optional[bool]:
        checked = [p for p in self.plateaus if p.scores]
        return all(p.is_plateau for p in checked) if checked else None


def stress_strategy(strategy_fn: Callable, universe: dict, cfg, name: str = "strategy",
                    params: tuple[str, ...] = (), drop_n: int = 5,
                    multiples: tuple[float, ...] = (1.0, 2.0, 3.0),
                    span: float = 0.25, points: int = 5) -> StressReport:
    rep = StressReport(name=name)
    base = strategy_fn(universe, cfg)
    if not base.metrics:
        logger.warning("%s produced no baseline metrics", name)
        return rep
    rep.baseline = dict(base.metrics)
    rep.costs = cost_stress(strategy_fn, universe, cfg, multiples)
    rep.no_top_trades, rep.no_top_trades_label = drop_best_trades(base, drop_n)
    rep.no_best_year, rep.no_best_year_label = drop_best_year(base)
    for p in params:
        rep.plateaus.append(
            parameter_plateau(strategy_fn, universe, cfg, p, span=span, points=points))
    return rep


def _verdict(ok: Optional[bool]) -> str:
    return "n/a" if ok is None else ("PASS" if ok else "FAIL")


def print_report(rep: StressReport) -> None:
    b = rep.baseline
    print()
    print("=" * 78)
    print(f"ROBUSTNESS: {rep.name}")
    print("-" * 78)
    print(f"baseline           {b.get('total_return_pct', 0):>9.2f}% "
          f"CAGR {b.get('cagr_pct', 0):>7.2f}%  Sharpe {b.get('sharpe', 0):>6.2f}  "
          f"maxDD {b.get('max_drawdown_pct', 0):>7.2f}%")
    print()
    print("cost stress (modeled costs are optimistic; real fills are worse)")
    for m, met in rep.costs:
        print(f"  {m:>4.1f}x costs      {met.get('total_return_pct', 0):>9.2f}% "
              f"CAGR {met.get('cagr_pct', 0):>7.2f}%  Sharpe {met.get('sharpe', 0):>6.2f}")
    print()
    if rep.no_top_trades:
        t = rep.no_top_trades
        print(f"concentration      {t.get('total_return_pct', 0):>9.2f}% "
              f"CAGR {t.get('cagr_pct', 0):>7.2f}%   ({rep.no_top_trades_label})")
    if rep.no_best_year:
        y = rep.no_best_year
        print(f"regime             {y.get('total_return_pct', 0):>9.2f}% "
              f"CAGR {y.get('cagr_pct', 0):>7.2f}%   ({rep.no_best_year_label})")
    if rep.plateaus:
        print()
        print("parameter surface (+/-25%) - ship the plateau centre, not the peak")
        for p in rep.plateaus:
            print("  " + (p.summary() if p.scores else f"{p.param:<20} not measurable"))
    print("-" * 78)
    print(f"  gate 4  survives 2x costs        {_verdict(rep.survives_2x_costs())}")
    print(f"  gate 5  parameters are plateaus  {_verdict(rep.all_plateaus())}")
    print(f"  gate 6  survives trade removal   {_verdict(rep.survives_trade_removal())}")
    print(f"          survives best-year drop  {_verdict(rep.survives_year_removal())}")
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser(description="Cost stress and robustness checks")
    ap.add_argument("--strategy", default="momentum",
                    choices=["momentum", "trend_filter", "buy_hold"])
    ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--drop", type=int, default=5, help="best trades/days to remove")
    ap.add_argument("--points", type=int, default=5, help="plateau sweep resolution")
    ap.add_argument("--no-pit", action="store_true",
                    help="disable point-in-time index membership (biased, for comparison)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from backtest.lab import (
        LabConfig, backtest_buy_hold, backtest_momentum, backtest_trend_filter,
    )
    from data.alpaca_data import default_stock_universe, fetch_stock_universe_alpaca

    fns = {"momentum": backtest_momentum, "trend_filter": backtest_trend_filter,
           "buy_hold": backtest_buy_hold}
    sweeps = {"momentum": ("top_n", "momentum_lookback", "momentum_skip"),
              "trend_filter": ("ma_window",), "buy_hold": ()}

    cfg = LabConfig(initial_capital=args.capital, cost_bps=args.cost_bps,
                    top_n=args.top_n,
                    point_in_time_membership=not args.no_pit)

    symbols = default_stock_universe()
    if cfg.benchmark not in symbols:
        symbols.append(cfg.benchmark)
    lookback = int(args.years * 365) + 500
    logger.info("fetching %d days...", lookback)
    universe = fetch_stock_universe_alpaca(lookback_days=lookback, tickers=symbols)
    if not universe:
        logger.error("no data - check ALPACA_API_KEY / ALPACA_SECRET_KEY")
        return 1
    logger.info("fetched %d symbols", len(universe))

    rep = stress_strategy(fns[args.strategy], universe, cfg, name=args.strategy,
                          params=sweeps[args.strategy], drop_n=args.drop,
                          points=args.points)
    if not rep.baseline:
        return 1
    print_report(rep)

    gates = [rep.survives_2x_costs(), rep.all_plateaus(), rep.survives_trade_removal()]
    return 0 if all(g is not False for g in gates) else 1


if __name__ == "__main__":
    sys.exit(main())
