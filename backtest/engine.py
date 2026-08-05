"""
backtest/engine.py
Historical replay of the LIVE weekly-rotation strategy.

Design rule #1: reuse the real decision code. This engine calls the same
stock_scorer.score_universe, stock_weekly.select_rotation_exits and
risk.manager.size_position that the scheduler calls on Monday morning. A
backtest that reimplements the strategy tests the reimplementation, not the
strategy - and would happily "validate" logic the bot does not run.

Design rule #2: point-in-time discipline. Every decision on bar T is made
from data through bar T only, and is EXECUTED at bar T+1's open. Peeking at
the close you trade on is the classic way a backtest invents an edge that
evaporates live.

What is modeled: entry/exit costs (spread + slippage + fees in bps), the 5%
stop with realistic gap fills (a gap-down opens below the stop, so the fill
is the open, not the stop), weekly rotation with the real keep-rank rule,
risk-based position sizing, and a cash constraint (no margin).

Design rule #3: candidates are filtered to the index members as of the
decision date (data/index_membership.py), so the engine cannot rank a
company that had not joined the index yet. Set
BacktestConfig.point_in_time_membership=False to measure how much that
correction is worth.

What is NOT modeled - read before trusting a number:
  - Survivorship, PARTIALLY corrected. Point-in-time membership stops us
    ranking names before they joined, but Alpaca has no bars for companies
    delisted years ago, so the losers that died cannot be traded in replay.
    Absolute returns remain optimistic, just less so.
  - Intrabar path: only OHLC is known, so a bar that touches both stop and
    target is resolved stop-first (conservative).
  - Dividends, borrow, taxes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from data.index_membership import filter_to_members
from risk.manager import size_position, stock_params
from strategies import stock_scorer
from strategies.stock_weekly import select_rotation_exits

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    top_n: int = 8                    # max concurrent positions
    keep_rank: int = 20               # rotation: hold while inside this rank
    stop_loss_pct: float = 0.05
    cost_bps: float = 10.0            # per side: spread + slippage + fees
    entry_weekday: int = 0            # 0 = Monday
    exit_weekday: int = 4             # 4 = Friday
    risk_per_trade: float = 0.02
    warmup_bars: int = 200            # need SMA200 before the first decision
    benchmark: str = "SPY"
    # Rank only names that were in the index on the decision date. Off = the
    # old survivorship-biased behaviour, kept so the difference is measurable.
    point_in_time_membership: bool = True


@dataclass
class BacktestTrade:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    units: float = 0.0
    reason: str = ""

    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.units

    @property
    def pnl_pct(self) -> float:
        cost = self.entry_price * self.units
        return (self.pnl / cost) if cost > 0 else 0.0

    @property
    def bars_held(self) -> Optional[int]:
        if self.exit_date is None:
            return None
        return (self.exit_date - self.entry_date).days


@dataclass
class BacktestResult:
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    trades: list[BacktestTrade] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    benchmark: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))

    def summary(self) -> str:
        m = self.metrics
        if not m:
            return "no results"
        lines = [
            f"Period        : {m.get('start')} -> {m.get('end')} ({m.get('years', 0):.2f}y)",
            f"Return        : {m.get('total_return_pct', 0):+.2f}%  "
            f"(CAGR {m.get('cagr_pct', 0):+.2f}%)",
            f"Benchmark     : {m.get('benchmark_return_pct', float('nan')):+.2f}%  "
            f"({m.get('benchmark')})",
            f"Max drawdown  : {m.get('max_drawdown_pct', 0):.2f}%",
            f"Sharpe        : {m.get('sharpe', 0):.2f}",
            f"Trades        : {m.get('trades', 0)}  "
            f"(win rate {m.get('win_rate_pct', 0):.1f}%)",
            f"Avg win/loss  : {m.get('avg_win_pct', 0):+.2f}% / {m.get('avg_loss_pct', 0):+.2f}%",
            f"Avg hold      : {m.get('avg_hold_days', 0):.1f} days",
            f"Costs paid    : ${m.get('total_costs', 0):,.2f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _trading_dates(universe: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    idx = None
    for df in universe.values():
        if df is None or df.empty:
            continue
        d = pd.DatetimeIndex(pd.to_datetime(df.index))
        idx = d if idx is None else idx.union(d)
    return (idx if idx is not None else pd.DatetimeIndex([])).sort_values()


def _slice_universe(universe: dict, upto: pd.Timestamp, min_bars: int) -> dict:
    """Point-in-time view: every frame truncated to bars <= `upto`."""
    out = {}
    for sym, df in universe.items():
        if df is None or df.empty:
            continue
        sub = df.loc[df.index <= upto]
        if len(sub) >= min_bars:
            out[sym] = sub
    return out


def _restrict_to_members(pit: dict, date: pd.Timestamp, keep=()) -> dict:
    """Drop names that were not in the index on `date`.

    `keep` is passed the currently-open positions so an existing holding is
    still scored (and can still be exited) after it leaves the index.

    When membership data is unavailable `members_on` returns an empty set;
    that means "unknown", so the universe passes through untouched rather
    than being emptied.
    """
    keep = set(keep)
    eligible = set(filter_to_members([s for s in pit if s not in keep], date))
    return {s: df for s, df in pit.items() if s in eligible or s in keep}


def _bar(df: pd.DataFrame, date: pd.Timestamp) -> Optional[pd.Series]:
    try:
        if date in df.index:
            return df.loc[date]
    except Exception:
        pass
    return None


def _metrics(equity: pd.Series, trades: list[BacktestTrade], costs: float,
             benchmark: pd.Series, cfg: BacktestConfig) -> dict:
    if equity.empty:
        return {}
    start, end = equity.index[0], equity.index[-1]
    years = max((end - start).days / 365.25, 1e-9)
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0

    daily = equity.pct_change().dropna()
    sharpe = 0.0
    if len(daily) > 2 and daily.std() > 0:
        sharpe = float(daily.mean() / daily.std() * np.sqrt(252))

    running_max = equity.cummax()
    drawdown = (equity / running_max - 1.0)
    closed = [t for t in trades if t.exit_price is not None]
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl < 0]
    holds = [t.bars_held for t in closed if t.bars_held is not None]

    bench_ret = float("nan")
    if benchmark is not None and not benchmark.empty:
        bench_ret = float(benchmark.iloc[-1] / benchmark.iloc[0] - 1.0) * 100

    return {
        "start": str(start.date()), "end": str(end.date()), "years": years,
        "total_return_pct": total_ret * 100,
        "cagr_pct": cagr * 100,
        "max_drawdown_pct": float(drawdown.min()) * 100,
        "sharpe": sharpe,
        "trades": len(closed),
        "win_rate_pct": (100.0 * len(wins) / len(closed)) if closed else 0.0,
        "avg_win_pct": float(np.mean([t.pnl_pct for t in wins]) * 100) if wins else 0.0,
        "avg_loss_pct": float(np.mean([t.pnl_pct for t in losses]) * 100) if losses else 0.0,
        "avg_hold_days": float(np.mean(holds)) if holds else 0.0,
        "total_costs": costs,
        "final_equity": float(equity.iloc[-1]),
        "benchmark": cfg.benchmark,
        "benchmark_return_pct": bench_ret,
    }


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------

def run_backtest(
    universe: dict[str, pd.DataFrame],
    cfg: Optional[BacktestConfig] = None,
    ml_scorer=None,
    ml_weight: float = 0.0,
) -> BacktestResult:
    """Replay the weekly rotation strategy over `universe` (ticker -> OHLCV
    frame WITH features already applied by data.fetcher.add_features)."""
    cfg = cfg or BacktestConfig()
    dates = _trading_dates(universe)
    if len(dates) <= cfg.warmup_bars + 5:
        logger.warning("backtest: not enough history (%d bars)", len(dates))
        return BacktestResult()

    cash = float(cfg.initial_capital)
    open_pos: dict[str, BacktestTrade] = {}
    stops: dict[str, float] = {}
    trades: list[BacktestTrade] = []
    equity_points: list[tuple[pd.Timestamp, float]] = []
    total_costs = 0.0
    pending_entries: list[str] = []      # decided on T, executed at T+1 open
    pending_exits: list[str] = []

    params = stock_params(risk_per_trade=cfg.risk_per_trade)
    cost_rate = cfg.cost_bps / 10_000.0

    bench_df = universe.get(cfg.benchmark)

    for i in range(cfg.warmup_bars, len(dates)):
        today = dates[i]

        # ---- 1. execute what yesterday decided, at today's OPEN -----------
        for sym in pending_exits:
            trade = open_pos.get(sym)
            bar = _bar(universe.get(sym, pd.DataFrame()), today)
            if trade is None or bar is None:
                continue
            px = float(bar.get("open", bar.get("close", 0.0)) or 0.0)
            if px <= 0:
                continue
            proceeds = px * trade.units
            cost = proceeds * cost_rate
            cash += proceeds - cost
            total_costs += cost
            trade.exit_date, trade.exit_price = today, px
            trade.reason = trade.reason or "rotation"
            trades.append(trade)
            open_pos.pop(sym, None)
            stops.pop(sym, None)
        pending_exits = []

        for sym in pending_entries:
            if sym in open_pos or len(open_pos) >= cfg.top_n:
                continue
            df = universe.get(sym)
            bar = _bar(df, today) if df is not None else None
            if bar is None:
                continue
            px = float(bar.get("open", bar.get("close", 0.0)) or 0.0)
            if px <= 0:
                continue
            equity_now = cash + sum(
                t.units * float(_bar(universe[s], today).get("close", t.entry_price))
                for s, t in open_pos.items() if _bar(universe.get(s, pd.DataFrame()), today) is not None
            )
            stop_px = px * (1.0 - cfg.stop_loss_pct)
            sizing = size_position(
                params, equity=equity_now, available_cash=cash,
                entry_price=px, stop_price=stop_px,
            )
            if not sizing.tradable or sizing.units <= 0:
                continue
            spend = sizing.units * px
            cost = spend * cost_rate
            if spend + cost > cash:
                continue
            cash -= spend + cost
            total_costs += cost
            open_pos[sym] = BacktestTrade(symbol=sym, entry_date=today,
                                          entry_price=px, units=sizing.units)
            stops[sym] = stop_px
        pending_entries = []

        # ---- 2. intraday stop checks on today's bar -----------------------
        for sym in list(open_pos.keys()):
            df = universe.get(sym)
            bar = _bar(df, today) if df is not None else None
            if bar is None:
                continue
            stop_px = stops.get(sym, 0.0)
            low = float(bar.get("low", bar.get("close", 0.0)) or 0.0)
            if stop_px <= 0 or low > stop_px:
                continue
            # Gap-down opens BELOW the stop: fill at the open, not the stop.
            open_px = float(bar.get("open", bar.get("close", 0.0)) or 0.0)
            fill = min(stop_px, open_px) if open_px > 0 else stop_px
            trade = open_pos[sym]
            proceeds = fill * trade.units
            cost = proceeds * cost_rate
            cash += proceeds - cost
            total_costs += cost
            trade.exit_date, trade.exit_price, trade.reason = today, fill, "stop_loss"
            trades.append(trade)
            open_pos.pop(sym, None)
            stops.pop(sym, None)

        # ---- 3. mark equity ------------------------------------------------
        marked = cash
        for sym, trade in open_pos.items():
            bar = _bar(universe.get(sym, pd.DataFrame()), today)
            px = float(bar.get("close", trade.entry_price)) if bar is not None else trade.entry_price
            marked += trade.units * px
        equity_points.append((today, marked))

        # ---- 4. decisions made from data THROUGH today, executed T+1 ------
        weekday = today.weekday()
        needs_scan = weekday in (cfg.entry_weekday, cfg.exit_weekday)
        if not needs_scan or i + 1 >= len(dates):
            continue

        pit = _slice_universe(universe, today, cfg.warmup_bars)
        # Only rank what was actually in the index on this date. Names we
        # already hold stay in `pit` so their exit signal is still computed -
        # dropping out of the index is not a reason to stop scoring a
        # position we own.
        if cfg.point_in_time_membership:
            pit = _restrict_to_members(pit, today, keep=open_pos.keys())
        if not pit:
            continue
        try:
            ranked = stock_scorer.score_universe(
                pit, ml_scorer=ml_scorer, ml_weight=ml_weight).ranked
        except Exception as exc:
            logger.warning("backtest scan failed on %s: %s", today.date(), exc)
            continue
        if ranked is None or ranked.empty:
            continue

        if weekday == cfg.exit_weekday and open_pos:
            held = [{"symbol": s, "ticker": s} for s in open_pos]
            pending_exits = [p["symbol"] for p in
                             select_rotation_exits(held, ranked, keep_rank=cfg.keep_rank)]

        if weekday == cfg.entry_weekday:
            room = cfg.top_n - len(open_pos)
            if room > 0:
                col = "ticker" if "ticker" in ranked.columns else "symbol"
                candidates = [t for t in ranked[col].head(cfg.top_n * 3)
                              if t not in open_pos and t != cfg.benchmark]
                pending_entries = candidates[:room]

    equity = pd.Series(dict(equity_points)).sort_index()

    bench = pd.Series(dtype="float64")
    if bench_df is not None and not equity.empty:
        b = bench_df.loc[(bench_df.index >= equity.index[0]) &
                         (bench_df.index <= equity.index[-1])]
        if not b.empty:
            bench = pd.to_numeric(b["close"], errors="coerce").dropna()

    return BacktestResult(
        equity=equity, trades=trades, benchmark=bench,
        metrics=_metrics(equity, trades, total_costs, bench, cfg),
    )
