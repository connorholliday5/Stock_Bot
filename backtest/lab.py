"""
backtest/lab.py
Strategy lab: test genuinely DIFFERENT hypotheses, not tweaks of one.

The weekly-rotation strategy was falsified by backtest/engine.py (-1.6% vs
SPY +79.2% over 3.3y). Retuning its parameters against the same window is
how a backtest gets manufactured into a lie, so this module tests distinct,
documented approaches under identical accounting.

The variants, cheapest-to-believe first:

  buy_hold        The bar everything must clear. Beta is the easiest money
                  in the market and most active strategies destroy value
                  relative to it.
  trend_filter    Hold the index while it is above its long moving average,
                  hold cash otherwise (Faber, "A Quantitative Approach to
                  Tactical Asset Allocation"). Aims at smaller drawdowns
                  rather than higher returns, and trades a few times a year.
  momentum_12_1   Cross-sectional momentum: rank on the last 12 months of
                  return SKIPPING the most recent month (that skip avoids
                  short-term reversal, which is why the classic formulation
                  uses it), hold the top N equal-weight, rebalance monthly.
                  One of the most replicated anomalies in the literature -
                  and slow enough that costs do not eat it.

All three use the same point-in-time discipline as the main engine:
decisions on bar T use data through T and execute at T+1's open, with
per-side costs charged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig, BacktestResult, _metrics, _trading_dates

logger = logging.getLogger(__name__)


@dataclass
class LabConfig:
    initial_capital: float = 10_000.0
    cost_bps: float = 10.0
    top_n: int = 10
    ma_window: int = 200
    momentum_lookback: int = 252      # ~12 months
    momentum_skip: int = 21           # skip most recent ~1 month
    warmup_bars: int = 260
    benchmark: str = "SPY"


def _px(df: pd.DataFrame, date, col: str = "close") -> Optional[float]:
    try:
        if date in df.index:
            v = float(df.loc[date][col])
            return v if v > 0 else None
    except Exception:
        pass
    return None


def _finish(equity_points, trades_count, costs, universe, cfg) -> BacktestResult:
    equity = pd.Series(dict(equity_points)).sort_index()
    bench = pd.Series(dtype="float64")
    bdf = universe.get(cfg.benchmark)
    if bdf is not None and not equity.empty:
        b = bdf.loc[(bdf.index >= equity.index[0]) & (bdf.index <= equity.index[-1])]
        if not b.empty:
            bench = pd.to_numeric(b["close"], errors="coerce").dropna()
    shim = BacktestConfig(initial_capital=cfg.initial_capital,
                          cost_bps=cfg.cost_bps, benchmark=cfg.benchmark)
    m = _metrics(equity, [], costs, bench, shim)
    m["trades"] = trades_count
    return BacktestResult(equity=equity, trades=[], benchmark=bench, metrics=m)


# ---------------------------------------------------------------------------
# 1. buy and hold - the bar to clear
# ---------------------------------------------------------------------------

def backtest_buy_hold(universe: dict, cfg: Optional[LabConfig] = None,
                      symbol: Optional[str] = None) -> BacktestResult:
    cfg = cfg or LabConfig()
    sym = symbol or cfg.benchmark
    df = universe.get(sym)
    if df is None or df.empty:
        return BacktestResult()
    dates = _trading_dates({sym: df})[cfg.warmup_bars:]
    if len(dates) < 2:
        return BacktestResult()

    entry = _px(df, dates[0], "open") or _px(df, dates[0])
    units = (cfg.initial_capital * (1 - cfg.cost_bps / 10_000.0)) / entry
    costs = cfg.initial_capital * cfg.cost_bps / 10_000.0
    points = [(d, units * (_px(df, d) or entry)) for d in dates]
    return _finish(points, 1, costs, universe, cfg)


# ---------------------------------------------------------------------------
# 2. trend filter - in the index above its MA, cash below
# ---------------------------------------------------------------------------

def backtest_trend_filter(universe: dict, cfg: Optional[LabConfig] = None,
                          symbol: Optional[str] = None) -> BacktestResult:
    cfg = cfg or LabConfig()
    sym = symbol or cfg.benchmark
    df = universe.get(sym)
    if df is None or df.empty:
        return BacktestResult()

    close = pd.to_numeric(df["close"], errors="coerce")
    ma = close.rolling(cfg.ma_window).mean()
    dates = _trading_dates({sym: df})
    if len(dates) <= cfg.warmup_bars + 2:
        return BacktestResult()

    cash, units, costs, trades = cfg.initial_capital, 0.0, 0.0, 0
    cost_rate = cfg.cost_bps / 10_000.0
    want_in = False
    points = []

    for i in range(cfg.warmup_bars, len(dates)):
        today = dates[i]
        open_px = _px(df, today, "open") or _px(df, today)
        close_px = _px(df, today)
        if close_px is None:
            continue

        # act on YESTERDAY's signal at today's open
        if want_in and units == 0.0 and open_px:
            spend = cash / (1 + cost_rate)
            units, cash = spend / open_px, cash - spend - spend * cost_rate
            costs += spend * cost_rate
            trades += 1
        elif not want_in and units > 0.0 and open_px:
            proceeds = units * open_px
            cash += proceeds - proceeds * cost_rate
            costs += proceeds * cost_rate
            units = 0.0
            trades += 1

        points.append((today, cash + units * close_px))

        m = ma.loc[today] if today in ma.index else np.nan
        if pd.notna(m):
            want_in = bool(close_px > m)

    return _finish(points, trades, costs, universe, cfg)


# ---------------------------------------------------------------------------
# 3. cross-sectional momentum, 12-1, monthly rebalance
# ---------------------------------------------------------------------------

def backtest_momentum(universe: dict, cfg: Optional[LabConfig] = None) -> BacktestResult:
    cfg = cfg or LabConfig()
    syms = [s for s in universe if s != cfg.benchmark]
    dates = _trading_dates(universe)
    if len(dates) <= cfg.warmup_bars + 25:
        return BacktestResult()

    closes = pd.DataFrame({
        s: pd.to_numeric(universe[s]["close"], errors="coerce")
        for s in syms if universe[s] is not None and "close" in universe[s]
    }).reindex(dates).ffill()

    cash, costs, trades = cfg.initial_capital, 0.0, 0
    holdings: dict[str, float] = {}
    cost_rate = cfg.cost_bps / 10_000.0
    pending: Optional[list[str]] = None
    points = []

    for i in range(cfg.warmup_bars, len(dates)):
        today = dates[i]
        row = closes.loc[today]

        # rebalance decided last bar, executed at today's open
        if pending is not None:
            for s, u in list(holdings.items()):
                px = _px(universe[s], today, "open") or row.get(s)
                if px and px > 0:
                    proceeds = u * px
                    cash += proceeds - proceeds * cost_rate
                    costs += proceeds * cost_rate
                    trades += 1
            holdings = {}
            if pending:
                each = cash / len(pending)
                for s in pending:
                    px = _px(universe[s], today, "open") or row.get(s)
                    if not px or px <= 0:
                        continue
                    spend = each / (1 + cost_rate)
                    holdings[s] = spend / px
                    cash -= spend + spend * cost_rate
                    costs += spend * cost_rate
                    trades += 1
            pending = None

        equity = cash + sum(u * (row.get(s) or 0.0) for s, u in holdings.items())
        points.append((today, equity))

        # month end -> pick next month's book from data through today
        is_last_of_month = (i + 1 >= len(dates)) or (dates[i + 1].month != today.month)
        if not is_last_of_month or i + 1 >= len(dates):
            continue
        hist = closes.loc[:today]
        if len(hist) < cfg.momentum_lookback + 1:
            continue
        past = hist.iloc[-(cfg.momentum_lookback + 1)]
        recent = hist.iloc[-(cfg.momentum_skip + 1)]
        mom = (recent / past - 1.0).replace([np.inf, -np.inf], np.nan).dropna()
        if mom.empty:
            continue
        pending = list(mom.sort_values(ascending=False).head(cfg.top_n).index)

    return _finish(points, trades, costs, universe, cfg)


STRATEGIES = {
    "buy_hold": backtest_buy_hold,
    "trend_filter": backtest_trend_filter,
    "momentum_12_1": backtest_momentum,
}
