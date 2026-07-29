"""
backtest/crypto_lab.py
Test the crypto strategy before trusting it with more of the account.

The 24/7 crypto strategy has never been validated and - because of the
btc_only symbol bug - never actually executed a trade. Widening its universe
before measuring it would repeat exactly the mistake the stock side just
made: an untested strategy that felt reasonable and lost money.

Variants, all on identical data and costs:

  hold_btc        The bar. In crypto, buy-and-hold BTC beats most active
                  strategies, and the fee drag is a fraction of theirs.
  regime          The LIVE strategy: long while the trend is intact (close
                  above EMA200, EMA9 above EMA21, ADX trending, MACD
                  positive), flat otherwise. Uses the real predicates from
                  strategies.crypto_24h, so this tests the shipped logic.
  momentum        Rank coins by trailing return, hold the top N, rebalance
                  weekly - the crypto analogue of the stock winner.

Alpaca crypto fees are ~0.25% PER SIDE (25bps), an order of magnitude worse
than equities, so the default cost here is deliberately harsh: a strategy
that only wins at equity-like costs does not survive contact with crypto.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig, BacktestResult, _metrics, _trading_dates
from strategies.crypto_24h import (
    above_ema200, classify_regime, ema_stacked_bullish, macd_positive,
)
from database.db import MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class CryptoLabConfig:
    initial_capital: float = 10_000.0
    cost_bps: float = 25.0            # Alpaca crypto taker, per side
    top_n: int = 3
    momentum_lookback: int = 30       # bars
    rebalance_every: int = 42         # bars between momentum rebalances
    warmup_bars: int = 210            # EMA200 needs this
    benchmark: str = "BTC/USD"


def _finish(points, trades, costs, universe, cfg) -> BacktestResult:
    equity = pd.Series(dict(points)).sort_index()
    bench = pd.Series(dtype="float64")
    bdf = universe.get(cfg.benchmark)
    if bdf is not None and not equity.empty:
        b = bdf.loc[(bdf.index >= equity.index[0]) & (bdf.index <= equity.index[-1])]
        if not b.empty:
            bench = pd.to_numeric(b["close"], errors="coerce").dropna()
    shim = BacktestConfig(initial_capital=cfg.initial_capital,
                          cost_bps=cfg.cost_bps, benchmark=cfg.benchmark)
    m = _metrics(equity, [], costs, bench, shim)
    m["trades"] = trades
    return BacktestResult(equity=equity, trades=[], benchmark=bench, metrics=m)


def _close(df, date) -> Optional[float]:
    try:
        if date in df.index:
            v = float(df.loc[date]["close"])
            return v if v > 0 else None
    except Exception:
        pass
    return None


def backtest_hold(universe: dict, cfg: Optional[CryptoLabConfig] = None,
                  symbol: Optional[str] = None) -> BacktestResult:
    cfg = cfg or CryptoLabConfig()
    sym = symbol or cfg.benchmark
    df = universe.get(sym)
    if df is None or df.empty:
        return BacktestResult()
    dates = _trading_dates({sym: df})[cfg.warmup_bars:]
    if len(dates) < 2:
        return BacktestResult()
    entry = _close(df, dates[0])
    if not entry:
        return BacktestResult()
    cost = cfg.initial_capital * cfg.cost_bps / 10_000.0
    units = (cfg.initial_capital - cost) / entry
    points = [(d, units * (_close(df, d) or entry)) for d in dates]
    return _finish(points, 1, cost, universe, cfg)


def backtest_regime(universe: dict, cfg: Optional[CryptoLabConfig] = None,
                    symbols: Optional[list[str]] = None) -> BacktestResult:
    """The live strategy, replayed. Equal weight across whichever coins are
    currently in an intact uptrend; flat in the ones that are not."""
    cfg = cfg or CryptoLabConfig()
    syms = symbols or [s for s in universe]
    dates = _trading_dates(universe)
    if len(dates) <= cfg.warmup_bars + 5:
        return BacktestResult()

    cash = cfg.initial_capital
    units: dict[str, float] = {}
    cost_rate = cfg.cost_bps / 10_000.0
    costs = 0.0
    trades = 0
    pending: Optional[set] = None
    points = []

    for i in range(cfg.warmup_bars, len(dates)):
        today = dates[i]

        if pending is not None:
            # exit what is no longer wanted, then split cash across entrants
            for s in list(units):
                if s in pending:
                    continue
                px = _close(universe[s], today)
                if not px:
                    continue
                proceeds = units.pop(s) * px
                cash += proceeds - proceeds * cost_rate
                costs += proceeds * cost_rate
                trades += 1
            entrants = [s for s in pending if s not in units]
            if entrants:
                each = cash / len(entrants)
                for s in entrants:
                    px = _close(universe[s], today)
                    if not px or each <= 0:
                        continue
                    spend = each / (1 + cost_rate)
                    units[s] = spend / px
                    cash -= spend + spend * cost_rate
                    costs += spend * cost_rate
                    trades += 1
            pending = None

        equity = cash + sum(u * (_close(universe[s], today) or 0.0)
                            for s, u in units.items())
        points.append((today, equity))

        if i + 1 >= len(dates):
            continue
        wanted = set()
        for s in syms:
            df = universe.get(s)
            if df is None:
                continue
            hist = df.loc[df.index <= today]          # point-in-time
            if len(hist) < cfg.warmup_bars:
                continue
            try:
                if (above_ema200(hist) and ema_stacked_bullish(hist)
                        and macd_positive(hist)
                        and classify_regime(hist) == MarketRegime.TRENDING):
                    wanted.add(s)
            except Exception:
                continue
        if wanted != set(units):
            pending = wanted

    return _finish(points, trades, costs, universe, cfg)


def backtest_crypto_momentum(universe: dict,
                             cfg: Optional[CryptoLabConfig] = None) -> BacktestResult:
    """Rank coins by trailing return, hold top N, rebalance periodically."""
    cfg = cfg or CryptoLabConfig()
    syms = list(universe)
    dates = _trading_dates(universe)
    if len(dates) <= cfg.warmup_bars + cfg.rebalance_every:
        return BacktestResult()

    closes = pd.DataFrame({
        s: pd.to_numeric(universe[s]["close"], errors="coerce")
        for s in syms if universe[s] is not None and "close" in universe[s]
    }).reindex(dates).ffill()

    cash = cfg.initial_capital
    units: dict[str, float] = {}
    cost_rate = cfg.cost_bps / 10_000.0
    costs, trades = 0.0, 0
    pending: Optional[list[str]] = None
    points = []

    for i in range(cfg.warmup_bars, len(dates)):
        today = dates[i]
        row = closes.loc[today]

        if pending is not None:
            for s in list(units):
                px = row.get(s)
                if not px or px <= 0:
                    continue
                proceeds = units.pop(s) * px
                cash += proceeds - proceeds * cost_rate
                costs += proceeds * cost_rate
                trades += 1
            if pending:
                each = cash / len(pending)
                for s in pending:
                    px = row.get(s)
                    if not px or px <= 0:
                        continue
                    spend = each / (1 + cost_rate)
                    units[s] = spend / px
                    cash -= spend + spend * cost_rate
                    costs += spend * cost_rate
                    trades += 1
            pending = None

        points.append((today, cash + sum(u * (row.get(s) or 0.0)
                                         for s, u in units.items())))

        if i + 1 >= len(dates) or (i - cfg.warmup_bars) % cfg.rebalance_every:
            continue
        hist = closes.loc[:today]
        if len(hist) < cfg.momentum_lookback + 1:
            continue
        mom = (hist.iloc[-1] / hist.iloc[-(cfg.momentum_lookback + 1)] - 1.0)
        mom = mom.replace([np.inf, -np.inf], np.nan).dropna()
        mom = mom[mom > 0]                     # never hold a downtrend
        pending = list(mom.sort_values(ascending=False).head(cfg.top_n).index)

    return _finish(points, trades, costs, universe, cfg)


CRYPTO_STRATEGIES = {
    "hold_btc": backtest_hold,
    "regime": backtest_regime,
    "momentum": backtest_crypto_momentum,
}
