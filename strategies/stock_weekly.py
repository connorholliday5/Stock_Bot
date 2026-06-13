"""
strategies/stock_weekly.py
Phase 5 - Stock weekly swing strategy: turns the Sunday scoring output into sized,
risk-gated Monday buy plans and Friday exit plans, all routed through the shared
risk manager (risk.manager) and the Alpaca executor (execution.alpaca).

This is the stock-side counterpart to strategies/crypto_24h.py. Scoring lives in
strategies/stock_scorer.py; entry/exit signal generation lives in strategies/signals.py.
This module is the missing link that was never built in Phase 3: scored universe ->
SignalEvents -> position sizing -> executor primitives.

Flow:
  Sunday  : stock_scorer.score_universe -> ranked DataFrame (handled upstream)
  Monday  : run_stock_weekly_buys  -> size + gate + (optional) execute open_long
  Mon-Thu : midweek stop handling stays in the scheduler/monitor (5% stop, SPY breaker)
  Friday  : run_stock_weekly_sells -> close all open stock positions

Sizing:
  - Fixed-fractional 2% risk per trade (shared engine), half-Kelly as a CAP
  - Percentage stop: stop = entry * (1 - stock_stop_loss); TP at stock_take_profit
    (take-profit R-multiple = take_profit_pct / stop_loss_pct)
  - Portfolio heat capped at 10% (shared engine), accumulated across the cycle
  - Monthly drawdown halt suppresses all new entries
  - $50 floor, no-margin cash ceiling, modeled slippage buffer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from database.db import AssetType
from config import settings
from risk.manager import (
    RiskManager,
    size_position as _rm_size_position,
    stock_params as _rm_stock_params,
)
from strategies.signals import (
    AssetClass,
    SignalEvent,
    SignalType,
    generate_stock_exit_signals,
    generate_stock_signals,
    persist_signals,
)

logger = logging.getLogger(__name__)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Tunable parameters (defaults sourced from settings)
# ---------------------------------------------------------------------------

STOCK_STOP_LOSS_PCT = float(getattr(settings, "stock_stop_loss", 0.05))
STOCK_TAKE_PROFIT_PCT = float(getattr(settings, "stock_take_profit", 0.10))
MAX_STOCK_POSITIONS = int(getattr(settings, "max_stock_positions", 8))
MIN_POSITION_USD = float(getattr(settings, "min_position_size", 50.0))
DEFAULT_TOP_N = 10

DEFAULT_WIN_RATE = 0.5
DEFAULT_WIN_LOSS_RATIO = 1.5


def _current_week_number(now: Optional[datetime] = None) -> int:
    now = now or datetime.now(UTC)
    return int(now.isocalendar()[1])


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class StockEntryPlan:
    symbol: str
    entry_price: float
    units: float
    notional: float
    stop_price: float
    take_profit: float
    effective_risk_pct: float
    week_number: int
    signal: SignalEvent
    reason: str = "all_gates_passed"


@dataclass
class StockExitPlan:
    symbol: str
    close_price: float
    reason: str
    signal: SignalEvent


@dataclass
class StockCycleResult:
    entries: list[StockEntryPlan] = field(default_factory=list)
    exits: list[StockExitPlan] = field(default_factory=list)
    executed: list[dict] = field(default_factory=list)
    persisted: int = 0


# ---------------------------------------------------------------------------
# Entry generation
# ---------------------------------------------------------------------------

def generate_week_entries(
    scored_df: pd.DataFrame,
    universe: dict[str, pd.DataFrame],
    equity: float,
    available_cash: float,
    open_positions: Optional[list[dict]] = None,
    risk_manager: Optional[RiskManager] = None,
    top_n: int = DEFAULT_TOP_N,
    max_positions: int = MAX_STOCK_POSITIONS,
    win_rate: float = DEFAULT_WIN_RATE,
    win_loss_ratio: float = DEFAULT_WIN_LOSS_RATIO,
    sector_map: Optional[dict[str, str]] = None,
    max_per_sector: int = 0,
) -> list[StockEntryPlan]:
    """
    Build sized, gated long entry plans from the scored universe.

    Pass a RiskManager to enforce the monthly drawdown halt and live portfolio heat;
    omit it to size against a cold portfolio (heat 0, dedup from open_positions only).
    Cash and heat are tracked across the cycle so a batch of entries cannot collectively
    breach the cash ceiling or the heat cap.
    """
    open_positions = open_positions or []
    held = {p.get("ticker") or p.get("symbol") for p in open_positions}

    # Portfolio-level kill switch.
    if risk_manager is not None and risk_manager.is_halted():
        logger.warning("Stock entries suppressed: monthly drawdown halt active")
        return []

    params = _rm_stock_params(max_positions=max_positions)
    tp_r_mult = (STOCK_TAKE_PROFIT_PCT / STOCK_STOP_LOSS_PCT) if STOCK_STOP_LOSS_PCT > 0 else 2.0
    week = _current_week_number()

    # Starting heat and open count come from the live book when a manager is present.
    running_heat = 0.0
    if risk_manager is not None:
        running_heat = risk_manager.portfolio_heat(AssetType.STOCK, equity=equity)
        held |= set(risk_manager.open_symbols(AssetType.STOCK))

    open_count = len(held)
    remaining_cash = available_cash

    buy_signals = generate_stock_signals(scored_df, universe, top_n=top_n)
    plans: list[StockEntryPlan] = []

    for event in buy_signals:
        symbol = event.ticker
        if open_count + len(plans) >= max_positions:
            logger.info("Max stock positions reached (%d) - no more entries", max_positions)
            break
        if symbol in held:
            continue

        # Sector exposure guard (no-op unless a sector_map is supplied).
        if sector_map and max_per_sector:
            from risk.manager import check_group_exposure
            current_syms = list(held) + [p.symbol for p in plans]
            ok, reason = check_group_exposure(current_syms, symbol, sector_map, max_per_sector)
            if not ok:
                logger.info("%s: skipped by sector guard (%s)", symbol, reason)
                continue

        entry_price = float(event.close_price)
        if entry_price <= 0:
            continue
        stop_price = entry_price * (1.0 - STOCK_STOP_LOSS_PCT)

        sizing = _rm_size_position(
            params,
            equity=equity,
            available_cash=remaining_cash,
            entry_price=entry_price,
            stop_price=stop_price,
            tp_r_mult=tp_r_mult,
            regime=None,
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio,
            current_heat=running_heat,
        )
        if not sizing.tradable:
            logger.info("%s: entry gated by sizing (%s)", symbol, sizing.reason)
            continue

        plans.append(StockEntryPlan(
            symbol=symbol,
            entry_price=entry_price,
            units=sizing.units,
            notional=sizing.notional,
            stop_price=sizing.stop_price,
            take_profit=sizing.take_profit,
            effective_risk_pct=sizing.effective_risk_pct,
            week_number=week,
            signal=event,
        ))
        running_heat += sizing.effective_risk_pct
        remaining_cash = max(0.0, remaining_cash - sizing.notional)
        logger.info(
            "ENTRY %s units=%.4f notional=$%.2f stop=%.2f tp=%.2f risk=%.3f%%",
            symbol, sizing.units, sizing.notional, sizing.stop_price, sizing.take_profit,
            sizing.effective_risk_pct * 100,
        )

    return plans


# ---------------------------------------------------------------------------
# Exit generation
# ---------------------------------------------------------------------------

def generate_week_exits(
    open_positions: list[dict],
    universe: dict[str, pd.DataFrame],
) -> list[StockExitPlan]:
    """Friday scheduled exit: close every open stock position."""
    exit_signals = generate_stock_exit_signals(open_positions, universe)
    plans: list[StockExitPlan] = []
    for event in exit_signals:
        plans.append(StockExitPlan(
            symbol=event.ticker,
            close_price=float(event.close_price),
            reason=event.notes or "weekly_scheduled_exit",
            signal=event,
        ))
        logger.info("EXIT %s reason=%s", event.ticker, event.notes)
    return plans


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------

def run_stock_weekly_buys(
    scored_df: pd.DataFrame,
    universe: dict[str, pd.DataFrame],
    equity: float,
    available_cash: float,
    executor=None,
    risk_manager: Optional[RiskManager] = None,
    open_positions: Optional[list[dict]] = None,
    top_n: int = DEFAULT_TOP_N,
    persist: bool = True,
    win_rate: float = DEFAULT_WIN_RATE,
    win_loss_ratio: float = DEFAULT_WIN_LOSS_RATIO,
    sector_map: Optional[dict[str, str]] = None,
    max_per_sector: int = 0,
) -> StockCycleResult:
    """
    Monday entry cycle: score -> size/gate -> persist signals -> (optional) execute.
    Pass an executor (execution.alpaca.StockExecutor) to actually place orders; omit it
    to get plans only.
    """
    entries = generate_week_entries(
        scored_df=scored_df,
        universe=universe,
        equity=equity,
        available_cash=available_cash,
        open_positions=open_positions,
        risk_manager=risk_manager,
        top_n=top_n,
        win_rate=win_rate,
        win_loss_ratio=win_loss_ratio,
        sector_map=sector_map,
        max_per_sector=max_per_sector,
    )

    persisted = 0
    if persist:
        persisted = persist_signals([p.signal for p in entries])

    executed: list[dict] = []
    if executor is not None:
        for plan in entries:
            result = executor.open_long(
                symbol=plan.symbol,
                units=plan.units,
                entry_price=plan.entry_price,
                stop_loss=plan.stop_price,
                take_profit=plan.take_profit,
                week_number=plan.week_number,
            )
            executed.append(result)

    return StockCycleResult(entries=entries, exits=[], executed=executed, persisted=persisted)


def run_stock_weekly_sells(
    open_positions: list[dict],
    universe: dict[str, pd.DataFrame],
    executor=None,
    persist: bool = True,
) -> StockCycleResult:
    """
    Friday exit cycle: build exit plans -> persist signals -> (optional) execute close_long.
    """
    exits = generate_week_exits(open_positions, universe)

    persisted = 0
    if persist:
        persisted = persist_signals([p.signal for p in exits])

    executed: list[dict] = []
    if executor is not None:
        for plan in exits:
            result = executor.close_long(
                symbol=plan.symbol,
                exit_price=plan.close_price,
                reason=plan.reason,
            )
            executed.append(result)

    return StockCycleResult(entries=[], exits=exits, executed=executed, persisted=persisted)
