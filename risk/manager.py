"""
risk/manager.py
Phase 5 - Central risk engine shared by both the stock weekly and crypto 24/7
strategies. One sizing path, one set of portfolio-level controls.

Responsibilities:
  - half-Kelly position sizing (a CAP on the fixed-fractional risk, never a multiplier)
  - per-trade risk caps (2% stocks, 1.5% crypto by default)
  - portfolio heat caps (sum of capital-at-risk / equity; 10% stocks, 40% crypto)
  - per-position notional cap (35% crypto; stocks governed by heat, cap defaults to 1.0)
  - regime-aware sizing (smaller in high-vol / mean-reverting)
  - monthly drawdown circuit breaker (-15% halts new entries)
  - sector / correlation-group exposure guard (data-driven once a map is supplied)
  - fee/slippage-aware cash ceiling (no margin; never allocate beyond cash net of costs)
  - $50 minimum position floor

Design notes:
  - Sizing math (half_kelly_fraction, size_position) is PURE and DB-free so strategies
    and tests can call it directly. DB-backed portfolio state (heat, NAV, drawdown,
    open symbols) lives on RiskManager.
  - The crypto funding haircut and the crypto skip-on-negative-funding rule stay in
    strategies/crypto_24h.py because they are instrument-specific; crypto passes the
    haircut in as `risk_multiplier` and handles the hard skip before calling here.
  - The monthly drawdown baseline is reconstructed on the fly from starting_capital
    plus realized PnL on closed Trades (no schema change, no new table). Open-position
    marks are included when current_price is populated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional

from database.db import (
    AssetType,
    MarketRegime,
    OrderStatus,
    Position,
    Trade,
    get_db,
)
from config import settings

logger = logging.getLogger(__name__)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Tunable defaults (overridable via RiskParams / RiskManager args)
# ---------------------------------------------------------------------------

STOCK_RISK_PER_TRADE = 0.02
CRYPTO_RISK_PER_TRADE = 0.015

STOCK_MAX_HEAT = 0.10              # total capital-at-risk across open stocks <= 10% equity
CRYPTO_MAX_HEAT = 0.40             # generous ceiling; rarely binds at 3 positions / 1.5% risk

STOCK_MAX_POSITION_NOTIONAL_PCT = 1.0    # no explicit single-name cap; heat governs
CRYPTO_MAX_POSITION_NOTIONAL_PCT = 0.35  # carried over from Phase 4

MONTHLY_DRAWDOWN_HALT = -0.15      # <= -15% intramonth halts new entries

MIN_POSITION_USD = 50.0

DEFAULT_WIN_RATE = 0.5             # placeholders until ML phase calibrates from live stats
DEFAULT_WIN_LOSS_RATIO = 1.5

# Per-asset default fee/slippage used only as a sizing-time cash buffer so the
# executor's real fills do not get rejected for insufficient cash. Realized fees
# are modeled in the execution layer (ccxt_crypto / alpaca), not double-counted here.
STOCK_FEE_RATE = 0.0
STOCK_SLIPPAGE = 0.0005            # 5 bps assumed slippage buffer on stock fills
CRYPTO_FEE_RATE = 0.0              # ccxt_crypto already models taker fees on size 0 here
CRYPTO_SLIPPAGE = 0.0

# Regime size multipliers. HIGH_VOL is halved because fixed ATR stops are unreliable
# when realized vol is high; MEAN_REVERTING is trimmed; TRENDING/UNKNOWN unscaled.
REGIME_SIZE_MULT: dict[MarketRegime, float] = {
    MarketRegime.TRENDING: 1.0,
    MarketRegime.MEAN_REVERTING: 0.75,
    MarketRegime.HIGH_VOL: 0.5,
    MarketRegime.UNKNOWN: 1.0,
}


# ---------------------------------------------------------------------------
# Parameter + result containers
# ---------------------------------------------------------------------------

@dataclass
class RiskParams:
    asset_type: AssetType
    risk_per_trade: float
    max_portfolio_heat: float
    max_position_notional_pct: float
    min_position_usd: float = MIN_POSITION_USD
    fee_rate: float = 0.0
    slippage_pct: float = 0.0
    max_positions: int = 0             # 0 = not enforced here (executor enforces)


@dataclass
class PositionSize:
    units: float
    notional: float
    effective_risk_pct: float
    stop_price: float
    take_profit: float
    reason: str = "ok"

    @property
    def tradable(self) -> bool:
        return self.units > 0.0 and self.notional > 0.0


def stock_params(
    risk_per_trade: float = STOCK_RISK_PER_TRADE,
    max_heat: float = STOCK_MAX_HEAT,
    max_position_notional_pct: float = STOCK_MAX_POSITION_NOTIONAL_PCT,
    min_position_usd: Optional[float] = None,
    fee_rate: float = STOCK_FEE_RATE,
    slippage_pct: float = STOCK_SLIPPAGE,
    max_positions: Optional[int] = None,
) -> RiskParams:
    return RiskParams(
        asset_type=AssetType.STOCK,
        risk_per_trade=risk_per_trade,
        max_portfolio_heat=max_heat,
        max_position_notional_pct=max_position_notional_pct,
        min_position_usd=min_position_usd if min_position_usd is not None else getattr(settings, "min_position_size", MIN_POSITION_USD),
        fee_rate=fee_rate,
        slippage_pct=slippage_pct,
        max_positions=max_positions if max_positions is not None else getattr(settings, "max_stock_positions", 8),
    )


def crypto_params(
    risk_per_trade: float = CRYPTO_RISK_PER_TRADE,
    max_heat: float = CRYPTO_MAX_HEAT,
    max_position_notional_pct: float = CRYPTO_MAX_POSITION_NOTIONAL_PCT,
    min_position_usd: Optional[float] = None,
    fee_rate: float = CRYPTO_FEE_RATE,
    slippage_pct: float = CRYPTO_SLIPPAGE,
    max_positions: Optional[int] = None,
) -> RiskParams:
    return RiskParams(
        asset_type=AssetType.CRYPTO,
        risk_per_trade=risk_per_trade,
        max_portfolio_heat=max_heat,
        max_position_notional_pct=max_position_notional_pct,
        min_position_usd=min_position_usd if min_position_usd is not None else getattr(settings, "min_position_size", MIN_POSITION_USD),
        fee_rate=fee_rate,
        slippage_pct=slippage_pct,
        max_positions=max_positions if max_positions is not None else getattr(settings, "max_crypto_positions", 3),
    )


# ---------------------------------------------------------------------------
# Pure sizing math (no DB)
# ---------------------------------------------------------------------------

def half_kelly_fraction(
    win_rate: float = DEFAULT_WIN_RATE,
    win_loss_ratio: float = DEFAULT_WIN_LOSS_RATIO,
) -> float:
    """
    Half-Kelly bankroll fraction. f* = W - (1-W)/R; returned value is f*/2,
    floored at 0 (a non-positive edge yields no allocation).
    """
    if win_loss_ratio <= 0:
        return 0.0
    f_star = win_rate - (1.0 - win_rate) / win_loss_ratio
    return max(0.0, f_star / 2.0)


def regime_multiplier(regime: Optional[MarketRegime]) -> float:
    if regime is None:
        return 1.0
    return REGIME_SIZE_MULT.get(regime, 1.0)


def size_position(
    params: RiskParams,
    equity: float,
    available_cash: float,
    entry_price: float,
    *,
    stop_price: Optional[float] = None,
    atr: Optional[float] = None,
    atr_stop_mult: float = 2.0,
    tp_r_mult: float = 3.0,
    regime: Optional[MarketRegime] = None,
    risk_multiplier: float = 1.0,
    win_rate: float = DEFAULT_WIN_RATE,
    win_loss_ratio: float = DEFAULT_WIN_LOSS_RATIO,
    current_heat: float = 0.0,
) -> PositionSize:
    """
    Size a long position under the shared risk model.

    effective risk = min(risk_per_trade, half_kelly) * regime_mult * risk_multiplier,
    then clamped so current_heat + effective risk <= max_portfolio_heat.

    Stop distance comes from `stop_price` (entry - stop_price) when provided, else
    from `atr` (atr_stop_mult * atr). Notional is capped at max_position_notional_pct
    of equity, then at available cash net of fee/slippage (no margin), and floored at
    min_position_usd.
    """
    if equity <= 0 or entry_price <= 0:
        return PositionSize(0.0, 0.0, 0.0, 0.0, 0.0, "invalid_inputs")

    # Determine stop distance.
    if stop_price is not None and stop_price > 0:
        stop_distance = entry_price - stop_price
    elif atr is not None and atr > 0:
        stop_distance = atr_stop_mult * atr
    else:
        return PositionSize(0.0, 0.0, 0.0, 0.0, 0.0, "invalid_inputs")

    if stop_distance <= 0:
        return PositionSize(0.0, 0.0, 0.0, 0.0, 0.0, "invalid_stop")

    kelly = half_kelly_fraction(win_rate, win_loss_ratio)
    effective_risk_pct = min(params.risk_per_trade, kelly)
    effective_risk_pct *= regime_multiplier(regime)
    effective_risk_pct *= max(0.0, risk_multiplier)

    if effective_risk_pct <= 0:
        return PositionSize(0.0, 0.0, 0.0, 0.0, 0.0, "no_edge")

    # Portfolio heat clamp: never let total capital-at-risk exceed the cap.
    heat_room = params.max_portfolio_heat - max(0.0, current_heat)
    if heat_room <= 0:
        return PositionSize(0.0, 0.0, effective_risk_pct, 0.0, 0.0, "heat_exceeded")
    if effective_risk_pct > heat_room:
        effective_risk_pct = heat_room

    eff_entry = entry_price * (1.0 + max(0.0, params.slippage_pct))
    final_stop = entry_price - stop_distance
    take_profit = entry_price + tp_r_mult * stop_distance

    risk_amount = equity * effective_risk_pct
    units = risk_amount / stop_distance
    notional = units * eff_entry

    # Per-position concentration cap.
    max_notional = equity * params.max_position_notional_pct
    if max_notional > 0 and notional > max_notional:
        notional = max_notional
        units = notional / eff_entry

    # No margin: cannot deploy more than available cash net of fee/slippage buffer.
    if available_cash > 0:
        cash_ceiling = available_cash / (1.0 + max(0.0, params.fee_rate))
        if notional > cash_ceiling:
            notional = cash_ceiling
            units = notional / eff_entry

    if notional < params.min_position_usd:
        return PositionSize(0.0, notional, effective_risk_pct, final_stop, take_profit,
                            "below_min_position")

    return PositionSize(units, notional, effective_risk_pct, final_stop, take_profit, "ok")


# ---------------------------------------------------------------------------
# Sector / correlation-group exposure guard (pure)
# ---------------------------------------------------------------------------

def check_group_exposure(
    open_symbols: Iterable[str],
    new_symbol: str,
    group_map: Optional[dict[str, str]],
    max_per_group: int,
) -> tuple[bool, str]:
    """
    True if adding new_symbol keeps the count within max_per_group for its group.
    group_map maps symbol -> group label (sector for stocks, correlation cluster for
    crypto). If group_map is None or the symbol is unmapped, the guard is a no-op
    (returns True) and logs that no exposure data was supplied.
    """
    if not group_map or new_symbol not in group_map:
        logger.debug("group exposure guard skipped for %s (no group data)", new_symbol)
        return True, "no_group_data"

    group = group_map[new_symbol]
    count = sum(1 for s in open_symbols if group_map.get(s) == group)
    if count >= max_per_group:
        return False, f"group_full:{group}={count}>={max_per_group}"
    return True, "ok"


# ---------------------------------------------------------------------------
# DB-backed portfolio state + controls
# ---------------------------------------------------------------------------

class RiskManager:
    """Portfolio-level risk state and gating, backed by the live DB."""

    def __init__(
        self,
        starting_capital: Optional[float] = None,
        monthly_halt: float = MONTHLY_DRAWDOWN_HALT,
    ) -> None:
        self.starting_capital = (
            starting_capital if starting_capital is not None
            else float(getattr(settings, "starting_capital", 1000.0))
        )
        self.monthly_halt = monthly_halt

    # -- realized / NAV -----------------------------------------------------

    def realized_pnl(
        self,
        *,
        before: Optional[datetime] = None,
        after: Optional[datetime] = None,
        asset_type: Optional[AssetType] = None,
    ) -> float:
        with get_db() as db:
            q = db.query(Trade).filter(
                Trade.status == OrderStatus.FILLED,
                Trade.pnl.isnot(None),
                Trade.closed_at.isnot(None),
            )
            if asset_type is not None:
                q = q.filter(Trade.asset_type == asset_type)
            if before is not None:
                q = q.filter(Trade.closed_at < before)
            if after is not None:
                q = q.filter(Trade.closed_at >= after)
            return float(sum((t.pnl or 0.0) for t in q.all()))

    def _open_unrealized(self, asset_type: Optional[AssetType] = None) -> float:
        with get_db() as db:
            q = db.query(Position)
            if asset_type is not None:
                q = q.filter(Position.asset_type == asset_type)
            total = 0.0
            for p in q.all():
                if p.current_price is not None and p.entry_price is not None and p.quantity is not None:
                    total += (float(p.current_price) - float(p.entry_price)) * float(p.quantity)
            return total

    def current_nav(self) -> float:
        """starting_capital + all realized PnL + marked open-position PnL."""
        return self.starting_capital + self.realized_pnl() + self._open_unrealized()

    def month_start_nav(self, now: Optional[datetime] = None) -> float:
        """NAV baseline as of the first instant of the current UTC month."""
        now = now or datetime.now(UTC)
        month_start = datetime(now.year, now.month, 1, tzinfo=UTC)
        return self.starting_capital + self.realized_pnl(before=month_start)

    def monthly_drawdown(self, now: Optional[datetime] = None) -> float:
        """Fractional drawdown vs the month-start NAV. 0.0 if baseline non-positive."""
        baseline = self.month_start_nav(now)
        if baseline <= 0:
            return 0.0
        return (self.current_nav() - baseline) / baseline

    def is_halted(self, now: Optional[datetime] = None) -> bool:
        dd = self.monthly_drawdown(now)
        halted = dd <= self.monthly_halt
        if halted:
            logger.warning("DRAWDOWN HALT active: monthly drawdown %.2f%% <= %.2f%%",
                           dd * 100, self.monthly_halt * 100)
        return halted

    # -- heat / open positions ---------------------------------------------

    def open_symbols(self, asset_type: Optional[AssetType] = None) -> list[str]:
        with get_db() as db:
            q = db.query(Position.symbol)
            if asset_type is not None:
                q = q.filter(Position.asset_type == asset_type)
            return [r[0] for r in q.all()]

    def open_position_count(self, asset_type: Optional[AssetType] = None) -> int:
        with get_db() as db:
            q = db.query(Position)
            if asset_type is not None:
                q = q.filter(Position.asset_type == asset_type)
            return q.count()

    def portfolio_heat(self, asset_type: Optional[AssetType] = None, equity: Optional[float] = None) -> float:
        """
        Sum of capital-at-risk across open positions / equity, where per-position
        risk = (entry_price - stop_loss) * quantity (floored at 0). Uses current NAV
        as equity unless an explicit equity is supplied.
        """
        eq = equity if equity is not None else self.current_nav()
        if eq <= 0:
            return 0.0
        with get_db() as db:
            q = db.query(Position)
            if asset_type is not None:
                q = q.filter(Position.asset_type == asset_type)
            risk_total = 0.0
            for p in q.all():
                if p.stop_loss is None or p.entry_price is None or p.quantity is None:
                    continue
                per = (float(p.entry_price) - float(p.stop_loss)) * float(p.quantity)
                if per > 0:
                    risk_total += per
            return risk_total / eq

    # -- composite pre-trade gate ------------------------------------------

    def pre_trade_ok(
        self,
        params: RiskParams,
        symbol: str,
        *,
        now: Optional[datetime] = None,
        group_map: Optional[dict[str, str]] = None,
        max_per_group: int = 0,
    ) -> tuple[bool, str]:
        """
        Portfolio-level gate to run before sizing an entry:
          - drawdown halt
          - duplicate symbol
          - max positions for the asset class
          - sector / correlation-group exposure (if a map is supplied)
        Heat is enforced inside size_position via current_heat; pass
        portfolio_heat(asset_type) into size_position when sizing.
        """
        if self.is_halted(now):
            return False, "drawdown_halt"

        open_syms = self.open_symbols(params.asset_type)
        if symbol in open_syms:
            return False, "already_open"

        if params.max_positions and len(open_syms) >= params.max_positions:
            return False, f"max_positions:{len(open_syms)}>={params.max_positions}"

        if group_map and max_per_group:
            ok, reason = check_group_exposure(open_syms, symbol, group_map, max_per_group)
            if not ok:
                return False, reason

        return True, "ok"
