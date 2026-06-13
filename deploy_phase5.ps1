# deploy_phase5.ps1 - run from repo root: C:\Users\cwhol\OneDrive\Desktop\Programming\stock_bot
# Writes all Phase 5 files. ASCII, no && , one statement per line.
New-Item -ItemType Directory -Force -Path risk | Out-Null
New-Item -ItemType Directory -Force -Path execution | Out-Null
New-Item -ItemType Directory -Force -Path strategies | Out-Null
New-Item -ItemType Directory -Force -Path tests | Out-Null
if (-not (Test-Path "risk\__init__.py")) { Set-Content -Path "risk\__init__.py" -Value "" -Encoding ascii }

$fManager = @'
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
'@
Set-Content -Path "risk\manager.py" -Value $fManager -Encoding ascii
Write-Host "wrote risk\manager.py"

$fAlpaca = @'
"""
execution/alpaca.py
Phase 5 - Stock order execution via Alpaca.

Long-only weekly swing. Defaults to PAPER mode (no real orders); set paper=False to
route real market orders through the Alpaca trading client. Position and Trade rows are
written in both modes so position state (and the max-position backstop) is real, exactly
like execution/ccxt_crypto.py.

This module executes; it does not decide. The strategy layer (stock_weekly) sizes and
gates via the shared risk manager; the scheduler hands primitives here.

The Alpaca SDK (alpaca-py) is imported lazily so this module loads and paper-trades with
no SDK present (tests run fully offline). Live mode requires `pip install alpaca-py>=0.20`.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from database.db import (
    AssetType,
    OrderSide,
    OrderStatus,
    Position,
    Trade,
    get_db,
)
from config import settings

logger = logging.getLogger(__name__)

UTC = timezone.utc

DEFAULT_FEE_RATE = 0.0            # Alpaca US equities are commission-free
DEFAULT_SLIPPAGE = 0.0005         # 5 bps modeled slippage on paper fills
MIN_POSITION_USD = 50.0


class StockExecutor:
    """Places (or simulates) Alpaca equity market orders and records them in the DB."""

    def __init__(
        self,
        paper: bool = True,
        client=None,
        paper_cash: float = 0.0,
        fee_rate: float = DEFAULT_FEE_RATE,
        slippage_pct: float = DEFAULT_SLIPPAGE,
        max_positions: Optional[int] = None,
    ) -> None:
        self.paper = paper
        self.client = client          # an alpaca TradingClient for live; None in paper
        self.paper_cash = paper_cash
        self.fee_rate = fee_rate
        self.slippage_pct = slippage_pct
        self.max_positions = (
            max_positions if max_positions is not None
            else int(getattr(settings, "max_stock_positions", 8))
        )

    # -- client bootstrap (live only) --------------------------------------

    @classmethod
    def from_settings(cls, paper: Optional[bool] = None) -> "StockExecutor":
        """
        Build a StockExecutor from config. In live mode this constructs an Alpaca
        TradingClient; in paper mode no SDK is required.
        """
        is_paper = settings.is_paper if paper is None else paper
        client = None
        if not is_paper:
            try:
                from alpaca.trading.client import TradingClient  # type: ignore
                client = TradingClient(
                    settings.alpaca_api_key,
                    settings.alpaca_secret_key,
                    paper=False,
                )
            except Exception as exc:
                logger.error("Alpaca client init failed, falling back to paper: %s", exc)
                is_paper = True
        return cls(paper=is_paper, client=client)

    # -- balances / state ---------------------------------------------------

    def get_cash(self) -> float:
        if self.paper:
            return self.paper_cash
        if self.client is None:
            logger.error("Live mode but no Alpaca client configured")
            return 0.0
        try:
            account = self.client.get_account()
            return float(getattr(account, "cash", 0.0))
        except Exception as exc:
            logger.error("get_account failed: %s", exc)
            return 0.0

    @staticmethod
    def count_open_stock_positions() -> int:
        with get_db() as db:
            return db.query(Position).filter(Position.asset_type == AssetType.STOCK).count()

    @staticmethod
    def open_stock_symbols() -> list[str]:
        with get_db() as db:
            rows = db.query(Position.symbol).filter(Position.asset_type == AssetType.STOCK).all()
            return [r[0] for r in rows]

    # -- order placement ----------------------------------------------------

    def _place_market_order(self, symbol: str, side: str, units: float, ref_price: float) -> dict:
        """
        Return a normalized fill: {price, units, fee, status}.
        Paper mode fills at ref_price adjusted for modeled slippage; live routes through
        the Alpaca trading client.
        """
        if units <= 0:
            return {"price": 0.0, "units": 0.0, "fee": 0.0, "status": "rejected"}

        if self.paper:
            slip = 1.0 + self.slippage_pct if side == "buy" else 1.0 - self.slippage_pct
            fill_price = ref_price * slip
            fee = fill_price * units * self.fee_rate
            return {"price": fill_price, "units": units, "fee": fee, "status": "filled"}

        if self.client is None:
            logger.error("Live order requested but no Alpaca client configured")
            return {"price": 0.0, "units": 0.0, "fee": 0.0, "status": "rejected"}

        try:
            from alpaca.trading.requests import MarketOrderRequest  # type: ignore
            from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce  # type: ignore

            req = MarketOrderRequest(
                symbol=symbol.replace("/", ""),
                qty=units,
                side=AlpacaSide.BUY if side == "buy" else AlpacaSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = self.client.submit_order(req)
        except Exception as exc:
            logger.error("submit_order failed for %s %s: %s", side, symbol, exc)
            return {"price": 0.0, "units": 0.0, "fee": 0.0, "status": "rejected"}

        price = float(getattr(order, "filled_avg_price", None) or ref_price)
        filled = float(getattr(order, "filled_qty", None) or units)
        fee = price * filled * self.fee_rate
        return {"price": price, "units": filled, "fee": fee, "status": "filled"}

    # -- public actions -----------------------------------------------------

    def open_long(
        self,
        symbol: str,
        units: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        week_number: Optional[int] = None,
    ) -> dict:
        """Open a long equity position. Enforces max-position and min-notional backstops."""
        notional = units * entry_price
        if notional < MIN_POSITION_USD:
            logger.warning("open_long %s rejected: notional $%.2f < min $%.2f", symbol, notional, MIN_POSITION_USD)
            return {"status": "rejected", "reason": "below_min_notional", "symbol": symbol}

        with get_db() as db:
            if db.query(Position).filter_by(symbol=symbol).first():
                logger.warning("open_long %s rejected: position already open", symbol)
                return {"status": "rejected", "reason": "already_open", "symbol": symbol}

            open_count = db.query(Position).filter(Position.asset_type == AssetType.STOCK).count()
            if open_count >= self.max_positions:
                logger.warning("open_long %s rejected: %d open >= max %d", symbol, open_count, self.max_positions)
                return {"status": "rejected", "reason": "max_positions", "symbol": symbol}

            fill = self._place_market_order(symbol, "buy", units, entry_price)
            if fill["status"] != "filled":
                return {"status": "rejected", "reason": "order_not_filled", "symbol": symbol}

            now = datetime.now(UTC)
            position = Position(
                asset_type=AssetType.STOCK,
                symbol=symbol,
                quantity=fill["units"],
                entry_price=fill["price"],
                current_price=fill["price"],
                stop_loss=stop_loss,
                take_profit=take_profit,
                opened_at=now,
                week_number=week_number,
            )
            trade = Trade(
                asset_type=AssetType.STOCK,
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=fill["units"],
                entry_price=fill["price"],
                stop_loss=stop_loss,
                take_profit=take_profit,
                status=OrderStatus.FILLED,
                opened_at=now,
            )
            db.add(position)
            db.add(trade)

        logger.info("OPEN_LONG %s units=%.6f @ %.2f (paper=%s)", symbol, fill["units"], fill["price"], self.paper)
        return {
            "status": "filled", "symbol": symbol, "side": "buy",
            "units": fill["units"], "price": fill["price"], "fee": fill["fee"],
        }

    def close_long(
        self,
        symbol: str,
        exit_price: float,
        units: Optional[float] = None,
        reason: str = "",
    ) -> dict:
        """Close an open long position and record the realized P&L."""
        with get_db() as db:
            position = db.query(Position).filter_by(symbol=symbol).first()
            if position is None:
                logger.warning("close_long %s rejected: no open position", symbol)
                return {"status": "rejected", "reason": "no_position", "symbol": symbol}

            close_units = units if units is not None else position.quantity
            entry_price = position.entry_price

            fill = self._place_market_order(symbol, "sell", close_units, exit_price)
            if fill["status"] != "filled":
                return {"status": "rejected", "reason": "order_not_filled", "symbol": symbol}

            gross = (fill["price"] - entry_price) * fill["units"]
            entry_fee = entry_price * fill["units"] * self.fee_rate
            pnl = gross - fill["fee"] - entry_fee
            cost_basis = entry_price * fill["units"]
            pnl_pct = (pnl / cost_basis) if cost_basis > 0 else 0.0

            now = datetime.now(UTC)
            trade = Trade(
                asset_type=AssetType.STOCK,
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=fill["units"],
                entry_price=entry_price,
                exit_price=fill["price"],
                status=OrderStatus.FILLED,
                pnl=pnl,
                pnl_pct=pnl_pct,
                opened_at=position.opened_at,
                closed_at=now,
                close_reason=reason[:50] if reason else None,
            )
            db.add(trade)
            db.delete(position)

        logger.info("CLOSE_LONG %s units=%.6f @ %.2f pnl=$%.2f (%s)", symbol, fill["units"], fill["price"], pnl, reason)
        return {
            "status": "filled", "symbol": symbol, "side": "sell",
            "units": fill["units"], "price": fill["price"], "pnl": pnl, "pnl_pct": pnl_pct,
            "reason": reason,
        }
'@
Set-Content -Path "execution\alpaca.py" -Value $fAlpaca -Encoding ascii
Write-Host "wrote execution\alpaca.py"

$fStockWeekly = @'
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
'@
Set-Content -Path "strategies\stock_weekly.py" -Value $fStockWeekly -Encoding ascii
Write-Host "wrote strategies\stock_weekly.py"

$fCrypto24h = @'
"""
strategies/crypto_24h.py
Phase 4 - Crypto 24/7 Trend-Following Strategy

Long-only spot strategy (no shorting, no margin) on 4H bars.

Entry (all conditions required):
  - Hard trend gate: close > EMA200 (never trade against the 200-period EMA)
  - Regime is TRENDING (ADX + realized volatility); HIGH_VOL and MEAN_REVERTING skipped
  - Bullish EMA 9/21 crossover within the lookback window
  - MACD confirmation: macd > signal and histogram > 0
  - Volume spike: volume_ratio >= threshold
  - Symbol not already held, open crypto positions < MAX_CRYPTO_POSITIONS
  - BTC-only until 60 days profitable (btc_only flag, default True)

Sizing:
  - Fixed-fractional risk of 1.5% per trade, with stop at ATR_STOP_MULT * ATR
  - half-Kelly acts as a CAP on that risk: effective risk = min(1.5%, half_kelly)
    so a weak/negative edge shrinks or zeroes the position, never enlarges it
  - Negative funding applies a size haircut; very negative funding skips the trade
  - Enforces MIN_POSITION_USD floor and no-margin cash ceiling

Exit (any condition):
  - close < EMA200, or bearish EMA 9/21 cross, or MACD histogram < 0,
    or ATR trailing-stop breach

Signals are persisted through strategies.signals.persist_signals (single DB write
path; SELL maps to SignalDirection.FLAT). Actionable plans are returned for the
executor; this module does not place orders itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from database.db import AssetType, MarketRegime
from risk.manager import (
    RiskManager,
    crypto_params as _rm_crypto_params,
    half_kelly_fraction as _rm_half_kelly,
    size_position as _rm_size_position,
)
from strategies.signals import (
    AssetClass,
    SignalEvent,
    SignalType,
    persist_signals,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

RISK_PER_TRADE = 0.015          # 1.5% of equity risked per trade
ATR_STOP_MULT = 2.0             # stop distance = 2 * ATR
TP_R_MULT = 3.0                 # take-profit at 3R (3:1 reward:risk)
MAX_CRYPTO_POSITIONS = 3
MAX_POSITION_NOTIONAL_PCT = 0.35   # cap any single position at 35% of equity
VOLUME_SPIKE_MIN = 1.5

ADX_TREND_MIN = 25.0            # ADX >= this => trending
ADX_CHOP_MAX = 20.0             # ADX <= this => mean reverting / choppy
REALIZED_VOL_HIGH = 0.10        # realized_vol >= this => high-vol regime

EMA_CROSS_LOOKBACK = 3          # bars within which a bullish cross still counts

FUNDING_HAIRCUT = 0.5           # size multiplier when funding < 0
FUNDING_SKIP = -0.001           # funding <= this => skip the trade entirely

DEFAULT_WIN_RATE = 0.5          # placeholder until live stats calibrate (later phase)
DEFAULT_WIN_LOSS_RATIO = 1.5

MIN_BARS_24H = 210              # need a valid EMA200 (200) plus headroom
MIN_POSITION_USD = 50.0

BTC_SYMBOL = "BTC/USDT"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class SizingResult:
    units: float
    notional: float
    effective_risk_pct: float
    stop_price: float
    take_profit: float
    funding_rate: float
    reason: str = ""

    @property
    def tradable(self) -> bool:
        return self.units > 0 and self.notional >= MIN_POSITION_USD


@dataclass
class CryptoEntryPlan:
    symbol: str
    entry_price: float
    units: float
    notional: float
    stop_price: float
    take_profit: float
    effective_risk_pct: float
    regime: str
    funding_rate: float
    signal: SignalEvent
    reason: str = ""


@dataclass
class CryptoExitPlan:
    symbol: str
    close_price: float
    reason: str
    signal: SignalEvent


@dataclass
class CryptoCycleResult:
    entries: list[CryptoEntryPlan] = field(default_factory=list)
    exits: list[CryptoExitPlan] = field(default_factory=list)
    persisted: int = 0


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

def classify_regime(
    df: pd.DataFrame,
    adx_trend_min: float = ADX_TREND_MIN,
    adx_chop_max: float = ADX_CHOP_MAX,
    rv_high: float = REALIZED_VOL_HIGH,
) -> MarketRegime:
    """
    Classify the latest bar's regime using ADX (trend strength) and realized
    volatility. High volatility takes precedence because it makes fixed ATR
    stops unreliable.
    """
    if df is None or df.empty:
        return MarketRegime.UNKNOWN

    last = df.iloc[-1]
    adx = last.get("adx", float("nan"))
    rv = last.get("realized_vol", float("nan"))

    adx = float(adx) if pd.notna(adx) else None
    rv = float(rv) if pd.notna(rv) else None

    if rv is not None and rv >= rv_high:
        return MarketRegime.HIGH_VOL
    if adx is None:
        return MarketRegime.UNKNOWN
    if adx >= adx_trend_min:
        return MarketRegime.TRENDING
    if adx <= adx_chop_max:
        return MarketRegime.MEAN_REVERTING
    return MarketRegime.UNKNOWN


# ---------------------------------------------------------------------------
# Signal primitives
# ---------------------------------------------------------------------------

def bullish_ema_cross(df: pd.DataFrame, lookback: int = EMA_CROSS_LOOKBACK) -> bool:
    """
    True if EMA9 is above EMA21 on the last bar AND crossed up from below within
    the last `lookback` bars.
    """
    if df is None or len(df) < lookback + 2:
        return False
    if "ema_9" not in df.columns or "ema_21" not in df.columns:
        return False

    ema9 = df["ema_9"]
    ema21 = df["ema_21"]
    if pd.isna(ema9.iloc[-1]) or pd.isna(ema21.iloc[-1]):
        return False
    if ema9.iloc[-1] <= ema21.iloc[-1]:
        return False

    diff = (ema9 - ema21)
    window = diff.iloc[-(lookback + 1):]
    # a cross-up exists if any adjacent pair flips from <=0 to >0 in the window
    prev = window.shift(1)
    crossed = ((prev <= 0) & (window > 0)).any()
    return bool(crossed)


def bearish_ema_cross(df: pd.DataFrame) -> bool:
    """True if EMA9 is at or below EMA21 on the last bar (trend turned down)."""
    if df is None or df.empty:
        return False
    if "ema_9" not in df.columns or "ema_21" not in df.columns:
        return False
    ema9 = df["ema_9"].iloc[-1]
    ema21 = df["ema_21"].iloc[-1]
    if pd.isna(ema9) or pd.isna(ema21):
        return False
    return bool(ema9 <= ema21)


def macd_confirms_long(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    if "macd" not in df.columns or "macd_signal" not in df.columns or "macd_hist" not in df.columns:
        return False
    last = df.iloc[-1]
    macd = last.get("macd")
    sig = last.get("macd_signal")
    hist = last.get("macd_hist")
    if pd.isna(macd) or pd.isna(sig) or pd.isna(hist):
        return False
    return bool(macd > sig and hist > 0)


def above_ema200(df: pd.DataFrame) -> bool:
    if df is None or df.empty or "ema_200" not in df.columns:
        return False
    last = df.iloc[-1]
    ema200 = last.get("ema_200")
    close = last.get("close")
    if pd.isna(ema200) or pd.isna(close):
        return False
    return bool(close > ema200)


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

def half_kelly_fraction(
    win_rate: float = DEFAULT_WIN_RATE,
    win_loss_ratio: float = DEFAULT_WIN_LOSS_RATIO,
) -> float:
    """
    Half-Kelly bankroll fraction. Delegates to the central risk engine
    (risk.manager.half_kelly_fraction) so both strategies share one Kelly model.
    """
    return _rm_half_kelly(win_rate, win_loss_ratio)


def compute_position_size(
    equity: float,
    available_cash: float,
    entry_price: float,
    atr: float,
    funding_rate: float = 0.0,
    risk_per_trade: float = RISK_PER_TRADE,
    atr_stop_mult: float = ATR_STOP_MULT,
    tp_r_mult: float = TP_R_MULT,
    win_rate: float = DEFAULT_WIN_RATE,
    win_loss_ratio: float = DEFAULT_WIN_LOSS_RATIO,
    current_heat: float = 0.0,
) -> SizingResult:
    """
    Size a long spot position via the central risk engine (risk.manager.size_position).

    Crypto-specific funding rules stay here: funding <= FUNDING_SKIP skips the trade;
    funding < 0 applies FUNDING_HAIRCUT to the effective risk (passed to the engine as
    risk_multiplier). Everything else - half-Kelly cap, 35% concentration cap, no-margin
    cash ceiling, $50 floor - is the shared engine. Outputs (units, notional, stop, TP,
    reasons) match the Phase 4 behavior exactly so existing tests stay green.
    """
    if entry_price <= 0 or atr <= 0 or equity <= 0:
        return SizingResult(0.0, 0.0, 0.0, 0.0, 0.0, funding_rate, "invalid_inputs")

    # Effective-risk reporting value for the hard-skip branch (pre-haircut, like Phase 4).
    kelly = half_kelly_fraction(win_rate, win_loss_ratio)
    pre_haircut_risk = min(risk_per_trade, kelly)

    if funding_rate <= FUNDING_SKIP:
        return SizingResult(0.0, 0.0, pre_haircut_risk, 0.0, 0.0, funding_rate,
                            f"funding_too_negative={funding_rate:.5f}")

    risk_mult = FUNDING_HAIRCUT if funding_rate < 0 else 1.0

    params = _rm_crypto_params(
        risk_per_trade=risk_per_trade,
        max_position_notional_pct=MAX_POSITION_NOTIONAL_PCT,
        min_position_usd=MIN_POSITION_USD,
        fee_rate=0.0,
        slippage_pct=0.0,
    )
    ps = _rm_size_position(
        params,
        equity=equity,
        available_cash=available_cash,
        entry_price=entry_price,
        atr=atr,
        atr_stop_mult=atr_stop_mult,
        tp_r_mult=tp_r_mult,
        regime=None,                 # crypto gates HIGH_VOL out before sizing
        risk_multiplier=risk_mult,
        win_rate=win_rate,
        win_loss_ratio=win_loss_ratio,
        current_heat=current_heat,
    )

    # Translate engine reasons into the Phase 4 crypto vocabulary.
    if ps.reason == "no_edge":
        return SizingResult(0.0, 0.0, 0.0, 0.0, 0.0, funding_rate, "no_edge_zero_risk")
    if ps.reason == "below_min_position":
        return SizingResult(0.0, ps.notional, ps.effective_risk_pct, ps.stop_price,
                            ps.take_profit, funding_rate, f"below_min_position=${ps.notional:.2f}")
    if ps.reason in ("invalid_inputs", "invalid_stop"):
        return SizingResult(0.0, 0.0, 0.0, 0.0, 0.0, funding_rate, "invalid_inputs")
    if ps.reason == "heat_exceeded":
        return SizingResult(0.0, 0.0, ps.effective_risk_pct, 0.0, 0.0, funding_rate, "heat_exceeded")

    return SizingResult(ps.units, ps.notional, ps.effective_risk_pct, ps.stop_price,
                        ps.take_profit, funding_rate, "ok")


# ---------------------------------------------------------------------------
# Signal-event builders
# ---------------------------------------------------------------------------

def _last_float(df: pd.DataFrame, col: str, default: float = 0.0) -> float:
    if col not in df.columns:
        return default
    val = df[col].iloc[-1]
    return float(val) if pd.notna(val) else default


def _entry_signal_event(symbol: str, df: pd.DataFrame, regime: MarketRegime, note: str) -> SignalEvent:
    return SignalEvent(
        ticker=symbol,
        signal_type=SignalType.BUY,
        asset_class=AssetClass.CRYPTO,
        composite_score=_last_float(df, "adx") / 100.0,  # ADX-normalized confidence proxy
        rsi=_last_float(df, "rsi", 50.0),
        momentum=_last_float(df, "momentum"),
        volume_ratio=_last_float(df, "volume_ratio", 1.0),
        trend_score=1.0,
        close_price=_last_float(df, "close"),
        atr=_last_float(df, "atr"),
        sma_50=_last_float(df, "sma_50"),
        sma_200=_last_float(df, "sma_200"),
        adx=_last_float(df, "adx"),
        notes=f"24h_entry regime={regime.value} {note}",
    )


def _exit_signal_event(symbol: str, df: pd.DataFrame, reason: str) -> SignalEvent:
    return SignalEvent(
        ticker=symbol,
        signal_type=SignalType.SELL,
        asset_class=AssetClass.CRYPTO,
        composite_score=0.0,
        rsi=_last_float(df, "rsi", 50.0),
        momentum=_last_float(df, "momentum"),
        volume_ratio=_last_float(df, "volume_ratio", 1.0),
        trend_score=0.0,
        close_price=_last_float(df, "close"),
        atr=_last_float(df, "atr"),
        sma_50=_last_float(df, "sma_50"),
        sma_200=_last_float(df, "sma_200"),
        adx=_last_float(df, "adx"),
        notes=f"24h_exit {reason}",
    )


# ---------------------------------------------------------------------------
# Entry / exit generation
# ---------------------------------------------------------------------------

def _position_symbol(pos: dict) -> str:
    return pos.get("symbol") or pos.get("ticker") or ""


def generate_24h_entries(
    universe: dict[str, pd.DataFrame],
    funding_rates: dict[str, float],
    open_positions: list[dict],
    equity: float,
    available_cash: float,
    btc_only: bool = True,
    max_positions: int = MAX_CRYPTO_POSITIONS,
    win_rate: float = DEFAULT_WIN_RATE,
    win_loss_ratio: float = DEFAULT_WIN_LOSS_RATIO,
    risk_manager: Optional[RiskManager] = None,
) -> list[CryptoEntryPlan]:
    """Evaluate every symbol and return sizeable, gated long entry plans."""
    held = {_position_symbol(p) for p in open_positions}
    open_count = len(held)
    plans: list[CryptoEntryPlan] = []

    # Portfolio-level kill switch: no new entries while the monthly drawdown halt is on.
    if risk_manager is not None and risk_manager.is_halted():
        logger.warning("Crypto entries suppressed: monthly drawdown halt active")
        return plans

    # Shared heat reading (capital-at-risk across open crypto positions / equity).
    current_heat = 0.0
    if risk_manager is not None:
        current_heat = risk_manager.portfolio_heat(AssetType.CRYPTO, equity=equity)

    for symbol, df in universe.items():
        if open_count + len(plans) >= max_positions:
            logger.info("Max crypto positions reached (%d) - no more entries", max_positions)
            break
        if symbol in held:
            continue
        if btc_only and symbol != BTC_SYMBOL:
            logger.debug("btc_only gate: skipping %s", symbol)
            continue
        if df is None or len(df) < MIN_BARS_24H:
            logger.debug("%s: insufficient bars for 24h entry", symbol)
            continue

        if not above_ema200(df):
            continue
        regime = classify_regime(df)
        if regime != MarketRegime.TRENDING:
            logger.debug("%s: regime %s not tradable for trend entry", symbol, regime.value)
            continue
        if not bullish_ema_cross(df):
            continue
        if not macd_confirms_long(df):
            continue
        if _last_float(df, "volume_ratio", 0.0) < VOLUME_SPIKE_MIN:
            continue

        entry_price = _last_float(df, "close")
        atr = _last_float(df, "atr")
        funding = float(funding_rates.get(symbol, 0.0))

        sizing = compute_position_size(
            equity=equity,
            available_cash=available_cash,
            entry_price=entry_price,
            atr=atr,
            funding_rate=funding,
            win_rate=win_rate,
            win_loss_ratio=win_loss_ratio,
            current_heat=current_heat,
        )
        if not sizing.tradable:
            logger.info("%s: entry gated by sizing (%s)", symbol, sizing.reason)
            continue

        plans.append(CryptoEntryPlan(
            symbol=symbol,
            entry_price=entry_price,
            units=sizing.units,
            notional=sizing.notional,
            stop_price=sizing.stop_price,
            take_profit=sizing.take_profit,
            effective_risk_pct=sizing.effective_risk_pct,
            regime=regime.value,
            funding_rate=funding,
            signal=_entry_signal_event(symbol, df, regime, f"funding={funding:.5f}"),
            reason="all_gates_passed",
        ))
        logger.info(
            "ENTRY %s units=%.6f notional=$%.2f stop=%.2f tp=%.2f risk=%.3f%% funding=%.5f",
            symbol, sizing.units, sizing.notional, sizing.stop_price, sizing.take_profit,
            sizing.effective_risk_pct * 100, funding,
        )

    return plans


def generate_24h_exits(
    open_positions: list[dict],
    universe: dict[str, pd.DataFrame],
    atr_stop_mult: float = ATR_STOP_MULT,
) -> list[CryptoExitPlan]:
    """Return exit plans for held positions whose trend has broken down."""
    plans: list[CryptoExitPlan] = []

    for pos in open_positions:
        symbol = _position_symbol(pos)
        df = universe.get(symbol)
        if df is None or df.empty:
            plans.append(CryptoExitPlan(
                symbol=symbol, close_price=0.0, reason="exit_no_data",
                signal=_exit_signal_event(symbol, pd.DataFrame(), "no_data"),
            ))
            continue

        close = _last_float(df, "close")
        atr = _last_float(df, "atr")
        entry_price = float(pos.get("entry_price", 0.0))
        stop_loss = float(pos.get("stop_loss", 0.0))

        reason: Optional[str] = None
        if not above_ema200(df):
            reason = "below_ema200"
        elif bearish_ema_cross(df):
            reason = "ema_bearish_cross"
        elif "macd_hist" in df.columns and pd.notna(df["macd_hist"].iloc[-1]) and df["macd_hist"].iloc[-1] < 0:
            reason = "macd_hist_negative"
        else:
            # ATR trailing stop: prefer the stored stop_loss, else derive from entry.
            stop = stop_loss if stop_loss > 0 else (entry_price - atr_stop_mult * atr if entry_price > 0 and atr > 0 else 0.0)
            if stop > 0 and close > 0 and close < stop:
                reason = f"trailing_stop_breach close={close:.2f} stop={stop:.2f}"

        if reason:
            plans.append(CryptoExitPlan(
                symbol=symbol, close_price=close, reason=reason,
                signal=_exit_signal_event(symbol, df, reason),
            ))
            logger.info("EXIT %s reason=%s", symbol, reason)

    return plans


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_crypto_24h_pipeline(
    universe: dict[str, pd.DataFrame],
    funding_rates: Optional[dict[str, float]] = None,
    open_positions: Optional[list[dict]] = None,
    equity: float = 0.0,
    available_cash: float = 0.0,
    btc_only: bool = True,
    persist: bool = True,
    win_rate: float = DEFAULT_WIN_RATE,
    win_loss_ratio: float = DEFAULT_WIN_LOSS_RATIO,
    risk_manager: Optional[RiskManager] = None,
) -> CryptoCycleResult:
    """
    Full 24/7 crypto decision cycle. Exits are evaluated first (free up slots),
    then entries against the remaining capacity. All signal events are persisted
    once via persist_signals; entry/exit plans are returned for the executor.

    Pass a RiskManager to enforce the monthly drawdown halt and shared portfolio
    heat; omit it (default) for the Phase 4 standalone behavior.
    """
    funding_rates = funding_rates or {}
    open_positions = open_positions or []

    exits = generate_24h_exits(open_positions, universe)
    exiting_symbols = {e.symbol for e in exits}
    remaining_positions = [p for p in open_positions if _position_symbol(p) not in exiting_symbols]

    entries = generate_24h_entries(
        universe=universe,
        funding_rates=funding_rates,
        open_positions=remaining_positions,
        equity=equity,
        available_cash=available_cash,
        btc_only=btc_only,
        win_rate=win_rate,
        win_loss_ratio=win_loss_ratio,
        risk_manager=risk_manager,
    )

    persisted = 0
    if persist:
        events = [e.signal for e in exits] + [p.signal for p in entries]
        persisted = persist_signals(events)

    return CryptoCycleResult(entries=entries, exits=exits, persisted=persisted)
'@
Set-Content -Path "strategies\crypto_24h.py" -Value $fCrypto24h -Encoding ascii
Write-Host "wrote strategies\crypto_24h.py"

$fTestP5 = @'
"""
tests/test_phase5.py
Phase 5 - Risk management. Fully offline: synthetic frames, no network.

DB-backed tests run the REAL schema (database/db.py) on an in-memory SQLite database
via a StaticPool, so the actual Enum bindings and persistence paths are exercised.
"""

from __future__ import annotations

import os

# Dummy env so config.settings (required fields) imports cleanly before any project import.
for _k, _v in {
    "ALPACA_API_KEY": "test", "ALPACA_SECRET_KEY": "test",
    "POLYGON_API_KEY": "test", "BINANCE_API_KEY": "test", "BINANCE_SECRET_KEY": "test",
    "DB_PASSWORD": "test", "TELEGRAM_BOT_TOKEN": "test", "TELEGRAM_CHAT_ID": "test",
    "STARTING_CAPITAL": "10000",
}.items():
    os.environ.setdefault(_k, _v)

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Bind the real schema to a shared in-memory SQLite engine BEFORE importing modules
# that capture SessionLocal / get_db.
import database.db as dbmod

_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
_TestSession = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
dbmod.engine = _engine
dbmod.SessionLocal = _TestSession
dbmod.Base.metadata.create_all(_engine)

import strategies.signals as signals_mod
signals_mod.SessionLocal = _TestSession  # persist_signals binds SessionLocal by name

from database.db import (
    AssetType, MarketRegime, OrderSide, OrderStatus, Position, Signal, Trade,
)
from risk.manager import (
    RiskManager, half_kelly_fraction, size_position, stock_params, crypto_params,
    check_group_exposure,
)
from execution.alpaca import StockExecutor
from strategies import crypto_24h, stock_weekly

UTC = timezone.utc


@pytest.fixture(autouse=True)
def clean_db():
    """Wipe mutable tables before each test for isolation."""
    with dbmod.get_db() as db:
        db.query(Position).delete()
        db.query(Trade).delete()
        db.query(Signal).delete()
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_position(symbol, asset_type, entry, stop, qty, current=None):
    with dbmod.get_db() as db:
        db.add(Position(
            asset_type=asset_type, symbol=symbol, quantity=qty,
            entry_price=entry, current_price=current if current is not None else entry,
            stop_loss=stop, take_profit=entry * 1.1,
            opened_at=datetime.now(UTC),
        ))


def _add_closed_trade(symbol, asset_type, pnl, closed_at):
    with dbmod.get_db() as db:
        db.add(Trade(
            asset_type=asset_type, symbol=symbol, side=OrderSide.SELL,
            quantity=1.0, entry_price=100.0, exit_price=100.0 + pnl,
            status=OrderStatus.FILLED, pnl=pnl, pnl_pct=pnl / 100.0,
            opened_at=closed_at - timedelta(days=1), closed_at=closed_at,
        ))


def _stock_feature_df(close, rsi=50.0, vol=1.2, atr=2.0, above50=1):
    return pd.DataFrame({
        "close": [close], "rsi": [rsi], "volume_ratio": [vol], "atr": [atr],
        "above_sma50": [above50], "momentum": [0.05], "sma_50": [close * 0.98],
        "sma_200": [close * 0.9],
    })


# ===========================================================================
# half-Kelly
# ===========================================================================

def test_half_kelly_positive_edge():
    # f* = 0.5 - 0.5/1.5 = 0.16667; half = 0.08333
    assert half_kelly_fraction(0.5, 1.5) == pytest.approx(0.083333, abs=1e-5)


def test_half_kelly_negative_edge_floors_zero():
    assert half_kelly_fraction(0.2, 1.5) == 0.0


def test_half_kelly_bad_ratio():
    assert half_kelly_fraction(0.6, 0.0) == 0.0


def test_crypto_half_kelly_delegates_to_engine():
    assert crypto_24h.half_kelly_fraction(0.5, 1.5) == half_kelly_fraction(0.5, 1.5)


# ===========================================================================
# size_position - generic engine
# ===========================================================================

def test_size_stock_basic():
    ps = size_position(
        stock_params(), equity=10000, available_cash=10000,
        entry_price=100.0, stop_price=95.0, tp_r_mult=2.0,
    )
    assert ps.tradable
    assert ps.effective_risk_pct == pytest.approx(0.02)
    assert ps.units == pytest.approx(40.0, abs=1e-6)   # 200 risk / 5 stop
    assert ps.stop_price == pytest.approx(95.0)
    assert ps.take_profit == pytest.approx(110.0)


def test_size_heat_clamp():
    ps = size_position(
        stock_params(), equity=10000, available_cash=10000,
        entry_price=100.0, stop_price=95.0, tp_r_mult=2.0, current_heat=0.095,
    )
    # room = 0.10 - 0.095 = 0.005
    assert ps.effective_risk_pct == pytest.approx(0.005)
    assert ps.units == pytest.approx(10.0, abs=1e-6)


def test_size_heat_exceeded():
    ps = size_position(
        stock_params(), equity=10000, available_cash=10000,
        entry_price=100.0, stop_price=95.0, current_heat=0.10,
    )
    assert not ps.tradable
    assert ps.reason == "heat_exceeded"


def test_size_min_floor():
    ps = size_position(
        stock_params(), equity=10000, available_cash=30.0,
        entry_price=100.0, stop_price=95.0,
    )
    assert not ps.tradable
    assert ps.reason == "below_min_position"


def test_size_regime_high_vol_halves():
    full = size_position(stock_params(), 10000, 10000, 100.0, stop_price=95.0,
                         regime=MarketRegime.TRENDING)
    hv = size_position(stock_params(), 10000, 10000, 100.0, stop_price=95.0,
                       regime=MarketRegime.HIGH_VOL)
    assert hv.effective_risk_pct == pytest.approx(full.effective_risk_pct * 0.5)


def test_size_invalid_inputs():
    assert not size_position(stock_params(), 0, 1000, 100.0, stop_price=95.0).tradable
    assert not size_position(stock_params(), 1000, 1000, 0.0, stop_price=95.0).tradable
    assert size_position(stock_params(), 1000, 1000, 100.0).reason == "invalid_inputs"  # no stop/atr


def test_size_cash_ceiling_no_margin():
    ps = size_position(
        crypto_params(), equity=100000, available_cash=500.0,
        entry_price=100.0, atr=1.0,
    )
    # huge equity would size large, but cash caps notional at <= 500
    assert ps.notional <= 500.0 + 1e-6


# ===========================================================================
# crypto compute_position_size - integration preserves Phase 4 math
# ===========================================================================

def test_crypto_sizing_concentration_cap():
    s = crypto_24h.compute_position_size(
        equity=10000, available_cash=10000, entry_price=100.0, atr=1.0,
    )
    # 1.5% of 10k = 150 risk / 2 stop = 75 units -> $7500, capped at 35% = $3500 -> 35 units
    assert s.units == pytest.approx(35.0, abs=1e-6)
    assert s.notional == pytest.approx(3500.0, abs=1e-3)
    assert s.stop_price == pytest.approx(98.0)
    assert s.take_profit == pytest.approx(106.0)
    assert s.reason == "ok"


def test_crypto_sizing_uncapped():
    s = crypto_24h.compute_position_size(
        equity=1000, available_cash=1000, entry_price=100.0, atr=5.0,
    )
    assert s.units == pytest.approx(1.5, abs=1e-6)
    assert s.notional == pytest.approx(150.0, abs=1e-6)


def test_crypto_funding_skip():
    s = crypto_24h.compute_position_size(10000, 10000, 100.0, 1.0, funding_rate=-0.002)
    assert s.units == 0.0
    assert s.reason.startswith("funding_too_negative")


def test_crypto_funding_haircut():
    s = crypto_24h.compute_position_size(1000, 1000, 100.0, 5.0, funding_rate=-0.0005)
    # risk halved: 0.0075 * 1000 = 7.5 / 10 = 0.75 units
    assert s.units == pytest.approx(0.75, abs=1e-6)


def test_crypto_no_edge():
    s = crypto_24h.compute_position_size(1000, 1000, 100.0, 5.0, win_rate=0.2)
    assert s.units == 0.0
    assert s.reason == "no_edge_zero_risk"


def test_crypto_below_min():
    s = crypto_24h.compute_position_size(1000, 40.0, 100.0, 5.0)
    assert s.units == 0.0
    assert s.reason.startswith("below_min_position")


# ===========================================================================
# RiskManager - NAV / drawdown / halt
# ===========================================================================

def test_monthly_drawdown_triggers_halt():
    now = datetime.now(UTC)
    month_start = datetime(now.year, now.month, 1, tzinfo=UTC)
    _add_closed_trade("AAA", AssetType.STOCK, +500.0, month_start - timedelta(days=1))
    _add_closed_trade("BBB", AssetType.STOCK, -2000.0, now)
    rm = RiskManager(starting_capital=10000)
    # baseline = 10500, current = 8500 -> -19.05%
    assert rm.month_start_nav(now) == pytest.approx(10500.0)
    assert rm.current_nav() == pytest.approx(8500.0)
    assert rm.monthly_drawdown(now) == pytest.approx(-2000 / 10500, abs=1e-4)
    assert rm.is_halted(now) is True


def test_monthly_drawdown_within_limit():
    now = datetime.now(UTC)
    month_start = datetime(now.year, now.month, 1, tzinfo=UTC)
    _add_closed_trade("AAA", AssetType.STOCK, +500.0, month_start - timedelta(days=1))
    _add_closed_trade("BBB", AssetType.STOCK, -1000.0, now)
    rm = RiskManager(starting_capital=10000)
    assert rm.is_halted(now) is False


def test_portfolio_heat():
    _add_position("AAA", AssetType.STOCK, entry=100.0, stop=95.0, qty=10)   # risk 50
    _add_position("BBB", AssetType.STOCK, entry=50.0, stop=47.5, qty=20)    # risk 50
    rm = RiskManager(starting_capital=10000)
    assert rm.portfolio_heat(AssetType.STOCK, equity=10000) == pytest.approx(0.01)


def test_pre_trade_gate_blocks_on_halt():
    now = datetime.now(UTC)
    month_start = datetime(now.year, now.month, 1, tzinfo=UTC)
    _add_closed_trade("AAA", AssetType.STOCK, +500.0, month_start - timedelta(days=1))
    _add_closed_trade("BBB", AssetType.STOCK, -2000.0, now)
    rm = RiskManager(starting_capital=10000)
    ok, reason = rm.pre_trade_ok(stock_params(), "MSFT", now=now)
    assert ok is False and reason == "drawdown_halt"


def test_pre_trade_gate_blocks_duplicate():
    _add_position("MSFT", AssetType.STOCK, entry=100.0, stop=95.0, qty=5)
    rm = RiskManager(starting_capital=10000)
    ok, reason = rm.pre_trade_ok(stock_params(), "MSFT")
    assert ok is False and reason == "already_open"


# ===========================================================================
# Sector / correlation guard
# ===========================================================================

def test_group_exposure_guard():
    smap = {"AAA": "tech", "BBB": "tech", "CCC": "energy"}
    ok, _ = check_group_exposure(["AAA"], "BBB", smap, max_per_group=2)
    assert ok is True
    blocked, reason = check_group_exposure(["AAA", "BBB"], "CCC", {"AAA": "tech", "BBB": "tech", "CCC": "tech"}, 2)
    assert blocked is False and reason.startswith("group_full")


def test_group_exposure_noop_without_map():
    ok, reason = check_group_exposure(["AAA"], "BBB", None, 2)
    assert ok is True and reason == "no_group_data"


# ===========================================================================
# StockExecutor (paper) on real schema
# ===========================================================================

def test_stock_executor_open_and_close():
    ex = StockExecutor(paper=True, paper_cash=10000, max_positions=8)
    res = ex.open_long("AAPL", units=10, entry_price=100.0, stop_loss=95.0, take_profit=110.0, week_number=22)
    assert res["status"] == "filled"
    assert ex.count_open_stock_positions() == 1

    close = ex.close_long("AAPL", exit_price=110.0, reason="weekly_exit")
    assert close["status"] == "filled"
    assert ex.count_open_stock_positions() == 0
    # bought ~100.05 (slippage), sold ~109.945 -> positive pnl
    assert close["pnl"] > 0


def test_stock_executor_backstops():
    ex = StockExecutor(paper=True, paper_cash=10000, max_positions=1)
    assert ex.open_long("AAPL", 10, 100.0, 95.0, 110.0)["status"] == "filled"
    # duplicate
    assert ex.open_long("AAPL", 10, 100.0, 95.0, 110.0)["reason"] == "already_open"
    # max positions
    assert ex.open_long("MSFT", 10, 100.0, 95.0, 110.0)["reason"] == "max_positions"
    # below min notional
    ex2 = StockExecutor(paper=True, paper_cash=10000, max_positions=8)
    assert ex2.open_long("NVDA", 0.1, 100.0, 95.0, 110.0)["reason"] == "below_min_notional"


# ===========================================================================
# stock_weekly pipeline
# ===========================================================================

def _scored_and_universe():
    scored = pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "composite": [0.80, 0.70],
        "trend_score": [0.9, 0.8],
        "rsi_score": [1.0, 1.0],
        "momentum_score": [0.6, 0.5],
        "volume_score": [0.6, 0.6],
        "volatility_score": [0.5, 0.5],
    })
    universe = {
        "AAA": _stock_feature_df(close=100.0, rsi=55.0),
        "BBB": _stock_feature_df(close=50.0, rsi=50.0),
    }
    return scored, universe


def test_stock_weekly_entries_sized_and_gated():
    scored, universe = _scored_and_universe()
    rm = RiskManager(starting_capital=10000)
    plans = stock_weekly.generate_week_entries(
        scored_df=scored, universe=universe,
        equity=10000, available_cash=10000, risk_manager=rm,
    )
    assert len(plans) == 2
    for p in plans:
        assert p.units > 0
        assert p.notional >= 50.0
        assert p.stop_price == pytest.approx(p.entry_price * 0.95)


def test_stock_weekly_buys_execute_and_persist():
    scored, universe = _scored_and_universe()
    rm = RiskManager(starting_capital=10000)
    ex = StockExecutor(paper=True, paper_cash=10000, max_positions=8)
    result = stock_weekly.run_stock_weekly_buys(
        scored_df=scored, universe=universe,
        equity=10000, available_cash=10000, executor=ex, risk_manager=rm,
    )
    assert len(result.entries) == 2
    assert all(r["status"] == "filled" for r in result.executed)
    assert result.persisted == 2
    assert ex.count_open_stock_positions() == 2


def test_stock_weekly_sells_close_all():
    ex = StockExecutor(paper=True, paper_cash=10000, max_positions=8)
    ex.open_long("AAA", 10, 100.0, 95.0, 110.0, week_number=22)
    open_positions = [{"ticker": "AAA", "entry_price": 100.0}]
    universe = {"AAA": _stock_feature_df(close=105.0)}
    result = stock_weekly.run_stock_weekly_sells(open_positions, universe, executor=ex)
    assert len(result.exits) == 1
    assert result.executed[0]["status"] == "filled"
    assert ex.count_open_stock_positions() == 0


def test_stock_weekly_halt_suppresses_entries():
    now = datetime.now(UTC)
    month_start = datetime(now.year, now.month, 1, tzinfo=UTC)
    _add_closed_trade("X", AssetType.STOCK, +500.0, month_start - timedelta(days=1))
    _add_closed_trade("Y", AssetType.STOCK, -2000.0, now)
    rm = RiskManager(starting_capital=10000)
    scored, universe = _scored_and_universe()
    plans = stock_weekly.generate_week_entries(
        scored_df=scored, universe=universe, equity=10000, available_cash=10000, risk_manager=rm,
    )
    assert plans == []


# ===========================================================================
# crypto pipeline honors the shared halt
# ===========================================================================

def test_crypto_pipeline_halt_suppresses_entries():
    now = datetime.now(UTC)
    month_start = datetime(now.year, now.month, 1, tzinfo=UTC)
    _add_closed_trade("X", AssetType.CRYPTO, +500.0, month_start - timedelta(days=1))
    _add_closed_trade("Y", AssetType.CRYPTO, -2000.0, now)
    rm = RiskManager(starting_capital=10000)
    universe = {"BTC/USDT": _stock_feature_df(close=60000.0)}
    result = crypto_24h.run_crypto_24h_pipeline(
        universe=universe, equity=10000, available_cash=10000,
        risk_manager=rm, persist=False,
    )
    assert result.entries == []


def test_persist_signals_writes_real_schema():
    from strategies.signals import SignalEvent, SignalType, AssetClass, persist_signals
    ev = SignalEvent(
        ticker="AAA", signal_type=SignalType.BUY, asset_class=AssetClass.STOCK,
        composite_score=0.8, rsi=55.0, momentum=0.05, volume_ratio=1.2,
        trend_score=0.9, close_price=100.0, atr=2.0,
    )
    n = persist_signals([ev])
    assert n == 1
    with dbmod.get_db() as db:
        row = db.query(Signal).filter_by(symbol="AAA").first()
        assert row is not None
        assert row.direction == "long"
        assert row.asset_type == "stock"
'@
Set-Content -Path "tests\test_phase5.py" -Value $fTestP5 -Encoding ascii
Write-Host "wrote tests\test_phase5.py"

Write-Host "Phase 5 files written. Run: python -m pytest tests/test_phase5.py -q"
