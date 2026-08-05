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

FILL_POLL_ATTEMPTS = 12
FILL_POLL_INTERVAL_S = 0.5


def await_order_fill(client, order, attempts: int = FILL_POLL_ATTEMPTS,
                     interval_s: float = FILL_POLL_INTERVAL_S):
    """Poll an Alpaca order until it reports a fill.

    submit_order returns immediately with status 'accepted'/'new' and
    filled_qty=0; trusting that snapshot recorded qty-0 positions. Market
    orders during trading hours fill in well under the ~6s polled here. If the
    order still isn't filled (e.g. submitted after hours, queued to next
    open), the caller falls back to the submitted qty at the reference price.
    """
    import time

    order_id = getattr(order, "id", None)
    if order_id is None or client is None:
        return order
    for _ in range(attempts):
        filled_qty = float(getattr(order, "filled_qty", 0) or 0)
        status = str(getattr(order, "status", "")).lower()
        if filled_qty > 0 and ("filled" in status or "partially" in status):
            return order
        time.sleep(interval_s)
        try:
            order = client.get_order_by_id(order_id)
        except Exception as exc:
            logger.warning("await_order_fill: get_order_by_id failed: %s", exc)
            return order
    return order


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
        # Backstop floor; the scheduler tightens this per-cycle from account
        # equity via risk.manager.effective_min_position.
        self.min_position_usd = float(
            getattr(settings, "min_position_size", MIN_POSITION_USD))

    # -- client bootstrap (live only) --------------------------------------

    @classmethod
    def from_settings(cls, paper: Optional[bool] = None) -> "StockExecutor":
        """
        Build a StockExecutor from config, honoring settings.execution_mode:
        "sim" simulates fills locally (no SDK needed); "paper" routes real
        orders to the Alpaca paper API; "live" routes to the live API.
        """
        if paper is None:
            is_sim = settings.execution_mode == "sim"
        else:
            is_sim = paper
        client = None
        if not is_sim:
            try:
                from alpaca.trading.client import TradingClient  # type: ignore
                client = TradingClient(
                    settings.alpaca_api_key,
                    settings.alpaca_secret_key,
                    paper=settings.alpaca_paper,
                )
            except Exception as exc:
                logger.error("Alpaca client init failed, falling back to sim: %s", exc)
                is_sim = True
        return cls(paper=is_sim, client=client)

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

        order = await_order_fill(self.client, order)
        price = float(getattr(order, "filled_avg_price", None) or ref_price)
        filled = float(getattr(order, "filled_qty", 0) or 0)
        if filled <= 0:
            # Not filled within the poll window (e.g. after-hours queue). The
            # order is live at the broker; record the submitted size at the
            # reference price rather than a qty-0 phantom.
            logger.warning("order %s %s not filled within poll window; recording submitted qty",
                           side, symbol)
            filled = units
            price = ref_price
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
        if notional < self.min_position_usd:
            logger.warning("open_long %s rejected: notional $%.2f < min $%.2f", symbol, notional, self.min_position_usd)
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
