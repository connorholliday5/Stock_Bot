"""
execution/ccxt_crypto.py
Phase 4 - Crypto order execution via CCXT (Binance spot).

Long-only spot. Defaults to PAPER mode (no real orders) for Phase 1 paper trading;
set paper=False to route real market orders through CCXT. Position and Trade rows
are written in both modes so position state (and the max-position backstop) is real.

This module executes; it does not decide. The strategy layer (crypto_24h) sizes and
gates; the scheduler hands primitives here.
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

logger = logging.getLogger(__name__)

UTC = timezone.utc

DEFAULT_FEE_RATE = 0.001        # 0.1% taker, per side
MAX_CRYPTO_POSITIONS = 3
MIN_POSITION_USD = 50.0


class CryptoExecutor:
    """Places (or simulates) Binance spot market orders and records them in the DB."""

    def __init__(
        self,
        paper: bool = True,
        exchange=None,
        paper_cash: float = 0.0,
        fee_rate: float = DEFAULT_FEE_RATE,
        max_positions: int = MAX_CRYPTO_POSITIONS,
    ) -> None:
        self.paper = paper
        self.exchange = exchange          # a ccxt exchange (e.g. CryptoFetcher().exchange) for live
        self.paper_cash = paper_cash
        self.fee_rate = fee_rate
        self.max_positions = max_positions

    # -- balances / state ---------------------------------------------------

    def get_cash(self, asset: str = "USDT") -> float:
        if self.paper:
            return self.paper_cash
        if self.exchange is None:
            logger.error("Live mode but no exchange configured")
            return 0.0
        try:
            bal = self.exchange.fetch_balance()
            return float(bal.get("free", {}).get(asset, 0.0))
        except Exception as exc:
            logger.error("fetch_balance failed: %s", exc)
            return 0.0

    @staticmethod
    def count_open_crypto_positions() -> int:
        with get_db() as db:
            return db.query(Position).filter(Position.asset_type == AssetType.CRYPTO).count()

    @staticmethod
    def open_crypto_symbols() -> list[str]:
        with get_db() as db:
            rows = db.query(Position.symbol).filter(Position.asset_type == AssetType.CRYPTO).all()
            return [r[0] for r in rows]

    # -- order placement ----------------------------------------------------

    def _place_market_order(self, symbol: str, side: str, units: float, ref_price: float) -> dict:
        """
        Return a normalized fill: {price, units, fee, status}.
        Paper mode fills at ref_price; live mode routes through CCXT.
        """
        if units <= 0:
            return {"price": 0.0, "units": 0.0, "fee": 0.0, "status": "rejected"}

        if self.paper:
            fee = ref_price * units * self.fee_rate
            return {"price": ref_price, "units": units, "fee": fee, "status": "filled"}

        if self.exchange is None:
            logger.error("Live order requested but no exchange configured")
            return {"price": 0.0, "units": 0.0, "fee": 0.0, "status": "rejected"}

        try:
            order = self.exchange.create_order(symbol, "market", side, units)
        except Exception as exc:
            logger.error("create_order failed for %s %s: %s", side, symbol, exc)
            return {"price": 0.0, "units": 0.0, "fee": 0.0, "status": "rejected"}

        price = float(order.get("average") or order.get("price") or ref_price)
        filled = float(order.get("filled") or units)
        fee_info = order.get("fee") or {}
        fee = float(fee_info.get("cost", price * filled * self.fee_rate))
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
        """Open a long spot position. Enforces max-position and min-notional backstops."""
        notional = units * entry_price
        if notional < MIN_POSITION_USD:
            logger.warning("open_long %s rejected: notional $%.2f < min $%.2f", symbol, notional, MIN_POSITION_USD)
            return {"status": "rejected", "reason": "below_min_notional", "symbol": symbol}

        with get_db() as db:
            open_count = db.query(Position).filter(Position.asset_type == AssetType.CRYPTO).count()
            if open_count >= self.max_positions:
                logger.warning("open_long %s rejected: %d open >= max %d", symbol, open_count, self.max_positions)
                return {"status": "rejected", "reason": "max_positions", "symbol": symbol}

            if db.query(Position).filter_by(symbol=symbol).first():
                logger.warning("open_long %s rejected: position already open", symbol)
                return {"status": "rejected", "reason": "already_open", "symbol": symbol}

            fill = self._place_market_order(symbol, "buy", units, entry_price)
            if fill["status"] != "filled":
                return {"status": "rejected", "reason": "order_not_filled", "symbol": symbol}

            now = datetime.now(UTC)
            position = Position(
                asset_type=AssetType.CRYPTO,
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
                asset_type=AssetType.CRYPTO,
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
                asset_type=AssetType.CRYPTO,
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