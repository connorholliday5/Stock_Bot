"""
execution/alpaca_crypto.py
24/7 crypto execution through the SAME Alpaca account as the stock strategy,
so the money shown in the Alpaca dashboard funds both books and no second
exchange (or extra API budget) is needed.

AlpacaCryptoExecutor reuses CryptoExecutor's DB bookkeeping (Position/Trade
rows, max-position and min-notional backstops) and swaps only the order
transport: Alpaca crypto market orders (GTC - crypto trades around the clock)
instead of ccxt/Binance.

Symbol mapping: strategy/data symbols are Binance-style ("BTC/USDT" - public
Binance OHLCV remains the free, keyless data source). Orders translate to
Alpaca pairs ("BTC/USD"): USDT/BUSD quotes become USD.

Modes follow settings.execution_mode exactly like the stock executor:
  sim   - local simulated fills (default; no SDK, no orders)
  paper - real orders to Alpaca's paper API
  live  - real money
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from config import settings
from execution.ccxt_crypto import CryptoExecutor

# Alpaca crypto taker fee tier 0 (0.25%); overridable per instance.
ALPACA_CRYPTO_FEE_RATE = 0.0025

_QUOTE_MAP = {"USDT": "USD", "BUSD": "USD", "USDC": "USD"}


def to_alpaca_symbol(symbol: str) -> str:
    """'BTC/USDT' -> 'BTC/USD'; passthrough for anything already USD-quoted."""
    if "/" not in symbol:
        return symbol
    base, quote = symbol.split("/", 1)
    return f"{base}/{_QUOTE_MAP.get(quote.upper(), quote.upper())}"


class AlpacaCryptoExecutor(CryptoExecutor):
    """CryptoExecutor with Alpaca order routing instead of ccxt."""

    def __init__(
        self,
        paper: bool = True,
        client=None,
        paper_cash: float = 0.0,
        fee_rate: float = ALPACA_CRYPTO_FEE_RATE,
        max_positions: Optional[int] = None,
    ) -> None:
        super().__init__(
            paper=paper,
            exchange=None,
            paper_cash=paper_cash,
            fee_rate=fee_rate,
            max_positions=(max_positions if max_positions is not None
                           else int(getattr(settings, "max_crypto_positions", 3))),
        )
        self.client = client  # an alpaca TradingClient when not simulating

    @classmethod
    def from_settings(cls, paper: Optional[bool] = None) -> "AlpacaCryptoExecutor":
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
                logger.error("Alpaca crypto client init failed, falling back to sim: {}", exc)
                is_sim = True
        return cls(paper=is_sim, client=client)

    # -- balances -----------------------------------------------------------

    def get_cash(self, asset: str = "USD") -> float:
        if self.paper:
            return self.paper_cash
        try:
            from execution.account import broker_account
            snap = broker_account().snapshot()
            return float(snap.cash) if snap is not None else 0.0
        except Exception as exc:
            logger.error("AlpacaCryptoExecutor.get_cash failed: {}", exc)
            return 0.0

    # -- order transport ------------------------------------------------------

    def _place_market_order(self, symbol: str, side: str, units: float, ref_price: float) -> dict:
        if units <= 0:
            return {"price": 0.0, "units": 0.0, "fee": 0.0, "status": "rejected"}

        if self.paper:
            fee = ref_price * units * self.fee_rate
            return {"price": ref_price, "units": units, "fee": fee, "status": "filled"}

        if self.client is None:
            logger.error("Live crypto order requested but no Alpaca client configured")
            return {"price": 0.0, "units": 0.0, "fee": 0.0, "status": "rejected"}

        try:
            from alpaca.trading.requests import MarketOrderRequest  # type: ignore
            from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce  # type: ignore

            req = MarketOrderRequest(
                symbol=to_alpaca_symbol(symbol),
                qty=units,
                side=AlpacaSide.BUY if side == "buy" else AlpacaSide.SELL,
                time_in_force=TimeInForce.GTC,  # crypto: 24/7, GTC required
            )
            order = self.client.submit_order(req)
        except Exception as exc:
            logger.error("crypto submit_order failed for {} {}: {}", side, symbol, exc)
            return {"price": 0.0, "units": 0.0, "fee": 0.0, "status": "rejected"}

        from execution.alpaca import await_order_fill
        order = await_order_fill(self.client, order)
        price = float(getattr(order, "filled_avg_price", None) or ref_price)
        filled = float(getattr(order, "filled_qty", 0) or 0)
        if filled <= 0:
            logger.warning("crypto order {} {} not filled within poll window; "
                           "recording submitted qty", side, symbol)
            filled = units
            price = ref_price
        fee = price * filled * self.fee_rate
        return {"price": price, "units": filled, "fee": fee, "status": "filled"}


def crypto_executor_from_settings(paper: Optional[bool] = None):
    """Factory honoring CRYPTO_EXCHANGE: 'alpaca' (default) or 'binance'."""
    if getattr(settings, "crypto_exchange", "alpaca") == "binance":
        exchange = None
        is_sim = (settings.execution_mode == "sim") if paper is None else paper
        if not is_sim:
            try:
                from data.fetcher import CryptoFetcher
                exchange = CryptoFetcher().exchange
            except Exception as exc:
                logger.error("Binance client init failed, falling back to sim: {}", exc)
                is_sim = True
        return CryptoExecutor(paper=is_sim, exchange=exchange)
    return AlpacaCryptoExecutor.from_settings(paper=paper)
