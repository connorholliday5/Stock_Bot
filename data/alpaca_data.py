"""
data/alpaca_data.py
Market data via Alpaca's data API, so the whole bot runs with ONLY the two
Alpaca keys (Polygon and Binance stay optional upgrades).

  - Stocks: StockHistoricalDataClient on the free IEX feed. Daily bars for the
    scan universe are fetched in a handful of multi-symbol requests, far below
    Alpaca's ~200 req/min budget.
  - Crypto: CryptoHistoricalDataClient - keyless, 24/7, native "BTC/USD"
    symbols that match the AlpacaCryptoExecutor order path.

Both fetchers return {symbol: feature-engineered DataFrame} in exactly the
shape strategies expect (data.fetcher.add_features on lowercase OHLCV), so
they are drop-in replacements for the Polygon / ccxt paths.

alpaca-py is imported lazily; offline tests never touch it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from loguru import logger

from config import settings
from data.fetcher import add_features

UTC = timezone.utc

def default_stock_universe() -> list[str]:
    """Scan universe for the Alpaca (no-Polygon) path: the STOCK_UNIVERSE env
    override when set (comma-separated), else the full built-in S&P 500 list."""
    override = getattr(settings, "stock_universe", "") or ""
    tickers = [t.strip().upper() for t in override.split(",") if t.strip()]
    if tickers:
        return tickers
    from data.sp500 import SP500_TICKERS
    return list(SP500_TICKERS)

DEFAULT_CRYPTO_UNIVERSE = ["BTC/USD", "ETH/USD", "SOL/USD", "LTC/USD"]

_BATCH = 100          # symbols per multi-symbol bars request


def _timeframe(tf: str):
    """'4h' / '1d' / '15m' -> alpaca TimeFrame."""
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    tf = tf.strip().lower()
    amount = int("".join(c for c in tf if c.isdigit()) or 1)
    if tf.endswith("m"):
        return TimeFrame(amount, TimeFrameUnit.Minute)
    if tf.endswith("h"):
        return TimeFrame(amount, TimeFrameUnit.Hour)
    if tf.endswith("w"):
        return TimeFrame(amount, TimeFrameUnit.Week)
    return TimeFrame(amount, TimeFrameUnit.Day)


def _per_symbol_frames(bars_df: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Split alpaca's multi-index bars frame into feature-engineered frames."""
    out: dict[str, pd.DataFrame] = {}
    if bars_df is None or bars_df.empty:
        return out
    for symbol in symbols:
        try:
            if isinstance(bars_df.index, pd.MultiIndex):
                if symbol not in bars_df.index.get_level_values(0):
                    continue
                df = bars_df.xs(symbol, level=0).copy()
            else:
                df = bars_df.copy()
            df = df.rename(columns=str.lower)
            keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
            if len(keep) < 5 or df.empty:
                continue
            df = df[keep].sort_index()
            out[symbol] = add_features(df)
        except Exception as exc:
            logger.warning("alpaca_data: skipping {} ({})", symbol, exc)
    return out


def fetch_stock_universe_alpaca(
    lookback_days: int = 400,
    tickers: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    """Daily-bar feature universe from Alpaca IEX data. {} on total failure."""
    tickers = list(tickers or default_stock_universe())
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest

        client = StockHistoricalDataClient(
            settings.alpaca_api_key, settings.alpaca_secret_key
        )
        start = datetime.now(UTC) - timedelta(days=lookback_days)
        universe: dict[str, pd.DataFrame] = {}
        for i in range(0, len(tickers), _BATCH):
            chunk = tickers[i:i + _BATCH]
            req = StockBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=_timeframe("1d"),
                start=start,
                feed="iex",          # free plan; SIP needs a data subscription
            )
            bars = client.get_stock_bars(req)
            universe.update(_per_symbol_frames(getattr(bars, "df", None), chunk))
        logger.info("alpaca_data: stock universe fetched {}/{} symbols",
                    len(universe), len(tickers))
        return universe
    except Exception as exc:
        logger.warning("alpaca_data: stock universe fetch failed ({})", exc)
        return {}


def fetch_crypto_universe_alpaca(
    symbols: Optional[list[str]] = None,
    timeframe: str = "4h",
    limit: int = 300,
) -> dict[str, pd.DataFrame]:
    """4H (default) crypto feature universe from Alpaca. Keyless endpoint."""
    symbols = list(symbols or DEFAULT_CRYPTO_UNIVERSE)
    try:
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest

        client = CryptoHistoricalDataClient()
        tf = _timeframe(timeframe)
        hours_per_bar = {"m": 1 / 60, "h": 1, "d": 24, "w": 24 * 7}[timeframe.strip().lower()[-1]]
        amount = int("".join(c for c in timeframe if c.isdigit()) or 1)
        start = datetime.now(UTC) - timedelta(hours=limit * amount * hours_per_bar)
        req = CryptoBarsRequest(symbol_or_symbols=symbols, timeframe=tf, start=start)
        bars = client.get_crypto_bars(req)
        universe = _per_symbol_frames(getattr(bars, "df", None), symbols)
        logger.info("alpaca_data: crypto universe fetched {}/{} symbols",
                    len(universe), len(symbols))
        return universe
    except Exception as exc:
        logger.warning("alpaca_data: crypto universe fetch failed ({})", exc)
        return {}


def fetch_latest_crypto_price_alpaca(symbol: str) -> Optional[float]:
    """Latest trade price for one pair; None on failure."""
    prices = fetch_latest_crypto_prices_alpaca([symbol])
    return prices.get(symbol)


def fetch_latest_crypto_prices_alpaca(symbols: list[str]) -> dict[str, float]:
    """Latest trade prices for many pairs in ONE keyless request - cheap
    enough for the 15-minute crypto stop monitor. {} on failure."""
    symbols = [s for s in symbols if s]
    if not symbols:
        return {}
    try:
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests import CryptoLatestTradeRequest

        client = CryptoHistoricalDataClient()
        req = CryptoLatestTradeRequest(symbol_or_symbols=symbols)
        trades = client.get_crypto_latest_trade(req)
        out: dict[str, float] = {}
        for sym in symbols:
            trade = trades.get(sym)
            price = float(getattr(trade, "price", 0.0) or 0.0) if trade else 0.0
            if price > 0:
                out[sym] = price
        return out
    except Exception as exc:
        logger.warning("alpaca_data: latest price fetch failed for {} ({})", symbols, exc)
        return {}
