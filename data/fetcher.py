"""
data/fetcher.py
Phase 2 - Data Pipeline (extended in Phase 4)
Polygon.io for S&P 500 historical + real-time data.
CCXT Binance for crypto OHLCV.
Feature engineering: RSI, ATR, VWAP, volume ratio, momentum, SMA 50/200,
and (Phase 4) EMA 9/21/200, MACD, ADX, realized volatility.
Funding-rate fetch added for the crypto 24/7 strategy.
Earnings date guard.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import ccxt
import numpy as np
import pandas as pd
import requests

from config.settings import get_settings

logger = logging.getLogger(__name__)

EST = ZoneInfo("America/New_York")
UTC = timezone.utc

settings = get_settings()


# ---------------------------------------------------------------------------
# Polygon client
# ---------------------------------------------------------------------------

class PolygonClient:
    """Thin wrapper around the Polygon REST API."""

    BASE = "https://api.polygon.io"

    def __init__(self) -> None:
        self.api_key: str = settings.polygon_api_key
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def _get(self, path: str, params: dict | None = None, retries: int = 3) -> dict:
        url = f"{self.BASE}{path}"
        params = params or {}
        for attempt in range(retries):
            try:
                resp = self.session.get(url, params=params, timeout=15)
                if resp.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning("Polygon rate limit hit - sleeping %ds", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                logger.error("Polygon request failed (attempt %d): %s", attempt + 1, exc)
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return {}

    def _paginate(self, path: str, params: dict | None = None) -> list[dict]:
        """Follow next_url pagination, return all results."""
        params = params or {}
        results: list[dict] = []
        url: str | None = f"{self.BASE}{path}"
        while url:
            try:
                resp = self.session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                results.extend(data.get("results", []))
                next_url = data.get("next_url")
                url = next_url if next_url else None
                params = {}
            except requests.RequestException as exc:
                logger.error("Polygon pagination error: %s", exc)
                break
        return results

    def get_sp500_tickers(self) -> list[str]:
        """
        Return S&P 500 constituent tickers using Polygon's index snapshot endpoint.
        Falls back to a hard-coded seed list if the endpoint fails.
        """
        try:
            data = self._get("/v3/reference/indices/SPX/components", params={"limit": 600})
            tickers = [r["ticker"] for r in data.get("results", [])]
            if tickers:
                logger.info("Fetched %d S&P 500 tickers from Polygon", len(tickers))
                return tickers
        except Exception as exc:
            logger.warning("S&P 500 component fetch failed: %s - using seed list", exc)

        seed = [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK.B",
            "JPM", "LLY", "V", "UNH", "XOM", "MA", "JNJ", "PG", "HD", "AVGO",
            "COST", "MRK", "ABBV", "CVX", "CRM", "BAC", "NFLX", "AMD", "PEP",
            "KO", "TMO", "WMT", "MCD", "DIS", "CSCO", "ABT", "ADBE", "DHR",
            "NEE", "ACN", "TXN", "CMCSA", "VZ", "INTC", "PM", "RTX", "ORCL",
            "QCOM", "HON", "T", "AMGN", "IBM", "CAT", "DE", "GS", "MS",
            "BLK", "SPGI", "AXP", "NOW", "ISRG", "ELV", "MDT", "SYK", "PLD",
            "LMT", "CI", "MO", "BSX", "GE", "ADP", "MMC", "VRTX", "REGN",
            "GILD", "ADI", "PANW", "KLAC", "MRVL", "SNPS", "CDNS", "FTNT",
        ]
        logger.warning("Using %d-ticker seed list for S&P 500", len(seed))
        return seed

    def get_aggs(
        self,
        ticker: str,
        from_date: date,
        to_date: date,
        multiplier: int = 1,
        timespan: str = "day",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV aggregates for a ticker.
        Returns DataFrame with columns: open, high, low, close, volume, vwap, timestamp.
        """
        params = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
        }
        path = (
            f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}"
            f"/{from_date.isoformat()}/{to_date.isoformat()}"
        )
        try:
            data = self._get(path, params)
        except Exception as exc:
            logger.error("get_aggs failed for %s: %s", ticker, exc)
            return pd.DataFrame()

        results = data.get("results", [])
        if not results:
            logger.debug("No agg data for %s (%s to %s)", ticker, from_date, to_date)
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df.rename(
            columns={
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "vw": "vwap",
                "t": "timestamp_ms",
            },
            inplace=True,
        )
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df.drop(columns=["timestamp_ms"], errors="ignore", inplace=True)
        keep = ["open", "high", "low", "close", "volume", "vwap"]
        df = df[[c for c in keep if c in df.columns]].copy()
        df = df.astype(float)
        return df

    def get_earnings_dates(self, ticker: str, weeks_ahead: int = 2) -> list[date]:
        """Return upcoming earnings dates for ticker within the next N weeks."""
        today = date.today()
        to_date = today + timedelta(weeks=weeks_ahead)
        params = {
            "ticker": ticker,
            "published_utc.gte": today.isoformat(),
            "published_utc.lte": to_date.isoformat(),
            "limit": 10,
        }
        results: list[dict] = []
        try:
            results = self._paginate("/v2/reference/financials", params)
        except Exception as exc:
            logger.warning("Earnings fetch failed for %s: %s", ticker, exc)

        dates: list[date] = []
        for r in results:
            raw = r.get("filing_date") or r.get("report_date")
            if raw:
                try:
                    dates.append(date.fromisoformat(raw[:10]))
                except ValueError:
                    pass
        return dates

    def has_earnings_this_week(self, ticker: str) -> bool:
        """Return True if ticker reports earnings in the next 7 calendar days."""
        upcoming = self.get_earnings_dates(ticker, weeks_ahead=1)
        today = date.today()
        week_end = today + timedelta(days=7)
        return any(today <= d <= week_end for d in upcoming)

    def get_snapshot(self, ticker: str) -> dict:
        """Return the latest snapshot (previous close, etc.) for a ticker."""
        try:
            data = self._get(f"/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}")
            return data.get("ticker", {})
        except Exception as exc:
            logger.error("Snapshot failed for %s: %s", ticker, exc)
            return {}


# ---------------------------------------------------------------------------
# CCXT Binance client
# ---------------------------------------------------------------------------

CRYPTO_SYMBOLS: list[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
CCXT_TIMEFRAME = "1h"


class CryptoFetcher:
    """CCXT Binance wrapper for crypto OHLCV data."""

    def __init__(self) -> None:
        self.exchange = ccxt.binance(
            {
                "apiKey": settings.binance_api_key,
                "secret": settings.binance_secret_key,
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
            }
        )

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str = CCXT_TIMEFRAME,
        limit: int = 300,
        since: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles for a crypto pair.
        Returns DataFrame indexed by UTC timestamp.
        """
        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
        except ccxt.BaseError as exc:
            logger.error("CCXT fetch_ohlcv failed for %s: %s", symbol, exc)
            return pd.DataFrame()

        if not raw:
            logger.warning("No CCXT data for %s", symbol)
            return pd.DataFrame()

        df = pd.DataFrame(raw, columns=["timestamp_ms", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        df.drop(columns=["timestamp_ms"], inplace=True)
        df = df.astype(float)
        return df

    def get_all_symbols(self, timeframe: str = CCXT_TIMEFRAME, limit: int = 300) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV for all configured crypto symbols."""
        result: dict[str, pd.DataFrame] = {}
        for symbol in CRYPTO_SYMBOLS:
            df = self.get_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not df.empty:
                result[symbol] = df
            else:
                logger.warning("Empty OHLCV for %s - skipped", symbol)
        return result

    def get_ticker(self, symbol: str) -> dict:
        """Return current ticker info (bid, ask, last price)."""
        try:
            return self.exchange.fetch_ticker(symbol)
        except ccxt.BaseError as exc:
            logger.error("fetch_ticker failed for %s: %s", symbol, exc)
            return {}

    def get_funding_rate(self, symbol: str) -> float:
        """
        Return the current perpetual funding rate for a symbol as a float
        (e.g. 0.0001 = 0.01%). Spot has no funding; this queries the perp via
        CCXT fetch_funding_rate and degrades to 0.0 on any failure so the
        strategy treats missing data as neutral rather than crashing.
        """
        if not getattr(self.exchange, "has", {}).get("fetchFundingRate", False):
            logger.debug("Exchange reports no fetchFundingRate support for %s", symbol)
            return 0.0
        try:
            data = self.exchange.fetch_funding_rate(symbol)
        except Exception as exc:
            logger.warning("Funding-rate fetch failed for %s: %s", symbol, exc)
            return 0.0
        rate = data.get("fundingRate") if isinstance(data, dict) else None
        if rate is None:
            return 0.0
        try:
            return float(rate)
        except (TypeError, ValueError):
            return 0.0


def fetch_crypto_funding(symbols: Optional[list[str]] = None) -> dict[str, float]:
    """Return {symbol: funding_rate} for the configured crypto symbols."""
    symbols = symbols or CRYPTO_SYMBOLS
    fetcher = CryptoFetcher()
    rates: dict[str, float] = {}
    for symbol in symbols:
        rates[symbol] = fetcher.get_funding_rate(symbol)
    return rates


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.rename("rsi")


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean().rename("atr")


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP (resets each day for daily data; rolling 14 bars for intraday)."""
    if "vwap" in df.columns and df["vwap"].notna().all():
        return df["vwap"].rename("vwap_calc")
    typical = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical * df["volume"]
    rolling = 14
    vwap = tp_vol.rolling(rolling).sum() / df["volume"].rolling(rolling).sum()
    return vwap.rename("vwap_calc")


def compute_volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    avg_vol = df["volume"].rolling(period).mean()
    return (df["volume"] / avg_vol.replace(0, np.nan)).rename("volume_ratio")


def compute_momentum(series: pd.Series, period: int = 10) -> pd.Series:
    return (series / series.shift(period) - 1).rename("momentum")


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean().rename(f"sma_{period}")


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average. NaN until `period` observations exist."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean().rename(f"ema_{period}")


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (macd_line, signal_line, histogram) using standard 12/26/9."""
    ema_fast = series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = (ema_fast - ema_slow).rename("macd")
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean().rename("macd_signal")
    hist = (macd_line - signal_line).rename("macd_hist")
    return macd_line, signal_line, hist


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's ADX. Measures trend strength regardless of direction."""
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().rename("adx")


def compute_realized_vol(series: pd.Series, period: int = 20) -> pd.Series:
    """Rolling realized volatility: stdev of log returns scaled by sqrt(period)."""
    log_ret = np.log(series / series.shift(1))
    return (log_ret.rolling(period).std() * np.sqrt(period)).rename("realized_vol")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all required features. Returns a new augmented DataFrame.
    Requires columns: open, high, low, close, volume.

    Phase 2 features: rsi, atr, vwap_calc, volume_ratio, momentum, sma_50/200,
    above_sma50/200, golden_cross.
    Phase 4 features: ema_9, ema_21, ema_200, macd, macd_signal, macd_hist,
    adx, realized_vol.
    """
    if df.empty:
        return df

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"add_features: missing columns {missing}")

    df = df.copy()

    df["rsi"] = compute_rsi(df["close"])
    df["atr"] = compute_atr(df)
    df["vwap_calc"] = compute_vwap(df)
    df["volume_ratio"] = compute_volume_ratio(df)
    df["momentum"] = compute_momentum(df["close"])
    df["sma_50"] = compute_sma(df["close"], 50)
    df["sma_200"] = compute_sma(df["close"], 200)

    df["above_sma50"] = (df["close"] > df["sma_50"]).astype(int)
    df["above_sma200"] = (df["close"] > df["sma_200"]).astype(int)
    df["golden_cross"] = (df["sma_50"] > df["sma_200"]).astype(int)

    # Phase 4 additions
    df["ema_9"] = compute_ema(df["close"], 9)
    df["ema_21"] = compute_ema(df["close"], 21)
    df["ema_200"] = compute_ema(df["close"], 200)
    macd_line, macd_signal, macd_hist = compute_macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist
    df["adx"] = compute_adx(df)
    df["realized_vol"] = compute_realized_vol(df["close"])

    return df


# ---------------------------------------------------------------------------
# High-level fetch functions (called by scheduler)
# ---------------------------------------------------------------------------

def fetch_stock_universe(
    lookback_days: int = 252,
    exclude_earnings: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch and feature-engineer historical daily data for the full S&P 500 universe.
    Filters out tickers with upcoming earnings if exclude_earnings=True.
    Returns {ticker: DataFrame}.
    """
    polygon = PolygonClient()
    tickers = polygon.get_sp500_tickers()

    to_date = date.today()
    from_date = to_date - timedelta(days=lookback_days)

    universe: dict[str, pd.DataFrame] = {}
    skipped_earnings: list[str] = []
    skipped_data: list[str] = []

    for ticker in tickers:
        if exclude_earnings and polygon.has_earnings_this_week(ticker):
            skipped_earnings.append(ticker)
            logger.info("Earnings guard: skipping %s", ticker)
            continue

        df = polygon.get_aggs(ticker, from_date, to_date)
        if df.empty or len(df) < 210:
            skipped_data.append(ticker)
            logger.debug("Insufficient history for %s (%d bars)", ticker, len(df))
            continue

        try:
            df = add_features(df)
        except ValueError as exc:
            logger.warning("Feature engineering failed for %s: %s", ticker, exc)
            continue

        universe[ticker] = df

    logger.info(
        "Stock universe built: %d tickers | earnings skip: %d | data skip: %d",
        len(universe),
        len(skipped_earnings),
        len(skipped_data),
    )
    return universe


def fetch_crypto_universe(
    timeframe: str = CCXT_TIMEFRAME,
    limit: int = 300,
) -> dict[str, pd.DataFrame]:
    """
    Fetch and feature-engineer recent OHLCV for all crypto symbols.
    Pass timeframe="4h" for the Phase 4 24/7 strategy.
    Returns {symbol: DataFrame}.
    """
    fetcher = CryptoFetcher()
    raw = fetcher.get_all_symbols(timeframe=timeframe, limit=limit)

    universe: dict[str, pd.DataFrame] = {}
    for symbol, df in raw.items():
        if len(df) < 50:
            logger.warning("Crypto %s has only %d bars - skipped", symbol, len(df))
            continue
        try:
            df = add_features(df)
            universe[symbol] = df
        except ValueError as exc:
            logger.warning("Feature engineering failed for %s: %s", symbol, exc)

    logger.info("Crypto universe built: %d symbols", len(universe))
    return universe


def fetch_latest_stock_bar(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch the most recent daily bar for a single ticker (real-time check)."""
    polygon = PolygonClient()
    today = date.today()
    from_date = today - timedelta(days=5)
    df = polygon.get_aggs(ticker, from_date, today)
    if df.empty:
        return None
    return df.tail(1)


def fetch_latest_crypto_bar(symbol: str, timeframe: str = "5m") -> Optional[dict]:
    """Return the most recent OHLCV candle for a crypto symbol."""
    fetcher = CryptoFetcher()
    ticker_data = fetcher.get_ticker(symbol)
    return ticker_data if ticker_data else None
