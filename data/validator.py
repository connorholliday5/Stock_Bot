"""
data/validator.py
Phase 2 - Data Pipeline
Validates OHLCV DataFrames and computed features before they reach the strategy layer.
Catches: missing columns, NaN coverage, price anomalies, staleness, fee viability.

Phase 4 note: validate_crypto_df accepts an optional max_staleness_hours override
so higher-timeframe (4H) frames can be validated without the 1H staleness rule
falsely failing fresh data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

UTC = timezone.utc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_OHLCV = {"open", "high", "low", "close", "volume"}
REQUIRED_FEATURES = {
    "rsi",
    "atr",
    "vwap_calc",
    "volume_ratio",
    "momentum",
    "sma_50",
    "sma_200",
}

MIN_BARS_STOCK = 200
MIN_BARS_CRYPTO = 50

MAX_PRICE_SPIKE_RATIO = 3.0
MAX_STALENESS_HOURS_STOCK = 72
MAX_STALENESS_HOURS_CRYPTO = 2

RSI_VALID_RANGE = (0.0, 100.0)
MIN_VOLUME = 10_000

MIN_POSITION_USD = 50.0
STOCK_ROUND_TRIP_FEE_RATE = 0.002
CRYPTO_ROUND_TRIP_FEE_RATE = 0.002


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    ticker: str
    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> None:
        self.valid = False
        self.errors.append(reason)

    def warn(self, reason: str) -> None:
        self.warnings.append(reason)

    def summary(self) -> str:
        status = "PASS" if self.valid else "FAIL"
        parts = [f"[{status}] {self.ticker}"]
        if self.errors:
            parts.append("Errors: " + "; ".join(self.errors))
        if self.warnings:
            parts.append("Warnings: " + "; ".join(self.warnings))
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Core validators
# ---------------------------------------------------------------------------

def _check_columns(df: pd.DataFrame, required: set[str], result: ValidationResult) -> bool:
    missing = required - set(df.columns)
    if missing:
        result.fail(f"Missing columns: {sorted(missing)}")
        return False
    return True


def _check_bar_count(df: pd.DataFrame, minimum: int, result: ValidationResult) -> bool:
    if len(df) < minimum:
        result.fail(f"Insufficient bars: {len(df)} < {minimum}")
        return False
    return True


def _check_nan_coverage(df: pd.DataFrame, result: ValidationResult, threshold: float = 0.10) -> None:
    for col in df.columns:
        nan_frac = df[col].isna().mean()
        if nan_frac > threshold:
            result.warn(f"Column '{col}' has {nan_frac:.1%} NaN values")


def _check_ohlcv_sanity(df: pd.DataFrame, result: ValidationResult) -> None:
    recent = df.tail(30)

    bad_hl = (recent["high"] < recent["low"]).sum()
    if bad_hl > 0:
        result.fail(f"high < low on {bad_hl} bars")

    bad_open = ((recent["open"] > recent["high"]) | (recent["open"] < recent["low"])).sum()
    if bad_open > 0:
        result.warn(f"open outside [low, high] on {bad_open} bars")

    bad_close = ((recent["close"] > recent["high"]) | (recent["close"] < recent["low"])).sum()
    if bad_close > 0:
        result.fail(f"close outside [low, high] on {bad_close} bars")

    zero_vol = (recent["volume"] == 0).sum()
    if zero_vol > 3:
        result.warn(f"{zero_vol} bars with zero volume")

    low_vol = (recent["volume"] < MIN_VOLUME).sum()
    if low_vol > 5:
        result.warn(f"{low_vol} bars with volume < {MIN_VOLUME:,}")

    close = df["close"].dropna()
    if len(close) > 1:
        pct_change = close.pct_change().abs()
        spikes = (pct_change > (MAX_PRICE_SPIKE_RATIO - 1)).sum()
        if spikes > 0:
            result.warn(f"{spikes} bars with >200% single-bar price move (possible data error)")


def _check_rsi(df: pd.DataFrame, result: ValidationResult) -> None:
    if "rsi" not in df.columns:
        return
    rsi = df["rsi"].dropna()
    if len(rsi) == 0:
        result.warn("RSI is all NaN")
        return
    out_of_range = ((rsi < RSI_VALID_RANGE[0]) | (rsi > RSI_VALID_RANGE[1])).sum()
    if out_of_range > 0:
        result.fail(f"RSI out of [0, 100] on {out_of_range} bars")


def _check_atr(df: pd.DataFrame, result: ValidationResult) -> None:
    if "atr" not in df.columns:
        return
    atr = df["atr"].dropna()
    if len(atr) == 0:
        result.warn("ATR is all NaN")
        return
    if (atr < 0).any():
        result.fail("Negative ATR values detected")


def _check_sma_cross(df: pd.DataFrame, result: ValidationResult) -> None:
    if "sma_50" not in df.columns or "sma_200" not in df.columns:
        return
    recent = df[["sma_50", "sma_200"]].dropna()
    if len(recent) == 0:
        result.warn("SMA 50/200 both NaN - insufficient history for trend check")


def _check_staleness(
    df: pd.DataFrame,
    max_hours: float,
    result: ValidationResult,
) -> None:
    if df.index.empty:
        result.fail("DataFrame has no index - cannot check staleness")
        return
    last_ts = df.index[-1]
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=UTC)
    age_hours = (datetime.now(UTC) - last_ts).total_seconds() / 3600
    if age_hours > max_hours:
        result.fail(
            f"Data stale: last bar is {age_hours:.1f}h old (max {max_hours}h)"
        )


def _check_fee_viability(
    last_close: float,
    min_position: float,
    fee_rate: float,
    result: ValidationResult,
) -> None:
    """Ensure a minimum position covers round-trip fees at >1% expected gain."""
    if last_close <= 0:
        result.fail("Non-positive last close price")
        return
    if min_position < last_close * fee_rate * 100:
        result.warn(
            f"Minimum position ${min_position:.2f} may not cover round-trip fees "
            f"(fee ~{fee_rate*100:.2f}% = ${last_close*fee_rate:.4f}/share)"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_stock_df(ticker: str, df: pd.DataFrame) -> ValidationResult:
    """
    Full validation for a stock OHLCV + features DataFrame.
    Returns ValidationResult with .valid, .errors, .warnings.
    """
    result = ValidationResult(ticker=ticker)

    if df is None or df.empty:
        result.fail("DataFrame is None or empty")
        return result

    if not _check_bar_count(df, MIN_BARS_STOCK, result):
        return result

    if not _check_columns(df, REQUIRED_OHLCV, result):
        return result

    _check_ohlcv_sanity(df, result)
    _check_nan_coverage(df, result)
    _check_rsi(df, result)
    _check_atr(df, result)
    _check_sma_cross(df, result)
    _check_staleness(df, MAX_STALENESS_HOURS_STOCK, result)

    if "close" in df.columns:
        close_clean = df["close"].dropna()
        if not close_clean.empty:
            _check_fee_viability(close_clean.iloc[-1], MIN_POSITION_USD, STOCK_ROUND_TRIP_FEE_RATE, result)
        else:
            result.fail("close column is all NaN")

    if result.valid:
        logger.debug("Stock %s validated: OK (%d bars)", ticker, len(df))
    else:
        logger.warning("Stock %s validation FAILED: %s", ticker, "; ".join(result.errors))

    return result


def validate_crypto_df(
    symbol: str,
    df: pd.DataFrame,
    max_staleness_hours: Optional[float] = None,
) -> ValidationResult:
    """
    Full validation for a crypto OHLCV + features DataFrame.

    max_staleness_hours: override the default 1H staleness ceiling. Pass a larger
    value (e.g. 8.0) when validating 4H frames so fresh higher-timeframe bars are
    not falsely flagged stale. Defaults to MAX_STALENESS_HOURS_CRYPTO.
    """
    result = ValidationResult(ticker=symbol)

    if df is None or df.empty:
        result.fail("DataFrame is None or empty")
        return result

    if not _check_bar_count(df, MIN_BARS_CRYPTO, result):
        return result

    if not _check_columns(df, REQUIRED_OHLCV, result):
        return result

    staleness_ceiling = (
        max_staleness_hours if max_staleness_hours is not None else MAX_STALENESS_HOURS_CRYPTO
    )

    _check_ohlcv_sanity(df, result)
    _check_nan_coverage(df, result)
    _check_rsi(df, result)
    _check_atr(df, result)
    _check_staleness(df, staleness_ceiling, result)

    if "close" in df.columns:
        close_clean = df["close"].dropna()
        if not close_clean.empty:
            _check_fee_viability(close_clean.iloc[-1], MIN_POSITION_USD, CRYPTO_ROUND_TRIP_FEE_RATE, result)
        else:
            result.fail("close column is all NaN")

    if result.valid:
        logger.debug("Crypto %s validated: OK (%d bars)", symbol, len(df))
    else:
        logger.warning("Crypto %s validation FAILED: %s", symbol, "; ".join(result.errors))

    return result


def validate_universe(
    universe: dict[str, pd.DataFrame],
    asset_type: str = "stock",
) -> tuple[dict[str, pd.DataFrame], list[ValidationResult]]:
    """
    Validate every ticker/symbol in a universe dict.
    Returns (clean_universe, all_results).
    clean_universe contains only valid entries.
    """
    validate_fn = validate_stock_df if asset_type == "stock" else validate_crypto_df
    clean: dict[str, pd.DataFrame] = {}
    results: list[ValidationResult] = []

    for ticker, df in universe.items():
        res = validate_fn(ticker, df)
        results.append(res)
        if res.valid:
            clean[ticker] = df
        else:
            logger.info("Dropping %s from universe: %s", ticker, "; ".join(res.errors))

    pass_count = sum(1 for r in results if r.valid)
    logger.info(
        "Universe validation [%s]: %d/%d passed",
        asset_type,
        pass_count,
        len(results),
    )
    return clean, results


def log_validation_report(results: list[ValidationResult]) -> None:
    """Write a concise validation report to the logger."""
    failures = [r for r in results if not r.valid]
    warnings_only = [r for r in results if r.valid and r.warnings]

    logger.info("=== Validation Report ===")
    logger.info("Total: %d | Pass: %d | Fail: %d | Warn: %d",
                len(results),
                len(results) - len(failures),
                len(failures),
                len(warnings_only))

    for r in failures:
        logger.warning(r.summary())
    for r in warnings_only:
        logger.debug(r.summary())