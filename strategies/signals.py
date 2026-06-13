"""
strategies/signals.py
Buy/sell signal generation for stock and crypto strategies.
Persists Signal records to DB via db.py session pattern.

Phase 4 fix: SignalType (BUY/SELL/HOLD) is the strategy vocabulary; the DB stores
SignalDirection (long/short/flat). persist_signals translates at the boundary via
_DIRECTION_MAP. A spot bot never shorts, so SELL (exit a long) maps to FLAT, not
SHORT. SHORT stays reserved for a future margin strategy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from database.db import SessionLocal, Signal, SignalDirection

logger = logging.getLogger(__name__)

UTC = timezone.utc


class SignalType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class AssetClass(str, Enum):
    STOCK = "stock"
    CRYPTO = "crypto"


# Strategy-layer SignalType -> DB SignalDirection value string.
# Proven against the live schema: the Enum column accepts the value string.
_DIRECTION_MAP: dict[SignalType, str] = {
    SignalType.BUY: SignalDirection.LONG.value,    # "long"
    SignalType.SELL: SignalDirection.FLAT.value,   # "flat" (spot exit, not a short)
    SignalType.HOLD: SignalDirection.FLAT.value,   # "flat"
}


@dataclass
class SignalEvent:
    # In memory container populated by generators, mapped to Signal schema on write.
    ticker: str
    signal_type: SignalType
    asset_class: AssetClass
    composite_score: float
    rsi: float
    momentum: float
    volume_ratio: float
    trend_score: float
    close_price: float
    atr: float
    sma_50: float = 0.0
    sma_200: float = 0.0
    adx: float = 0.0
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "signal_type": self.signal_type.value,
            "direction": _DIRECTION_MAP[self.signal_type],
            "asset_class": self.asset_class.value,
            "composite_score": self.composite_score,
            "rsi": self.rsi,
            "momentum": self.momentum,
            "volume_ratio": self.volume_ratio,
            "trend_score": self.trend_score,
            "close_price": self.close_price,
            "atr": self.atr,
            "sma_50": self.sma_50,
            "sma_200": self.sma_200,
            "adx": self.adx,
            "generated_at": self.generated_at.isoformat(),
            "notes": self.notes,
        }


# Stock entry thresholds
STOCK_RSI_MIN = 30.0
STOCK_RSI_MAX = 75.0
STOCK_MIN_VOLUME_RATIO = 0.8
STOCK_MIN_COMPOSITE = 0.45
STOCK_TREND_REQUIRE_SMA50 = True


def generate_stock_signals(
    scored_df: pd.DataFrame,
    df_universe: dict[str, pd.DataFrame],
    top_n: int = 10,
) -> list[SignalEvent]:
    signals: list[SignalEvent] = []
    candidates = scored_df.head(top_n * 2)

    for _, row in candidates.iterrows():
        ticker = row["ticker"]
        df = df_universe.get(ticker)
        if df is None or df.empty:
            logger.debug("Skipping %s, no data in universe", ticker)
            continue

        last = df.iloc[-1]
        rsi = float(last.get("rsi", 50.0))
        volume_ratio = float(last.get("volume_ratio", 1.0))
        close = float(last.get("close", 0.0))
        atr = float(last.get("atr", 0.0))
        above_sma50 = bool(last.get("above_sma50", 0))
        momentum = float(last.get("momentum", 0.0))
        sma_50 = float(last.get("sma_50", 0.0))
        sma_200 = float(last.get("sma_200", 0.0))
        composite = float(row["composite"])
        trend_score = float(row["trend_score"])

        if not (STOCK_RSI_MIN <= rsi <= STOCK_RSI_MAX):
            continue
        if volume_ratio < STOCK_MIN_VOLUME_RATIO:
            continue
        if composite < STOCK_MIN_COMPOSITE:
            continue
        if STOCK_TREND_REQUIRE_SMA50 and not above_sma50:
            continue
        if close <= 0:
            continue

        signals.append(SignalEvent(
            ticker=ticker,
            signal_type=SignalType.BUY,
            asset_class=AssetClass.STOCK,
            composite_score=composite,
            rsi=rsi,
            momentum=momentum,
            volume_ratio=volume_ratio,
            trend_score=trend_score,
            close_price=close,
            atr=atr,
            sma_50=sma_50,
            sma_200=sma_200,
            notes=f"top_n_rank={int(row.name)+1}",
        ))
        if len(signals) >= top_n:
            break

    logger.info("Stock signal generation: %d BUY signals from %d candidates", len(signals), len(candidates))
    return signals


def generate_stock_exit_signals(
    open_positions: list[dict],
    df_universe: dict[str, pd.DataFrame],
) -> list[SignalEvent]:
    # Friday scheduled exit, time based, closes all open stock positions.
    signals: list[SignalEvent] = []
    for pos in open_positions:
        ticker = pos.get("ticker", "")
        df = df_universe.get(ticker)
        if df is None or df.empty:
            signals.append(SignalEvent(
                ticker=ticker, signal_type=SignalType.SELL, asset_class=AssetClass.STOCK,
                composite_score=0.0, rsi=50.0, momentum=0.0, volume_ratio=1.0,
                trend_score=0.0, close_price=0.0, atr=0.0, notes="weekly_exit_no_data",
            ))
            continue
        last = df.iloc[-1]
        signals.append(SignalEvent(
            ticker=ticker, signal_type=SignalType.SELL, asset_class=AssetClass.STOCK,
            composite_score=0.0, rsi=float(last.get("rsi", 50.0)), momentum=0.0,
            volume_ratio=float(last.get("volume_ratio", 1.0)), trend_score=0.0,
            close_price=float(last.get("close", 0.0)), atr=float(last.get("atr", 0.0)),
            notes="weekly_scheduled_exit",
        ))
    logger.info("Stock exit signals: %d SELL signals", len(signals))
    return signals


# Crypto entry thresholds
CRYPTO_RSI_MIN = 45.0
CRYPTO_RSI_MAX = 80.0
CRYPTO_MIN_VOLUME_RATIO = 1.0
CRYPTO_MIN_COMPOSITE = 0.50
CRYPTO_MOMENTUM_MIN = 0.0


def generate_crypto_signals(
    scored_df: pd.DataFrame,
    df_universe: dict[str, pd.DataFrame],
    top_n: int = 4,
) -> list[SignalEvent]:
    signals: list[SignalEvent] = []
    candidates = scored_df.head(top_n * 2)

    for _, row in candidates.iterrows():
        ticker = row["ticker"]
        df = df_universe.get(ticker)
        if df is None or df.empty:
            continue

        last = df.iloc[-1]
        rsi = float(last.get("rsi", 50.0))
        volume_ratio = float(last.get("volume_ratio", 1.0))
        close = float(last.get("close", 0.0))
        atr = float(last.get("atr", 0.0))
        momentum = float(last.get("momentum", 0.0))
        sma_50 = float(last.get("sma_50", 0.0))
        sma_200 = float(last.get("sma_200", 0.0))
        composite = float(row["composite"])
        trend_score = float(row["trend_score"])

        if not (CRYPTO_RSI_MIN <= rsi <= CRYPTO_RSI_MAX):
            continue
        if volume_ratio < CRYPTO_MIN_VOLUME_RATIO:
            continue
        if composite < CRYPTO_MIN_COMPOSITE:
            continue
        if momentum < CRYPTO_MOMENTUM_MIN:
            continue
        if close <= 0:
            continue

        signals.append(SignalEvent(
            ticker=ticker,
            signal_type=SignalType.BUY,
            asset_class=AssetClass.CRYPTO,
            composite_score=composite,
            rsi=rsi,
            momentum=momentum,
            volume_ratio=volume_ratio,
            trend_score=trend_score,
            close_price=close,
            atr=atr,
            sma_50=sma_50,
            sma_200=sma_200,
            notes="crypto_trend_entry",
        ))
        if len(signals) >= top_n:
            break

    logger.info("Crypto signal generation: %d BUY signals from %d candidates", len(signals), len(candidates))
    return signals


def generate_crypto_exit_signals(
    open_positions: list[dict],
    df_universe: dict[str, pd.DataFrame],
    trailing_stop_atr_mult: float = 2.0,
) -> list[SignalEvent]:
    # Exit on overbought RSI, negative momentum, or ATR trailing stop breach.
    signals: list[SignalEvent] = []
    for pos in open_positions:
        ticker = pos.get("ticker", "")
        entry_price = float(pos.get("entry_price", 0.0))
        df = df_universe.get(ticker)

        if df is None or df.empty:
            signals.append(SignalEvent(
                ticker=ticker, signal_type=SignalType.SELL, asset_class=AssetClass.CRYPTO,
                composite_score=0.0, rsi=50.0, momentum=0.0, volume_ratio=1.0,
                trend_score=0.0, close_price=0.0, atr=0.0, notes="crypto_exit_no_data",
            ))
            continue

        last = df.iloc[-1]
        close = float(last.get("close", 0.0))
        rsi = float(last.get("rsi", 50.0))
        atr = float(last.get("atr", 0.0))
        momentum = float(last.get("momentum", 0.0))
        volume_ratio = float(last.get("volume_ratio", 1.0))

        exit_reason: Optional[str] = None
        if rsi > CRYPTO_RSI_MAX:
            exit_reason = f"rsi_overbought={rsi:.1f}"
        elif momentum < 0:
            exit_reason = f"momentum_negative={momentum:.4f}"
        elif entry_price > 0 and atr > 0:
            trailing_stop = entry_price - trailing_stop_atr_mult * atr
            if close < trailing_stop:
                exit_reason = f"trailing_stop_breach close={close:.2f} stop={trailing_stop:.2f}"

        if exit_reason:
            signals.append(SignalEvent(
                ticker=ticker, signal_type=SignalType.SELL, asset_class=AssetClass.CRYPTO,
                composite_score=0.0, rsi=rsi, momentum=momentum, volume_ratio=volume_ratio,
                trend_score=0.0, close_price=close, atr=atr, notes=exit_reason,
            ))
            logger.info("Crypto exit signal %s, %s", ticker, exit_reason)

    logger.info("Crypto exit signals: %d SELL signals", len(signals))
    return signals


def persist_signals(signals: list[SignalEvent]) -> int:
    """
    Write SignalEvent list to the Signal table, mapped to the real schema:
        ticker -> symbol, signal_type -> direction (via _DIRECTION_MAP),
        asset_class -> asset_type, momentum -> momentum_20d, adx -> adx,
        generated_at -> created_at.
    Skips duplicates (same symbol, direction, asset_type on the same UTC day).
    Columns sentiment_score, ml_prediction, ml_confidence are left to defaults
    and populated in later phases.
    """
    if not signals:
        return 0

    written = 0
    session: Session = SessionLocal()
    try:
        for event in signals:
            direction_value = _DIRECTION_MAP[event.signal_type]
            today = event.generated_at.date()
            day_start = datetime(today.year, today.month, today.day, tzinfo=UTC)

            existing = (
                session.query(Signal)
                .filter_by(
                    symbol=event.ticker,
                    direction=direction_value,
                    asset_type=event.asset_class.value,
                )
                .filter(Signal.created_at >= day_start)
                .first()
            )
            if existing:
                logger.debug("Skipping duplicate signal %s %s", direction_value, event.ticker)
                continue

            record = Signal(
                symbol=event.ticker,
                direction=direction_value,
                asset_type=event.asset_class.value,
                composite_score=event.composite_score,
                rsi=event.rsi,
                sma_50=event.sma_50,
                sma_200=event.sma_200,
                adx=event.adx,
                volume_ratio=event.volume_ratio,
                momentum_20d=event.momentum,
                acted_on=False,
                created_at=event.generated_at,
            )
            session.add(record)
            written += 1

        session.commit()
        logger.info("Persisted %d signal(s) to DB", written)
    except Exception as exc:
        session.rollback()
        logger.exception("Failed to persist signals: %s", exc)
        raise
    finally:
        session.close()

    return written


def run_stock_scoring_pipeline(
    scored_df: pd.DataFrame,
    df_universe: dict[str, pd.DataFrame],
    top_n: int = 10,
    persist: bool = True,
) -> list[SignalEvent]:
    signals = generate_stock_signals(scored_df, df_universe, top_n=top_n)
    if persist:
        persist_signals(signals)
    return signals


def run_crypto_signal_pipeline(
    scored_df: pd.DataFrame,
    df_universe: dict[str, pd.DataFrame],
    open_positions: Optional[list[dict]] = None,
    top_n: int = 4,
    persist: bool = True,
) -> list[SignalEvent]:
    buy_signals = generate_crypto_signals(scored_df, df_universe, top_n=top_n)
    sell_signals = generate_crypto_exit_signals(open_positions or [], df_universe)
    all_signals = buy_signals + sell_signals
    if persist:
        persist_signals(all_signals)
    return all_signals