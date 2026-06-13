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
