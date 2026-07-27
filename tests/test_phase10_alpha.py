"""tests/test_phase10_alpha.py - profit-upgrade strategy changes:
regime-mode crypto entries/exits, stock rotation selection, and the
expanded crypto universe. Fully offline (synthetic frames)."""

from __future__ import annotations

import os

for _k, _v in {
    "ALPACA_API_KEY": "test", "ALPACA_SECRET_KEY": "test",
    "STARTING_CAPITAL": "10000",
}.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import pandas as pd
import pytest

from data.fetcher import add_features
from strategies import crypto_24h as c24
from strategies.crypto_24h import (
    ema_stacked_bullish, bullish_ema_cross, generate_24h_entries,
    generate_24h_exits,
)
from strategies.stock_weekly import select_rotation_exits


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------

def make_established_uptrend_4h(n=360, seed=7):
    """A steady uptrend that started long ago: EMA9 has been above EMA21 for
    hundreds of bars, so there is NO fresh cross - the exact market the old
    cross-gate could never enter."""
    rng = np.random.default_rng(seed)
    base = np.linspace(200, 520, n)
    noise = rng.normal(0, 0.4, n).cumsum() * 0.08
    close = pd.Series(base + noise)
    high, low = close * 1.003, close * 0.997
    openp = close.shift(1).fillna(close.iloc[0])
    vol = pd.Series(rng.uniform(950, 1050, n))
    vol.iloc[-1] = 1200          # ~1.2x average: below the 1.5x spike gate
    idx = pd.date_range("2025-01-01", periods=n, freq="4h", tz="UTC")
    return add_features(pd.DataFrame(
        {"open": openp.values, "high": high.values, "low": low.values,
         "close": close.values, "volume": vol.values}, index=idx))


# ---------------------------------------------------------------------------
# regime posture primitives
# ---------------------------------------------------------------------------

def test_established_trend_has_no_fresh_cross_but_is_stacked():
    df = make_established_uptrend_4h()
    assert ema_stacked_bullish(df)          # posture: trend intact
    assert not bullish_ema_cross(df)        # but no cross within 3 bars


def test_regime_mode_enters_mid_trend_cross_mode_does_not():
    df = make_established_uptrend_4h()
    common = dict(
        universe={"BTC/USDT": df},
        funding_rates={},
        open_positions=[],
        equity=10_000.0,
        available_cash=10_000.0,
        btc_only=True,
    )
    assert generate_24h_entries(entry_mode="cross", **common) == []
    plans = generate_24h_entries(entry_mode="regime", **common)
    assert len(plans) == 1
    assert plans[0].symbol == "BTC/USDT"
    assert plans[0].units > 0


def _exit_df(macd_hist=-0.5, close=110.0, ema200=100.0, ema9=109.0, ema21=105.0):
    return pd.DataFrame({
        "close": [close] * 5, "ema_200": [ema200] * 5,
        "ema_9": [ema9] * 5, "ema_21": [ema21] * 5,
        "macd_hist": [macd_hist] * 5, "atr": [1.0] * 5,
    })


def test_regime_mode_ignores_macd_noise_exit():
    """One negative MACD bar mid-uptrend exits in cross mode (legacy) but is
    noise to a posture position - regime mode holds until the trend breaks."""
    pos = {"symbol": "BTC/USDT", "entry_price": 100.0, "stop_loss": 90.0}
    universe = {"BTC/USDT": _exit_df(macd_hist=-0.5)}
    legacy = generate_24h_exits([pos], universe, entry_mode="cross")
    assert [e.reason for e in legacy] == ["macd_hist_negative"]
    posture = generate_24h_exits([pos], universe, entry_mode="regime")
    assert posture == []


def test_regime_mode_still_exits_on_trend_break_and_stop():
    pos = {"symbol": "BTC/USDT", "entry_price": 100.0, "stop_loss": 90.0}
    below200 = {"BTC/USDT": _exit_df(close=95.0, ema200=100.0)}
    assert [e.reason for e in generate_24h_exits([pos], below200, entry_mode="regime")] \
        == ["below_ema200"]
    bear_cross = {"BTC/USDT": _exit_df(ema9=104.0, ema21=105.0)}
    assert [e.reason for e in generate_24h_exits([pos], bear_cross, entry_mode="regime")] \
        == ["ema_bearish_cross"]
    stopped = {"BTC/USDT": _exit_df(close=89.0, ema200=80.0, ema9=95.0, ema21=90.0,
                                    macd_hist=0.5)}
    exits = generate_24h_exits([pos], stopped, entry_mode="regime")
    assert len(exits) == 1 and "trailing_stop" in exits[0].reason


# ---------------------------------------------------------------------------
# stock rotation selection
# ---------------------------------------------------------------------------

def _pos(sym):
    return {"symbol": sym, "ticker": sym, "entry_price": 100.0, "quantity": 1.0}


def test_rotation_sells_only_dropouts():
    ranked = pd.DataFrame({"ticker": [f"T{i}" for i in range(30)]})
    # T5 = rank 6 (held); T25 = rank 26 (outside top 20); GONE = unranked
    held = [_pos("T5"), _pos("T25"), _pos("GONE")]
    exits = select_rotation_exits(held, ranked, keep_rank=20)
    assert [p["symbol"] for p in exits] == ["T25", "GONE"]


def test_rotation_keep_rank_boundary():
    ranked = pd.DataFrame({"ticker": [f"T{i}" for i in range(30)]})
    held = [_pos("T19"), _pos("T20")]                   # ranks 20 and 21
    exits = select_rotation_exits(held, ranked, keep_rank=20)
    assert [p["symbol"] for p in exits] == ["T20"]


def test_rotation_holds_everything_without_ranking():
    held = [_pos("AAPL"), _pos("MSFT")]
    assert select_rotation_exits(held, None) == []
    assert select_rotation_exits(held, pd.DataFrame()) == []


# ---------------------------------------------------------------------------
# universe config
# ---------------------------------------------------------------------------

def test_crypto_universe_includes_doge():
    from config.settings import Settings
    s = Settings(_env_file=None)
    assert "DOGE/USDT" in s.crypto_universe
    assert s.crypto_entry_mode == "regime"
    assert s.stock_exit_mode == "rotate"
    assert s.crypto_allocation_pct == pytest.approx(0.25)


def test_doge_maps_to_alpaca_pair():
    from execution.alpaca_crypto import to_alpaca_symbol
    assert to_alpaca_symbol("DOGE/USDT") == "DOGE/USD"


# ---------------------------------------------------------------------------
# btc_only gate must be venue-agnostic (regression: the Alpaca-routed bot
# compared "BTC/USD" against a hardcoded "BTC/USDT", so EVERY symbol was
# rejected and no crypto entry could ever be generated - for two weeks the
# logs truthfully reported "opened=0" while the path was structurally dead)
# ---------------------------------------------------------------------------

def test_is_btc_accepts_both_venue_quotes():
    from strategies.crypto_24h import is_btc
    assert is_btc("BTC/USD")      # Alpaca
    assert is_btc("BTC/USDT")     # Binance
    assert is_btc("btc/usd")      # case-insensitive
    assert not is_btc("ETH/USD")
    assert not is_btc("DOGE/USDT")


def test_btc_only_gate_admits_alpaca_btc():
    """The exact live configuration: Alpaca-style universe + btc_only=True."""
    df = make_established_uptrend_4h()
    plans = generate_24h_entries(
        universe={"BTC/USD": df, "ETH/USD": df},
        funding_rates={},
        open_positions=[],
        equity=10_000.0,
        available_cash=10_000.0,
        btc_only=True,
        entry_mode="regime",
    )
    assert [p.symbol for p in plans] == ["BTC/USD"]   # BTC admitted, ETH gated


def test_gate_log_explains_every_rejection():
    """A zero-entry cycle must be explainable, not mysterious."""
    df = make_established_uptrend_4h()
    gate_log = {}
    generate_24h_entries(
        universe={"BTC/USD": df, "ETH/USD": df},
        funding_rates={},
        open_positions=[],
        equity=10_000.0,
        available_cash=10_000.0,
        btc_only=True,
        entry_mode="regime",
        gate_log=gate_log,
    )
    assert gate_log["BTC/USD"] == "ok"
    assert gate_log["ETH/USD"] == "btc_only"


def test_gate_summary_counts_reasons():
    from strategies.crypto_24h import CryptoCycleResult
    res = CryptoCycleResult(gate_reasons={
        "ETH/USD": "btc_only", "SOL/USD": "btc_only", "BTC/USD": "below_ema200",
    })
    summary = res.gate_summary()
    assert "btc_only=2" in summary
    assert "below_ema200=1" in summary
