"""
tests/test_phase3.py
Phase 3 â€” Stock Scoring Engine & Signal Generation Tests
Covers: composite scoring, component scorers, entry/exit filters, DB persistence.
All DB and external calls are mocked.
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.fetcher import add_features
from strategies.stock_scorer import (
    ScoringResult,
    score_universe,
    get_top_tickers,
    _score_rsi,
    _score_momentum,
    _score_volume,
    _score_trend,
    _score_volatility,
    WEIGHTS,
    DEFAULT_TOP_N,
)
from strategies.signals import (
    SignalEvent,
    SignalType,
    AssetClass,
    generate_stock_signals,
    generate_stock_exit_signals,
    generate_crypto_signals,
    generate_crypto_exit_signals,
    persist_signals,
    run_stock_scoring_pipeline,
    run_crypto_signal_pipeline,
    STOCK_RSI_MIN,
    STOCK_RSI_MAX,
    STOCK_MIN_COMPOSITE,
    CRYPTO_RSI_MIN,
    CRYPTO_RSI_MAX,
    CRYPTO_MIN_COMPOSITE,
)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 260, start_price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(
        end=datetime.now(UTC).replace(hour=16, minute=0, second=0, microsecond=0),
        periods=n,
        freq="D",
        tz="UTC",
    )
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0005, 0.015, size=n)
    close = start_price * np.cumprod(1 + returns)
    noise = rng.uniform(0.001, 0.01, size=n)
    high   = close * (1 + noise)
    low    = close * (1 - noise)
    open_  = close * (1 + rng.uniform(-0.005, 0.005, size=n))
    volume = rng.integers(100_000, 5_000_000, size=n).astype(float)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
    return df


def _make_crypto_ohlcv(n: int = 300, start_price: float = 50_000.0) -> pd.DataFrame:
    idx = pd.date_range(
        end=datetime.now(UTC),
        periods=n,
        freq="h",
        tz="UTC",
    )
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0001, 0.008, size=n)
    close = start_price * np.cumprod(1 + returns)
    noise = rng.uniform(0.001, 0.005, size=n)
    high   = close * (1 + noise)
    low    = close * (1 - noise)
    open_  = close * (1 + rng.uniform(-0.003, 0.003, size=n))
    volume = rng.uniform(10, 500, size=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _make_universe(tickers: list[str], n: int = 260) -> dict[str, pd.DataFrame]:
    prices = [100.0 * (1 + 0.2 * i) for i in range(len(tickers))]
    return {t: add_features(_make_ohlcv(n, start_price=p)) for t, p in zip(tickers, prices)}


def _make_crypto_universe(symbols: list[str]) -> dict[str, pd.DataFrame]:
    return {s: add_features(_make_crypto_ohlcv()) for s in symbols}


def _make_scored_df(tickers: list[str], scores: list[float] | None = None) -> pd.DataFrame:
    if scores is None:
        scores = [0.9 - 0.05 * i for i in range(len(tickers))]
    return pd.DataFrame({
        "ticker":          tickers,
        "composite":       scores,
        "rsi_score":       [0.7] * len(tickers),
        "momentum_score":  [0.7] * len(tickers),
        "volume_score":    [0.7] * len(tickers),
        "trend_score":     [0.7] * len(tickers),
        "volatility_score":[0.7] * len(tickers),
    })


# ---------------------------------------------------------------------------
# Weight sanity
# ---------------------------------------------------------------------------

class TestWeights:
    def test_weights_sum_to_one(self):
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

    def test_weights_keys(self):
        expected = {"rsi_score", "momentum_score", "volume_score", "trend_score", "volatility_score"}
        assert set(WEIGHTS.keys()) == expected

    def test_all_weights_positive(self):
        assert all(v > 0 for v in WEIGHTS.values())


# ---------------------------------------------------------------------------
# Component scorer tests
# ---------------------------------------------------------------------------

class TestScoreRsi:
    def test_ideal_range_gets_max(self):
        df = add_features(_make_ohlcv(260))
        df = df.copy()
        df["rsi"] = 50.0
        universe = {"TEST": df}
        s = _score_rsi(universe)
        assert s["TEST"] == pytest.approx(1.0)

    def test_overbought_gets_low_score(self):
        df = add_features(_make_ohlcv(260))
        df = df.copy()
        df["rsi"] = 85.0
        universe = {"TEST": df}
        s = _score_rsi(universe)
        assert s["TEST"] < 0.2

    def test_oversold_gets_low_score(self):
        df = add_features(_make_ohlcv(260))
        df = df.copy()
        df["rsi"] = 20.0
        universe = {"TEST": df}
        s = _score_rsi(universe)
        assert s["TEST"] < 0.2

    def test_all_nan_rsi_gets_zero(self):
        df = add_features(_make_ohlcv(260))
        df = df.copy()
        df["rsi"] = np.nan
        universe = {"TEST": df}
        s = _score_rsi(universe)
        assert s["TEST"] == pytest.approx(0.0)

    def test_scores_in_range(self):
        universe = _make_universe(["A", "B", "C"])
        s = _score_rsi(universe)
        assert (s >= 0).all() and (s <= 1).all()

    def test_returns_series_named_rsi_score(self):
        universe = _make_universe(["AAPL"])
        s = _score_rsi(universe)
        assert s.name == "rsi_score"


class TestScoreMomentum:
    def test_returns_series_named_momentum_score(self):
        universe = _make_universe(["AAPL", "MSFT"])
        s = _score_momentum(universe)
        assert s.name == "momentum_score"

    def test_scores_in_range(self):
        universe = _make_universe(["A", "B", "C", "D"])
        s = _score_momentum(universe)
        assert (s >= 0).all() and (s <= 1).all()

    def test_rank_is_relative(self):
        universe = _make_universe(["LOW", "HIGH"])
        df_low  = add_features(_make_ohlcv(260, start_price=50.0))
        df_high = add_features(_make_ohlcv(260, start_price=200.0))
        df_low["momentum"]  = -0.05
        df_high["momentum"] =  0.15
        universe = {"LOW": df_low, "HIGH": df_high}
        s = _score_momentum(universe)
        assert s["HIGH"] > s["LOW"]


class TestScoreVolume:
    def test_high_volume_ratio_scores_above_half(self):
        df = add_features(_make_ohlcv(260))
        df = df.copy()
        df["volume_ratio"] = 2.0
        s = _score_volume({"TEST": df})
        assert s["TEST"] > 0.5

    def test_low_volume_ratio_scores_below_half(self):
        df = add_features(_make_ohlcv(260))
        df = df.copy()
        df["volume_ratio"] = 0.2
        s = _score_volume({"TEST": df})
        assert s["TEST"] < 0.5

    def test_scores_in_range(self):
        universe = _make_universe(["A", "B"])
        s = _score_volume(universe)
        assert (s >= 0).all() and (s <= 1).all()

    def test_returns_named_series(self):
        universe = _make_universe(["A"])
        assert _score_volume(universe).name == "volume_score"


class TestScoreTrend:
    def test_all_flags_true_gets_max(self):
        df = add_features(_make_ohlcv(260))
        df = df.copy()
        df["above_sma50"]  = 1
        df["above_sma200"] = 1
        df["golden_cross"] = 1
        s = _score_trend({"TEST": df})
        assert s["TEST"] == pytest.approx(1.0)

    def test_all_flags_false_gets_zero(self):
        df = add_features(_make_ohlcv(260))
        df = df.copy()
        df["above_sma50"]  = 0
        df["above_sma200"] = 0
        df["golden_cross"] = 0
        s = _score_trend({"TEST": df})
        assert s["TEST"] == pytest.approx(0.0)

    def test_partial_flags(self):
        df = add_features(_make_ohlcv(260))
        df = df.copy()
        df["above_sma50"]  = 1
        df["above_sma200"] = 0
        df["golden_cross"] = 0
        s = _score_trend({"TEST": df})
        assert 0.0 < s["TEST"] < 1.0

    def test_returns_named_series(self):
        universe = _make_universe(["A"])
        assert _score_trend(universe).name == "trend_score"


class TestScoreVolatility:
    def test_low_atr_norm_gets_high_score(self):
        df_low  = add_features(_make_ohlcv(260, start_price=100.0))
        df_high = add_features(_make_ohlcv(260, start_price=100.0))
        df_low["atr"]  = 0.5
        df_high["atr"] = 10.0
        df_low["close"]  = 100.0
        df_high["close"] = 100.0
        s = _score_volatility({"LOW_VOL": df_low, "HIGH_VOL": df_high})
        assert s["LOW_VOL"] > s["HIGH_VOL"]

    def test_scores_in_range(self):
        universe = _make_universe(["A", "B", "C"])
        s = _score_volatility(universe)
        assert (s >= 0).all() and (s <= 1).all()

    def test_returns_named_series(self):
        universe = _make_universe(["A"])
        assert _score_volatility(universe).name == "volatility_score"


# ---------------------------------------------------------------------------
# score_universe integration tests
# ---------------------------------------------------------------------------

class TestScoreUniverse:
    def test_returns_scoring_result(self):
        universe = _make_universe(["AAPL", "MSFT", "GOOG"])
        result = score_universe(universe)
        assert isinstance(result, ScoringResult)

    def test_ranked_has_all_tickers(self):
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
        universe = _make_universe(tickers)
        result = score_universe(universe)
        assert set(result.ranked["ticker"].tolist()) == set(tickers)

    def test_ranked_descending_by_composite(self):
        universe = _make_universe(["A", "B", "C", "D", "E"])
        result = score_universe(universe)
        composites = result.ranked["composite"].tolist()
        assert composites == sorted(composites, reverse=True)

    def test_composite_in_range(self):
        universe = _make_universe(["AAPL", "MSFT"])
        result = score_universe(universe)
        assert (result.ranked["composite"] >= 0).all()
        assert (result.ranked["composite"] <= 1).all()

    def test_required_score_columns_present(self):
        universe = _make_universe(["AAPL"])
        result = score_universe(universe)
        for col in ["rsi_score", "momentum_score", "volume_score", "trend_score", "volatility_score", "composite"]:
            assert col in result.ranked.columns

    def test_empty_universe_returns_empty_ranked(self):
        result = score_universe({})
        assert result.ranked.empty

    def test_excludes_empty_dataframes(self):
        universe = {"AAPL": add_features(_make_ohlcv(260)), "BAD": pd.DataFrame()}
        result = score_universe(universe)
        assert "BAD" not in result.ranked["ticker"].values
        assert "BAD" in result.excluded

    def test_excludes_missing_features(self):
        df_bad = add_features(_make_ohlcv(260)).drop(columns=["rsi"])
        universe = {"AAPL": add_features(_make_ohlcv(260)), "NORSI": df_bad}
        result = score_universe(universe)
        assert "NORSI" in result.excluded

    def test_top_n_limits_result(self):
        universe = _make_universe([f"T{i}" for i in range(20)])
        result = score_universe(universe, top_n=5)
        assert len(result.top(5)) == 5

    def test_custom_weights(self):
        universe = _make_universe(["AAPL", "MSFT"])
        custom_w = {
            "rsi_score": 0.5,
            "momentum_score": 0.2,
            "volume_score": 0.1,
            "trend_score": 0.1,
            "volatility_score": 0.1,
        }
        result = score_universe(universe, weights=custom_w)
        assert not result.ranked.empty

    def test_score_date_set(self):
        universe = _make_universe(["AAPL"])
        result = score_universe(universe)
        assert result.score_date is not None

    def test_summary_string(self):
        universe = _make_universe(["AAPL", "MSFT"])
        result = score_universe(universe)
        s = result.summary()
        assert "Scored" in s
        assert "AAPL" in s or "MSFT" in s


class TestGetTopTickers:
    def test_returns_list_of_strings(self):
        universe = _make_universe(["AAPL", "MSFT", "GOOG"])
        tickers = get_top_tickers(universe, top_n=2)
        assert isinstance(tickers, list)
        assert all(isinstance(t, str) for t in tickers)

    def test_respects_top_n(self):
        universe = _make_universe([f"T{i}" for i in range(15)])
        tickers = get_top_tickers(universe, top_n=5)
        assert len(tickers) == 5

    def test_empty_universe_returns_empty(self):
        assert get_top_tickers({}) == []


# ---------------------------------------------------------------------------
# Signal generation tests â€” stocks
# ---------------------------------------------------------------------------

class TestGenerateStockSignals:
    def test_returns_list(self):
        universe   = _make_universe(["AAPL", "MSFT"])
        scored_df  = _make_scored_df(["AAPL", "MSFT"])
        signals    = generate_stock_signals(scored_df, universe)
        assert isinstance(signals, list)

    def test_signals_are_buy_type(self):
        universe  = _make_universe(["AAPL"])
        scored_df = _make_scored_df(["AAPL"])
        signals   = generate_stock_signals(scored_df, universe)
        assert all(s.signal_type == SignalType.BUY for s in signals)

    def test_signals_asset_class_stock(self):
        universe  = _make_universe(["AAPL"])
        scored_df = _make_scored_df(["AAPL"])
        signals   = generate_stock_signals(scored_df, universe)
        assert all(s.asset_class == AssetClass.STOCK for s in signals)

    def test_rsi_filter_rejects_overbought(self):
        universe  = _make_universe(["HOT"])
        universe["HOT"]["rsi"] = 90.0       # overbought
        scored_df = _make_scored_df(["HOT"])
        signals   = generate_stock_signals(scored_df, universe)
        tickers   = [s.ticker for s in signals]
        assert "HOT" not in tickers

    def test_rsi_filter_rejects_oversold(self):
        universe  = _make_universe(["COLD"])
        universe["COLD"]["rsi"] = 20.0      # oversold
        scored_df = _make_scored_df(["COLD"])
        signals   = generate_stock_signals(scored_df, universe)
        tickers   = [s.ticker for s in signals]
        assert "COLD" not in tickers

    def test_low_composite_filtered(self):
        universe  = _make_universe(["WEAK"])
        scored_df = _make_scored_df(["WEAK"], scores=[0.1])
        signals   = generate_stock_signals(scored_df, universe)
        assert not signals

    def test_below_sma50_filtered(self):
        universe  = _make_universe(["DOWN"])
        universe["DOWN"]["above_sma50"] = 0   # below SMA50
        scored_df = _make_scored_df(["DOWN"])
        signals   = generate_stock_signals(scored_df, universe)
        tickers   = [s.ticker for s in signals]
        assert "DOWN" not in tickers

    def test_top_n_cap(self):
        tickers   = [f"T{i}" for i in range(20)]
        universe  = _make_universe(tickers)
        scored_df = _make_scored_df(tickers)
        signals   = generate_stock_signals(scored_df, universe, top_n=3)
        assert len(signals) <= 3

    def test_missing_ticker_skipped(self):
        universe  = _make_universe(["AAPL"])
        scored_df = _make_scored_df(["AAPL", "GHOST"])
        signals   = generate_stock_signals(scored_df, universe)
        tickers   = [s.ticker for s in signals]
        assert "GHOST" not in tickers

    def test_signal_has_all_fields(self):
        universe  = _make_universe(["AAPL"])
        scored_df = _make_scored_df(["AAPL"])
        signals   = generate_stock_signals(scored_df, universe)
        if signals:
            s = signals[0]
            assert s.ticker == "AAPL"
            assert s.close_price > 0
            assert s.atr >= 0


class TestGenerateStockExitSignals:
    def test_generates_sell_for_each_position(self):
        universe  = _make_universe(["AAPL", "MSFT"])
        positions = [{"ticker": "AAPL"}, {"ticker": "MSFT"}]
        signals   = generate_stock_exit_signals(positions, universe)
        assert len(signals) == 2
        assert all(s.signal_type == SignalType.SELL for s in signals)

    def test_exit_without_data_still_produces_signal(self):
        positions = [{"ticker": "GHOST"}]
        signals   = generate_stock_exit_signals(positions, {})
        assert len(signals) == 1
        assert signals[0].notes == "weekly_exit_no_data"

    def test_exit_with_data_has_close_price(self):
        universe  = _make_universe(["AAPL"])
        positions = [{"ticker": "AAPL"}]
        signals   = generate_stock_exit_signals(positions, universe)
        assert signals[0].close_price > 0

    def test_empty_positions_returns_empty(self):
        signals = generate_stock_exit_signals([], {})
        assert signals == []


# ---------------------------------------------------------------------------
# Signal generation tests â€” crypto
# ---------------------------------------------------------------------------

class TestGenerateCryptoSignals:
    def test_returns_list(self):
        universe  = _make_crypto_universe(["BTC/USDT"])
        scored_df = _make_scored_df(["BTC/USDT"])
        signals   = generate_crypto_signals(scored_df, universe)
        assert isinstance(signals, list)

    def test_signals_buy_type(self):
        universe  = _make_crypto_universe(["BTC/USDT"])
        scored_df = _make_scored_df(["BTC/USDT"])
        signals   = generate_crypto_signals(scored_df, universe)
        assert all(s.signal_type == SignalType.BUY for s in signals)

    def test_signals_asset_class_crypto(self):
        universe  = _make_crypto_universe(["ETH/USDT"])
        scored_df = _make_scored_df(["ETH/USDT"])
        signals   = generate_crypto_signals(scored_df, universe)
        assert all(s.asset_class == AssetClass.CRYPTO for s in signals)

    def test_negative_momentum_filtered(self):
        universe  = _make_crypto_universe(["BTC/USDT"])
        universe["BTC/USDT"]["momentum"] = -0.1
        scored_df = _make_scored_df(["BTC/USDT"])
        signals   = generate_crypto_signals(scored_df, universe)
        tickers   = [s.ticker for s in signals]
        assert "BTC/USDT" not in tickers

    def test_low_volume_ratio_filtered(self):
        universe  = _make_crypto_universe(["ETH/USDT"])
        universe["ETH/USDT"]["volume_ratio"] = 0.3
        scored_df = _make_scored_df(["ETH/USDT"])
        signals   = generate_crypto_signals(scored_df, universe)
        tickers   = [s.ticker for s in signals]
        assert "ETH/USDT" not in tickers

    def test_top_n_respected(self):
        symbols   = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
        universe  = _make_crypto_universe(symbols)
        scored_df = _make_scored_df(symbols)
        signals   = generate_crypto_signals(scored_df, universe, top_n=2)
        assert len(signals) <= 2


class TestGenerateCryptoExitSignals:
    def test_rsi_overbought_triggers_exit(self):
        universe  = _make_crypto_universe(["BTC/USDT"])
        universe["BTC/USDT"]["rsi"] = 90.0
        positions = [{"ticker": "BTC/USDT", "entry_price": 50000.0}]
        signals   = generate_crypto_exit_signals(positions, universe)
        assert any(s.ticker == "BTC/USDT" and s.signal_type == SignalType.SELL for s in signals)

    def test_negative_momentum_triggers_exit(self):
        universe  = _make_crypto_universe(["ETH/USDT"])
        universe["ETH/USDT"]["rsi"]      = 55.0
        universe["ETH/USDT"]["momentum"] = -0.05
        positions = [{"ticker": "ETH/USDT", "entry_price": 3000.0}]
        signals   = generate_crypto_exit_signals(positions, universe)
        assert any(s.ticker == "ETH/USDT" for s in signals)

    def test_trailing_stop_breach_triggers_exit(self):
        universe  = _make_crypto_universe(["SOL/USDT"])
        universe["SOL/USDT"]["rsi"]      = 55.0
        universe["SOL/USDT"]["momentum"] = 0.01
        universe["SOL/USDT"]["close"]    = 50.0
        universe["SOL/USDT"]["atr"]      = 10.0
        positions = [{"ticker": "SOL/USDT", "entry_price": 100.0}]
        # trailing_stop = 100 - 2*10 = 80 > 50 â†’ exit
        signals = generate_crypto_exit_signals(positions, universe)
        assert any(s.ticker == "SOL/USDT" for s in signals)

    def test_no_exit_when_healthy(self):
        universe  = _make_crypto_universe(["BNB/USDT"])
        universe["BNB/USDT"]["rsi"]      = 55.0
        universe["BNB/USDT"]["momentum"] = 0.05
        universe["BNB/USDT"]["close"]    = 300.0
        universe["BNB/USDT"]["atr"]      = 5.0
        positions = [{"ticker": "BNB/USDT", "entry_price": 290.0}]
        # trailing_stop = 290 - 2*5 = 280 < 300 â†’ hold
        signals = generate_crypto_exit_signals(positions, universe)
        assert not any(s.ticker == "BNB/USDT" for s in signals)

    def test_no_data_still_exits(self):
        positions = [{"ticker": "GHOST/USDT", "entry_price": 100.0}]
        signals   = generate_crypto_exit_signals(positions, {})
        assert len(signals) == 1
        assert signals[0].notes == "crypto_exit_no_data"

    def test_empty_positions_returns_empty(self):
        signals = generate_crypto_exit_signals([], {})
        assert signals == []


# ---------------------------------------------------------------------------
# SignalEvent dataclass tests
# ---------------------------------------------------------------------------

class TestSignalEvent:
    def _make_event(self, **kwargs) -> SignalEvent:
        defaults = dict(
            ticker="AAPL",
            signal_type=SignalType.BUY,
            asset_class=AssetClass.STOCK,
            composite_score=0.75,
            rsi=52.0,
            momentum=0.03,
            volume_ratio=1.2,
            trend_score=0.8,
            close_price=150.0,
            atr=2.5,
        )
        defaults.update(kwargs)
        return SignalEvent(**defaults)

    def test_to_dict_has_all_keys(self):
        event = self._make_event()
        d = event.to_dict()
        for key in ["ticker", "signal_type", "asset_class", "composite_score", "rsi",
                    "momentum", "volume_ratio", "trend_score", "close_price", "atr",
                    "generated_at", "notes"]:
            assert key in d

    def test_signal_type_value_in_dict(self):
        event = self._make_event(signal_type=SignalType.SELL)
        assert event.to_dict()["signal_type"] == "SELL"

    def test_generated_at_is_utc(self):
        event = self._make_event()
        assert event.generated_at.tzinfo is not None


# ---------------------------------------------------------------------------
# DB persistence tests (mocked)
# ---------------------------------------------------------------------------

class TestPersistSignals:
    def _make_event(self, ticker="AAPL", signal_type=SignalType.BUY) -> SignalEvent:
        return SignalEvent(
            ticker=ticker,
            signal_type=signal_type,
            asset_class=AssetClass.STOCK,
            composite_score=0.75,
            rsi=52.0,
            momentum=0.03,
            volume_ratio=1.2,
            trend_score=0.8,
            close_price=150.0,
            atr=2.5,
        )

    @patch("strategies.signals.SessionLocal")
    def test_writes_new_signal(self, mock_session_local):
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.filter.return_value.first.return_value = None
        mock_session_local.return_value = mock_session

        event   = self._make_event()
        written = persist_signals([event])

        assert written == 1
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()

    @patch("strategies.signals.SessionLocal")
    def test_skips_duplicate_signal(self, mock_session_local):
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.filter.return_value.first.return_value = MagicMock()
        mock_session_local.return_value = mock_session

        event   = self._make_event()
        written = persist_signals([event])

        assert written == 0
        mock_session.add.assert_not_called()
        mock_session.commit.assert_called_once()

    @patch("strategies.signals.SessionLocal")
    def test_empty_list_returns_zero(self, mock_session_local):
        written = persist_signals([])
        assert written == 0
        mock_session_local.assert_not_called()

    @patch("strategies.signals.SessionLocal")
    def test_rolls_back_on_error(self, mock_session_local):
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.filter.return_value.first.return_value = None
        mock_session.commit.side_effect = RuntimeError("DB down")
        mock_session_local.return_value = mock_session

        event = self._make_event()
        with pytest.raises(RuntimeError):
            persist_signals([event])

        mock_session.rollback.assert_called_once()

    @patch("strategies.signals.SessionLocal")
    def test_session_closed_on_success(self, mock_session_local):
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.filter.return_value.first.return_value = None
        mock_session_local.return_value = mock_session

        persist_signals([self._make_event()])
        mock_session.close.assert_called_once()

    @patch("strategies.signals.SessionLocal")
    def test_session_closed_on_error(self, mock_session_local):
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.filter.return_value.first.return_value = None
        mock_session.commit.side_effect = RuntimeError("DB down")
        mock_session_local.return_value = mock_session

        with pytest.raises(RuntimeError):
            persist_signals([self._make_event()])

        mock_session.close.assert_called_once()

    @patch("strategies.signals.SessionLocal")
    def test_multiple_signals_written(self, mock_session_local):
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.filter.return_value.first.return_value = None
        mock_session_local.return_value = mock_session

        events  = [self._make_event(ticker=t) for t in ["AAPL", "MSFT", "GOOG"]]
        written = persist_signals(events)

        assert written == 3
        assert mock_session.add.call_count == 3


# ---------------------------------------------------------------------------
# Pipeline convenience wrappers
# ---------------------------------------------------------------------------

class TestRunStockScoringPipeline:
    @patch("strategies.signals.persist_signals", return_value=2)
    def test_calls_persist_when_flagged(self, mock_persist):
        universe  = _make_universe(["AAPL", "MSFT"])
        scored_df = _make_scored_df(["AAPL", "MSFT"])
        run_stock_scoring_pipeline(scored_df, universe, top_n=2, persist=True)
        mock_persist.assert_called_once()

    @patch("strategies.signals.persist_signals")
    def test_skips_persist_when_flagged_false(self, mock_persist):
        universe  = _make_universe(["AAPL"])
        scored_df = _make_scored_df(["AAPL"])
        run_stock_scoring_pipeline(scored_df, universe, persist=False)
        mock_persist.assert_not_called()

    def test_returns_signal_list(self):
        universe  = _make_universe(["AAPL"])
        scored_df = _make_scored_df(["AAPL"])
        with patch("strategies.signals.persist_signals"):
            result = run_stock_scoring_pipeline(scored_df, universe)
        assert isinstance(result, list)


class TestRunCryptoSignalPipeline:
    @patch("strategies.signals.persist_signals", return_value=1)
    def test_calls_persist_when_flagged(self, mock_persist):
        symbols   = ["BTC/USDT"]
        universe  = _make_crypto_universe(symbols)
        scored_df = _make_scored_df(symbols)
        run_crypto_signal_pipeline(scored_df, universe, persist=True)
        mock_persist.assert_called_once()

    @patch("strategies.signals.persist_signals")
    def test_skips_persist_when_false(self, mock_persist):
        symbols   = ["BTC/USDT"]
        universe  = _make_crypto_universe(symbols)
        scored_df = _make_scored_df(symbols)
        run_crypto_signal_pipeline(scored_df, universe, persist=False)
        mock_persist.assert_not_called()

    def test_combines_buy_and_sell_signals(self):
        symbols   = ["BTC/USDT"]
        universe  = _make_crypto_universe(symbols)
        scored_df = _make_scored_df(symbols)
        open_pos  = [{"ticker": "ETH/USDT", "entry_price": 3000.0}]
        with patch("strategies.signals.persist_signals"):
            result = run_crypto_signal_pipeline(scored_df, universe, open_positions=open_pos)
        types = {s.signal_type for s in result}
        assert SignalType.BUY in types or SignalType.SELL in types

