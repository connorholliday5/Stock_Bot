"""
tests/test_phase2.py
Phase 2 — Data Pipeline Tests
Covers: feature engineering, validation logic, earnings guard, fee viability.
All network calls are mocked.
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.fetcher import (
    add_features,
    compute_atr,
    compute_momentum,
    compute_rsi,
    compute_sma,
    compute_volume_ratio,
    compute_vwap,
)
from data.validator import (
    ValidationResult,
    validate_crypto_df,
    validate_stock_df,
    validate_universe,
    MIN_BARS_STOCK,
    MIN_BARS_CRYPTO,
)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 260, start_price: float = 100.0) -> pd.DataFrame:
    """Generate n bars of synthetic OHLCV data with UTC DatetimeIndex."""
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
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = close * (1 + rng.uniform(-0.005, 0.005, size=n))
    volume = rng.integers(100_000, 5_000_000, size=n).astype(float)
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3
    return df


def _make_crypto_ohlcv(n: int = 300, start_price: float = 50_000.0) -> pd.DataFrame:
    """Generate synthetic 1h crypto bars."""
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
    high = close * (1 + noise)
    low = close * (1 - noise)
    open_ = close * (1 + rng.uniform(-0.003, 0.003, size=n))
    volume = rng.uniform(10, 500, size=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Feature engineering tests
# ---------------------------------------------------------------------------

class TestRSI:
    def test_output_length(self):
        df = _make_ohlcv(260)
        rsi = compute_rsi(df["close"])
        assert len(rsi) == len(df)

    def test_range_after_warmup(self):
        df = _make_ohlcv(260)
        rsi = compute_rsi(df["close"]).dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_name(self):
        rsi = compute_rsi(_make_ohlcv()["close"])
        assert rsi.name == "rsi"

    def test_constant_series_rsi(self):
        s = pd.Series([100.0] * 50)
        rsi = compute_rsi(s).dropna()
        assert rsi.isna().all() or ((rsi == 50) | rsi.isna()).all()

    def test_rising_series_high_rsi(self):
        rng = np.random.default_rng(1)
        base = np.linspace(100, 200, 100)
        noise = rng.normal(0, 0.5, 100)
        s = pd.Series(base + noise)
        rsi = compute_rsi(s, period=14).dropna()
        assert len(rsi) > 0, "RSI should have non-NaN values after warmup"
        assert rsi.iloc[-1] > 70


class TestATR:
    def test_output_length(self):
        df = _make_ohlcv(260)
        atr = compute_atr(df)
        assert len(atr) == len(df)

    def test_non_negative(self):
        df = _make_ohlcv(260)
        atr = compute_atr(df).dropna()
        assert (atr >= 0).all()

    def test_name(self):
        atr = compute_atr(_make_ohlcv())
        assert atr.name == "atr"


class TestVWAP:
    def test_uses_existing_vwap_col(self):
        df = _make_ohlcv(260)
        vwap = compute_vwap(df)
        assert len(vwap) == len(df)

    def test_computes_without_vwap_col(self):
        df = _make_ohlcv(260)
        df.drop(columns=["vwap"], inplace=True)
        vwap = compute_vwap(df).dropna()
        assert len(vwap) > 0
        assert (vwap > 0).all()


class TestVolumeRatio:
    def test_output_length(self):
        df = _make_ohlcv(260)
        vr = compute_volume_ratio(df)
        assert len(vr) == len(df)

    def test_positive_values(self):
        df = _make_ohlcv(260)
        vr = compute_volume_ratio(df).dropna()
        assert (vr > 0).all()

    def test_name(self):
        assert compute_volume_ratio(_make_ohlcv()).name == "volume_ratio"


class TestMomentum:
    def test_output_length(self):
        df = _make_ohlcv(260)
        m = compute_momentum(df["close"])
        assert len(m) == len(df)

    def test_rising_positive(self):
        s = pd.Series(range(1, 50), dtype=float)
        m = compute_momentum(s, period=5).dropna()
        assert (m > 0).all()


class TestSMA:
    def test_sma50(self):
        df = _make_ohlcv(260)
        sma = compute_sma(df["close"], 50).dropna()
        assert len(sma) == len(df) - 49

    def test_sma200(self):
        df = _make_ohlcv(260)
        sma = compute_sma(df["close"], 200).dropna()
        assert len(sma) == len(df) - 199

    def test_name(self):
        df = _make_ohlcv(260)
        sma = compute_sma(df["close"], 50)
        assert sma.name == "sma_50"


class TestAddFeatures:
    def test_columns_added(self):
        df = _make_ohlcv(260)
        out = add_features(df)
        for col in ["rsi", "atr", "vwap_calc", "volume_ratio", "momentum", "sma_50", "sma_200"]:
            assert col in out.columns, f"Missing feature: {col}"

    def test_derived_flags(self):
        df = _make_ohlcv(260)
        out = add_features(df)
        assert "above_sma50" in out.columns
        assert "above_sma200" in out.columns
        assert "golden_cross" in out.columns
        assert set(out["above_sma50"].dropna().unique()).issubset({0, 1})

    def test_no_mutation_of_original(self):
        df = _make_ohlcv(260)
        orig_cols = set(df.columns)
        add_features(df)
        assert set(df.columns) == orig_cols

    def test_raises_on_missing_columns(self):
        df = _make_ohlcv(50).drop(columns=["close"])
        with pytest.raises(ValueError, match="missing columns"):
            add_features(df)

    def test_empty_df_returns_empty(self):
        result = add_features(pd.DataFrame())
        assert result.empty


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidationResult:
    def test_starts_valid(self):
        r = ValidationResult(ticker="TEST")
        assert r.valid is True

    def test_fail_sets_invalid(self):
        r = ValidationResult(ticker="TEST")
        r.fail("bad data")
        assert r.valid is False
        assert "bad data" in r.errors

    def test_warn_keeps_valid(self):
        r = ValidationResult(ticker="TEST")
        r.warn("something suspicious")
        assert r.valid is True
        assert "something suspicious" in r.warnings

    def test_summary_contains_ticker(self):
        r = ValidationResult(ticker="AAPL")
        r.fail("oops")
        assert "AAPL" in r.summary()
        assert "FAIL" in r.summary()


class TestValidateStockDf:
    def test_valid_stock(self):
        df = add_features(_make_ohlcv(260))
        result = validate_stock_df("AAPL", df)
        assert result.valid, result.errors

    def test_none_fails(self):
        result = validate_stock_df("X", None)
        assert not result.valid

    def test_empty_fails(self):
        result = validate_stock_df("X", pd.DataFrame())
        assert not result.valid

    def test_insufficient_bars_fails(self):
        df = add_features(_make_ohlcv(50))
        result = validate_stock_df("X", df)
        assert not result.valid
        assert any("Insufficient" in e for e in result.errors)

    def test_missing_ohlcv_column_fails(self):
        df = add_features(_make_ohlcv(260)).drop(columns=["close"])
        result = validate_stock_df("X", df)
        assert not result.valid
        assert any("Missing columns" in e for e in result.errors)

    def test_high_lt_low_fails(self):
        df = add_features(_make_ohlcv(260))
        df = df.copy()
        df.loc[df.index[-5:], "high"] = df.loc[df.index[-5:], "low"] - 1
        result = validate_stock_df("X", df)
        assert not result.valid
        assert any("high < low" in e for e in result.errors)

    def test_stale_data_fails(self):
        df = add_features(_make_ohlcv(260))
        old_idx = pd.date_range(
            end=datetime.now(UTC) - timedelta(days=5),
            periods=len(df),
            freq="D",
            tz="UTC",
        )
        df.index = old_idx
        result = validate_stock_df("X", df)
        assert not result.valid
        assert any("stale" in e.lower() for e in result.errors)

    def test_rsi_out_of_range_fails(self):
        df = add_features(_make_ohlcv(260))
        df = df.copy()
        df["rsi"] = 150.0
        result = validate_stock_df("X", df)
        assert not result.valid
        assert any("RSI out of" in e for e in result.errors)

    def test_negative_atr_fails(self):
        df = add_features(_make_ohlcv(260))
        df = df.copy()
        df["atr"] = -1.0
        result = validate_stock_df("X", df)
        assert not result.valid
        assert any("Negative ATR" in e for e in result.errors)


class TestValidateCryptoDf:
    def test_valid_crypto(self):
        df = add_features(_make_crypto_ohlcv(300))
        result = validate_crypto_df("BTC/USDT", df)
        assert result.valid, result.errors

    def test_none_fails(self):
        result = validate_crypto_df("ETH/USDT", None)
        assert not result.valid

    def test_insufficient_bars_fails(self):
        df = add_features(_make_crypto_ohlcv(20))
        result = validate_crypto_df("BTC/USDT", df)
        assert not result.valid

    def test_stale_crypto_fails(self):
        df = add_features(_make_crypto_ohlcv(300))
        old_idx = pd.date_range(
            end=datetime.now(UTC) - timedelta(hours=5),
            periods=len(df),
            freq="h",
            tz="UTC",
        )
        df.index = old_idx
        result = validate_crypto_df("BTC/USDT", df)
        assert not result.valid
        assert any("stale" in e.lower() for e in result.errors)


class TestValidateUniverse:
    def test_all_valid(self):
        universe = {
            "AAPL": add_features(_make_ohlcv(260)),
            "MSFT": add_features(_make_ohlcv(260, start_price=300.0)),
        }
        clean, results = validate_universe(universe, asset_type="stock")
        assert len(clean) == 2
        assert all(r.valid for r in results)

    def test_filters_invalid(self):
        universe = {
            "AAPL": add_features(_make_ohlcv(260)),
            "BAD": pd.DataFrame(),
        }
        clean, results = validate_universe(universe, asset_type="stock")
        assert "AAPL" in clean
        assert "BAD" not in clean
        assert len(results) == 2

    def test_all_invalid_returns_empty(self):
        universe = {"X": pd.DataFrame(), "Y": pd.DataFrame()}
        clean, results = validate_universe(universe, asset_type="stock")
        assert len(clean) == 0
        assert all(not r.valid for r in results)

    def test_crypto_type(self):
        universe = {
            "BTC/USDT": add_features(_make_crypto_ohlcv(300)),
        }
        clean, results = validate_universe(universe, asset_type="crypto")
        assert "BTC/USDT" in clean


# ---------------------------------------------------------------------------
# Polygon client (mocked)
# ---------------------------------------------------------------------------

class TestPolygonClient:
    @patch("data.fetcher.requests.Session")
    def test_get_aggs_returns_dataframe(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        payload = {
            "results": [
                {"o": 100, "h": 105, "l": 98, "c": 102, "v": 1_000_000, "vw": 101.5, "t": 1_700_000_000_000}
                for _ in range(5)
            ]
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload
        mock_session.get.return_value = mock_resp

        from data.fetcher import PolygonClient
        from datetime import date
        client = PolygonClient()
        df = client.get_aggs("AAPL", date(2024, 1, 1), date(2024, 1, 5))
        assert not df.empty
        assert "close" in df.columns

    @patch("data.fetcher.requests.Session")
    def test_has_earnings_this_week_true(self, mock_session_cls):
        from datetime import date
        from data.fetcher import PolygonClient

        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        today = date.today()
        payload = {
            "results": [{"filing_date": today.isoformat()}],
            "next_url": None,
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload
        mock_session.get.return_value = mock_resp

        client = PolygonClient()
        assert client.has_earnings_this_week("AAPL") is True

    @patch("data.fetcher.requests.Session")
    def test_has_earnings_this_week_false(self, mock_session_cls):
        from data.fetcher import PolygonClient

        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        payload = {"results": [], "next_url": None}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = payload
        mock_session.get.return_value = mock_resp

        client = PolygonClient()
        assert client.has_earnings_this_week("AAPL") is False

    @patch("data.fetcher.requests.Session")
    def test_rate_limit_retry(self, mock_session_cls):
        from data.fetcher import PolygonClient
        from datetime import date

        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session

        rate_resp = MagicMock()
        rate_resp.status_code = 429

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = {"results": []}

        mock_session.get.side_effect = [rate_resp, ok_resp]

        client = PolygonClient()
        with patch("data.fetcher.time.sleep"):
            df = client.get_aggs("AAPL", date(2024, 1, 1), date(2024, 1, 5))
        assert df.empty


# ---------------------------------------------------------------------------
# CCXT client (mocked)
# ---------------------------------------------------------------------------

class TestCryptoFetcher:
    @patch("data.fetcher.ccxt.binance")
    def test_get_ohlcv_returns_dataframe(self, mock_binance_cls):
        mock_exchange = MagicMock()
        mock_binance_cls.return_value = mock_exchange

        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        raw = [
            [now_ms - i * 3_600_000, 50000 + i, 51000 + i, 49000 + i, 50500 + i, 1.5]
            for i in range(10)
        ]
        mock_exchange.fetch_ohlcv.return_value = raw

        from data.fetcher import CryptoFetcher
        fetcher = CryptoFetcher()
        df = fetcher.get_ohlcv("BTC/USDT")
        assert not df.empty
        assert "close" in df.columns
        assert len(df) == 10

    @patch("data.fetcher.ccxt.binance")
    def test_get_ohlcv_empty_on_error(self, mock_binance_cls):
        import ccxt as ccxt_lib
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.side_effect = ccxt_lib.NetworkError("timeout")
        mock_binance_cls.return_value = mock_exchange

        from data.fetcher import CryptoFetcher
        fetcher = CryptoFetcher()
        df = fetcher.get_ohlcv("BTC/USDT")
        assert df.empty

    @patch("data.fetcher.ccxt.binance")
    def test_get_all_symbols(self, mock_binance_cls):
        mock_exchange = MagicMock()
        mock_binance_cls.return_value = mock_exchange

        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        raw = [
            [now_ms - i * 3_600_000, 100 + i, 110 + i, 90 + i, 105 + i, 2.0]
            for i in range(60)
        ]
        mock_exchange.fetch_ohlcv.return_value = raw

        from data.fetcher import CryptoFetcher, CRYPTO_SYMBOLS
        fetcher = CryptoFetcher()
        result = fetcher.get_all_symbols()
        assert len(result) == len(CRYPTO_SYMBOLS)
        for sym in CRYPTO_SYMBOLS:
            assert sym in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_bar_df(self):
        df = _make_ohlcv(1)
        result = validate_stock_df("X", df)
        assert not result.valid

    def test_all_nan_close(self):
        df = _make_ohlcv(260)
        df["close"] = np.nan
        result = validate_stock_df("X", df)
        assert not result.valid

    def test_zero_volume_warns(self):
        df = add_features(_make_ohlcv(260))
        df = df.copy()
        df.loc[df.index[-10:], "volume"] = 0
        result = validate_stock_df("AAPL", df)
        assert result.warnings

    def test_duplicate_index_does_not_crash(self):
        df = _make_ohlcv(260)
        df = pd.concat([df, df.iloc[:5]])
        df = add_features(df.sort_index())
        result = validate_stock_df("X", df)
        assert isinstance(result.valid, bool)