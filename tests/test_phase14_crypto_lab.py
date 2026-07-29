"""tests/test_phase14_crypto_lab.py - crypto strategy backtests."""

from __future__ import annotations

import os

for _k, _v in {"ALPACA_API_KEY": "test", "ALPACA_SECRET_KEY": "test",
               "STARTING_CAPITAL": "10000"}.items():
    os.environ.setdefault(_k, _v)

import numpy as np
import pandas as pd
import pytest

from backtest.crypto_lab import (
    CryptoLabConfig, backtest_crypto_momentum, backtest_hold, backtest_regime,
)
from data.fetcher import add_features


def _coin(seed: int, n: int = 500, drift: float = 0.001) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    close = 100 * np.cumprod(1 + drift + rng.normal(0, 0.02, n))
    op = pd.Series(close).shift(1).fillna(close[0]).to_numpy()
    return add_features(pd.DataFrame(
        {"open": op, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": rng.uniform(1e5, 2e5, n)}, index=idx))


def _universe() -> dict:
    return {"BTC/USD": _coin(1), "ETH/USD": _coin(2), "SOL/USD": _coin(3),
            "DOGE/USD": _coin(4, drift=-0.0005)}


def test_all_crypto_strategies_produce_metrics():
    uni = _universe()
    cfg = CryptoLabConfig(warmup_bars=210)
    for name, res in (("hold", backtest_hold(uni, cfg)),
                      ("regime", backtest_regime(uni, cfg)),
                      ("momentum", backtest_crypto_momentum(uni, cfg))):
        assert res.metrics, f"{name} produced nothing"
        assert (res.equity > 0).all()


def test_hold_tracks_btc():
    uni = _universe()
    cfg = CryptoLabConfig(warmup_bars=210)
    res = backtest_hold(uni, cfg)
    assert res.metrics["trades"] == 1
    assert abs(res.metrics["total_return_pct"]
               - res.metrics["benchmark_return_pct"]) < 3.0


def test_crypto_costs_are_brutal_by_default():
    """Alpaca crypto is ~25bps per side - an order of magnitude worse than
    equities. A strategy that only survives at equity costs is not viable."""
    assert CryptoLabConfig().cost_bps == 25.0
    uni = _universe()
    cheap = backtest_regime(uni, CryptoLabConfig(warmup_bars=210, cost_bps=1.0))
    dear = backtest_regime(uni, CryptoLabConfig(warmup_bars=210, cost_bps=100.0))
    if cheap.metrics.get("trades", 0) > 0:
        assert dear.metrics["total_costs"] > cheap.metrics["total_costs"]
        assert (dear.metrics["total_return_pct"]
                <= cheap.metrics["total_return_pct"] + 1e-9)


def test_regime_uses_the_real_live_predicates():
    """This must test the SHIPPED logic, not a lookalike."""
    import backtest.crypto_lab as lab
    import strategies.crypto_24h as live
    assert lab.above_ema200 is live.above_ema200
    assert lab.ema_stacked_bullish is live.ema_stacked_bullish
    assert lab.macd_positive is live.macd_positive


def test_regime_goes_flat_and_holds_cash():
    """A downtrending coin must not be held; equity should not track it."""
    down = {"BTC/USD": _coin(9, drift=-0.004)}
    res = backtest_regime(down, CryptoLabConfig(warmup_bars=210))
    hold = backtest_hold(down, CryptoLabConfig(warmup_bars=210))
    assert res.metrics["total_return_pct"] > hold.metrics["total_return_pct"]


def test_momentum_never_holds_a_downtrend():
    uni = {"A": _coin(11, drift=-0.003), "B": _coin(12, drift=-0.003)}
    res = backtest_crypto_momentum(uni, CryptoLabConfig(warmup_bars=210, top_n=2))
    # all candidates negative -> stays in cash -> roughly flat, not -50%
    assert res.metrics["total_return_pct"] > -15.0


def test_insufficient_history_is_clean():
    tiny = {"BTC/USD": _coin(1, n=30)}
    cfg = CryptoLabConfig(warmup_bars=210)
    assert backtest_hold(tiny, cfg).equity.empty
    assert backtest_regime(tiny, cfg).equity.empty
    assert backtest_crypto_momentum(tiny, cfg).equity.empty
