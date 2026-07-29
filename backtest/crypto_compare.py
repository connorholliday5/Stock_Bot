"""
backtest/crypto_compare.py
Compare crypto strategies - including BTC-only vs a wider universe.

    python -m backtest.crypto_compare --years 2
    python -m backtest.crypto_compare --years 3 --timeframe 1d

Answers two questions the live bot cannot: does the regime strategy beat
simply holding BTC, and does widening the coin universe help or hurt?
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

WIDE = ["BTC/USD", "ETH/USD", "SOL/USD", "LTC/USD",
        "DOGE/USD", "LINK/USD", "AVAX/USD"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare crypto strategies")
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--cost-bps", type=float, default=25.0,
                    help="per side; Alpaca crypto taker is ~25bps")
    ap.add_argument("--timeframe", type=str, default="4h")
    ap.add_argument("--top-n", type=int, default=3)
    args = ap.parse_args()

    from backtest.crypto_lab import (
        CryptoLabConfig, backtest_crypto_momentum, backtest_hold, backtest_regime,
    )
    from data.alpaca_data import fetch_crypto_universe_alpaca

    hours = {"1h": 1, "4h": 4, "1d": 24}.get(args.timeframe, 4)
    bars = int(args.years * 365 * 24 / hours) + 300
    logger.info("fetching {} {} bars per symbol...", bars, args.timeframe)
    universe = fetch_crypto_universe_alpaca(symbols=WIDE, timeframe=args.timeframe,
                                            limit=bars)
    if not universe:
        logger.error("no crypto data fetched")
        return 1
    logger.info("fetched {}: {}", len(universe), ", ".join(universe))

    cfg = CryptoLabConfig(initial_capital=args.capital, cost_bps=args.cost_bps,
                          top_n=args.top_n)
    btc_only = {k: v for k, v in universe.items() if k == "BTC/USD"}

    rows = []
    rows.append(("hold BTC", backtest_hold(universe, cfg).metrics))
    if btc_only:
        rows.append(("regime BTC-only", backtest_regime(btc_only, cfg).metrics))
    rows.append((f"regime wide ({len(universe)})", backtest_regime(universe, cfg).metrics))
    rows.append((f"momentum top{args.top_n}", backtest_crypto_momentum(universe, cfg).metrics))

    rows = [(n, m) for n, m in rows if m]
    if not rows:
        logger.error("no results")
        return 1

    bench = rows[0][1].get("benchmark_return_pct", float("nan"))
    print()
    print("=" * 86)
    print(f"{'strategy':<24} {'return':>10} {'CAGR':>9} {'maxDD':>9} "
          f"{'Sharpe':>8} {'trades':>7} {'costs':>10}")
    print("-" * 86)
    for name, m in sorted(rows, key=lambda r: -r[1].get("total_return_pct", 0)):
        print(f"{name:<24} {m.get('total_return_pct', 0):>9.2f}% "
              f"{m.get('cagr_pct', 0):>8.2f}% {m.get('max_drawdown_pct', 0):>8.2f}% "
              f"{m.get('sharpe', 0):>8.2f} {m.get('trades', 0):>7} "
              f"${m.get('total_costs', 0):>9,.0f}")
    print("-" * 86)
    print(f"{'BTC buy & hold':<24} {bench:>9.2f}%   <- the bar to clear")
    print("=" * 86)
    print(f"Costs modeled at {args.cost_bps:.0f}bps per side (Alpaca crypto taker).")
    print("Crypto has no survivorship bias here - these coins all still trade.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
