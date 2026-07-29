"""
backtest/compare.py
Run every strategy over the SAME history and print them side by side.

    python -m backtest.compare --years 3
    python -m backtest.compare --years 5 --top-n 15

Same data, same costs, same accounting - so the comparison is apples to
apples and the honest question ("does any of this beat buying the index?")
gets a direct answer.
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare strategies on identical history")
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--top-n", type=int, default=10, help="names held by momentum")
    ap.add_argument("--ma", type=int, default=200, help="trend-filter MA window")
    ap.add_argument("--include-rotation", action="store_true",
                    help="also run the bot's current weekly rotation (slow)")
    ap.add_argument("--vs", type=str, default="MTUM,QQQ",
                    help="also buy-and-hold these REAL tradeable funds. MTUM is a "
                         "live momentum ETF: its record includes every name that "
                         "blew up and left the index, so it is the reality check "
                         "on a survivorship-biased momentum backtest. Empty to skip.")
    args = ap.parse_args()

    from backtest.lab import LabConfig, STRATEGIES
    from backtest.engine import BacktestConfig, run_backtest
    from data.alpaca_data import default_stock_universe, fetch_stock_universe_alpaca

    cfg = LabConfig(initial_capital=args.capital, cost_bps=args.cost_bps,
                    top_n=args.top_n, ma_window=args.ma)

    symbols = default_stock_universe()
    extras = [s.strip().upper() for s in args.vs.split(",") if s.strip()]
    for s in [cfg.benchmark] + extras:
        if s not in symbols:
            symbols.append(s)

    lookback = int(args.years * 365) + 500        # + warmup for 12m momentum
    logger.info("fetching {} days...", lookback)
    universe = fetch_stock_universe_alpaca(lookback_days=lookback, tickers=symbols)
    if not universe:
        logger.error("no data - check ALPACA_API_KEY / ALPACA_SECRET_KEY")
        return 1
    logger.info("fetched {} symbols", len(universe))

    rows = []
    for name, fn in STRATEGIES.items():
        logger.info("running {}...", name)
        try:
            res = fn(universe, cfg)
        except Exception as exc:
            logger.warning("{} failed: {}", name, exc)
            continue
        if res.metrics:
            rows.append((name, res.metrics))

    # Buy-and-hold of REAL funds. These carry no survivorship bias - every
    # constituent that collapsed and was removed is already in their record -
    # so they are the honest floor a stock-picking backtest must clear.
    from backtest.lab import backtest_buy_hold
    for sym in extras:
        if sym not in universe:
            logger.warning("{} not available; skipping", sym)
            continue
        try:
            res = backtest_buy_hold(universe, cfg, symbol=sym)
        except Exception as exc:
            logger.warning("{} buy-hold failed: {}", sym, exc)
            continue
        if res.metrics:
            rows.append((f"hold {sym}", res.metrics))

    if args.include_rotation:
        logger.info("running current weekly rotation...")
        res = run_backtest(universe, BacktestConfig(
            initial_capital=args.capital, cost_bps=args.cost_bps, keep_rank=100))
        if res.metrics:
            rows.append(("weekly_rotation*", res.metrics))

    if not rows:
        logger.error("no strategy produced results")
        return 1

    bench = rows[0][1].get("benchmark_return_pct", float("nan"))
    print()
    print("=" * 86)
    print(f"{'strategy':<20} {'return':>10} {'CAGR':>9} {'maxDD':>9} "
          f"{'Sharpe':>8} {'trades':>8} {'costs':>10}")
    print("-" * 86)
    for name, m in sorted(rows, key=lambda r: -r[1].get("total_return_pct", 0)):
        print(f"{name:<20} {m.get('total_return_pct', 0):>9.2f}% "
              f"{m.get('cagr_pct', 0):>8.2f}% {m.get('max_drawdown_pct', 0):>8.2f}% "
              f"{m.get('sharpe', 0):>8.2f} {m.get('trades', 0):>8} "
              f"${m.get('total_costs', 0):>9,.0f}")
    print("-" * 86)
    print(f"{'SPY buy & hold':<20} {bench:>9.2f}%   <- the bar every strategy must clear")
    print("=" * 86)
    print("* weekly_rotation is the bot's current live strategy.")
    print("NOTE: universe is TODAY's index membership (survivorship bias);")
    print("      absolute returns are optimistic for every row alike.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
