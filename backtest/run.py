"""
backtest/run.py
CLI: replay the live strategy over real history.

    python -m backtest.run --years 3
    python -m backtest.run --years 5 --top-n 5 --cost-bps 15
    python -m backtest.run --years 3 --tickers AAPL,MSFT,NVDA,JPM,XOM

Data comes from the same Alpaca fetcher the bot uses, so a run needs the
ALPACA_* keys in .env (the data endpoint is free).
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from backtest.engine import BacktestConfig, run_backtest


def main() -> int:
    ap = argparse.ArgumentParser(description="Backtest the weekly rotation strategy")
    ap.add_argument("--years", type=float, default=3.0, help="history to replay")
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--top-n", type=int, default=8, help="max concurrent positions")
    ap.add_argument("--keep-rank", type=int, default=20, help="rotation hold rank")
    ap.add_argument("--stop", type=float, default=0.05, help="stop loss fraction")
    ap.add_argument("--cost-bps", type=float, default=10.0, help="cost per side, bps")
    ap.add_argument("--tickers", type=str, default="", help="comma list (default: S&P 500)")
    ap.add_argument("--csv", type=str, default="", help="write equity curve to this path")
    args = ap.parse_args()

    lookback_days = int(args.years * 365) + 400          # + warmup for SMA200
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] or None

    from data.alpaca_data import fetch_stock_universe_alpaca
    from backtest.engine import BacktestConfig as _Cfg

    symbols = list(tickers) if tickers else None
    if symbols is not None and _Cfg().benchmark not in symbols:
        symbols.append(_Cfg().benchmark)                 # benchmark needs bars too

    logger.info("fetching {} days of history...", lookback_days)
    universe = fetch_stock_universe_alpaca(lookback_days=lookback_days, tickers=symbols)
    if not universe:
        logger.error("no data fetched - check ALPACA_API_KEY / ALPACA_SECRET_KEY")
        return 1
    logger.info("fetched {} symbols", len(universe))

    cfg = BacktestConfig(
        initial_capital=args.capital, top_n=args.top_n, keep_rank=args.keep_rank,
        stop_loss_pct=args.stop, cost_bps=args.cost_bps,
    )
    result = run_backtest(universe, cfg)
    if not result.metrics:
        logger.error("backtest produced no results (insufficient history)")
        return 1

    print()
    print("=" * 62)
    print(result.summary())
    print("=" * 62)
    print("NOTE: universe is TODAY's index membership, so delisted names are")
    print("absent (survivorship bias) - treat absolute returns as optimistic.")

    if args.csv:
        result.equity.to_csv(args.csv, header=["equity"])
        logger.info("equity curve -> {}", args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
