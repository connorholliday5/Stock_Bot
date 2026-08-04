"""
backtest/compare.py
Run every strategy over the SAME history and print them side by side.

    python -m backtest.compare --years 3
    python -m backtest.compare --years 5 --top-n 15

Same data, same costs, same accounting - so the comparison is apples to
apples and the honest question ("does any of this beat buying the index?")
gets a direct answer.

Every row also carries a DEFLATED Sharpe. Raw Sharpe is meaningless without
the trial count: the best of N zero-edge strategies scores well above zero by
construction, so "we found one with Sharpe 0.8" is only interesting relative
to what N tries would have produced by luck. The count comes from
RESEARCH_LOG.md - the whole research history, not the size of this run's
grid - so the bar rises every time we test something new. That is correct and
uncomfortable, which is the point.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from loguru import logger

RESEARCH_LOG = Path(__file__).resolve().parent.parent / "RESEARCH_LOG.md"


def count_trials(path: Path = RESEARCH_LOG, default: int = 20) -> tuple[int, str]:
    """Number of strategy variants ever evaluated, from RESEARCH_LOG.md.

    Counts numbered rows in the trials table. If the log is missing we do NOT
    fall back to 1 - assuming a single trial is the most flattering possible
    assumption and would silently inflate every DSR on the page.
    """
    try:
        rows = [ln for ln in path.read_text().splitlines()
                if re.match(r"^\|\s*\d+\s*\|", ln)]
        if rows:
            return len(rows), path.name
    except Exception as exc:
        logger.warning("could not read {}: {}", path.name, exc)
    logger.warning("RESEARCH_LOG.md unreadable; assuming N={} trials", default)
    return default, "assumed"


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare strategies on identical history")
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--top-n", type=int, default=10, help="names held by momentum")
    ap.add_argument("--ma", type=int, default=200, help="trend-filter MA window")
    ap.add_argument("--include-rotation", action="store_true",
                    help="also run the bot's current weekly rotation (slow)")
    ap.add_argument("--weightings", type=str, default="equal,rank,score",
                    help="how momentum splits capital across its top N: equal "
                         "(1/N), rank (linear decay to the best name), score "
                         "(proportional to momentum). Comma separated.")
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

    from dataclasses import replace as _replace

    rows = []
    for name, fn in STRATEGIES.items():
        logger.info("running {}...", name)
        try:
            res = fn(universe, cfg)
        except Exception as exc:
            logger.warning("{} failed: {}", name, exc)
            continue
        if res.metrics:
            rows.append((name, res.metrics, res))

    # Does piling capital into the strongest name beat splitting evenly?
    # Momentum RANK carries information; the magnitude of a momentum score is
    # a far weaker predictor, and the top-scoring name is usually the most
    # extended - so this is a question to measure, not assume.
    for mode in [w.strip() for w in args.weightings.split(",") if w.strip()]:
        if mode == "equal":
            continue                       # already covered by momentum_12_1
        logger.info("running momentum ({} weighted)...", mode)
        try:
            res = STRATEGIES["momentum_12_1"](universe, _replace(cfg, weighting=mode))
        except Exception as exc:
            logger.warning("momentum {} failed: {}", mode, exc)
            continue
        if res.metrics:
            rows.append((f"momentum [{mode}]", res.metrics, res))

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
            rows.append((f"hold {sym}", res.metrics, res))

    if args.include_rotation:
        logger.info("running current weekly rotation...")
        res = run_backtest(universe, BacktestConfig(
            initial_capital=args.capital, cost_bps=args.cost_bps, keep_rank=100))
        if res.metrics:
            rows.append(("weekly_rotation*", res.metrics, res))

    if not rows:
        logger.error("no strategy produced results")
        return 1

    from backtest.validate import deflated_sharpe_ratio, expected_max_sharpe
    from data.index_membership import is_available as pit_available

    n_trials, source = count_trials()
    bench = rows[0][1].get("benchmark_return_pct", float("nan"))

    print()
    print("=" * 104)
    print(f"{'strategy':<20} {'return':>10} {'CAGR':>9} {'maxDD':>9} "
          f"{'Sharpe':>8} {'DSR':>7} {'trades':>8} {'costs':>10}")
    print("-" * 104)
    for name, m, res in sorted(rows, key=lambda r: -r[1].get("total_return_pct", 0)):
        dsr_txt = "   n/a"
        try:
            daily = res.equity.pct_change().dropna()
            d = deflated_sharpe_ratio(daily, n_trials=n_trials)
            dsr_txt = f"{d.deflated_sharpe:>6.3f}"
        except Exception:
            pass
        print(f"{name:<20} {m.get('total_return_pct', 0):>9.2f}% "
              f"{m.get('cagr_pct', 0):>8.2f}% {m.get('max_drawdown_pct', 0):>8.2f}% "
              f"{m.get('sharpe', 0):>8.2f} {dsr_txt:>7} {m.get('trades', 0):>8} "
              f"${m.get('total_costs', 0):>9,.0f}")
    print("-" * 104)
    print(f"{'SPY buy & hold':<20} {bench:>9.2f}%   <- the bar every strategy must clear")
    print("=" * 104)

    floor = expected_max_sharpe(n_trials, args.years)
    print(f"DSR = probability the edge is real given N={n_trials} trials "
          f"(source: {source}).")
    print(f"     A zero-edge strategy, best of {n_trials} tries on {args.years:g}y, "
          f"is expected to score Sharpe {floor:.2f} by luck alone -")
    print(f"     so any raw Sharpe below {floor:.2f} is not evidence of anything. "
          f"DSR >= 0.95 is the gate.")
    print("* weekly_rotation is the bot's current live strategy.")
    if pit_available():
        print("NOTE: candidates are filtered to point-in-time index membership, but")
        print("      Alpaca has no bars for delisted names, so survivorship is only")
        print("      PARTIALLY corrected. Absolute returns remain optimistic.")
    else:
        print("NOTE: membership data unavailable - universe is TODAY's index")
        print("      membership (full survivorship bias). Returns are optimistic.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
