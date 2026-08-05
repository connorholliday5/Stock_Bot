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


def _confirm_sweep(spec: str, universe, cfg, btc_only, ppy: int, n_trials: int) -> None:
    """How much of the cost drag is whipsaw?

    The 2y run charged the regime gate 27% of capital in fees across 112
    trades, while cutting drawdown roughly in half. That combination says the
    signal is doing its job and the TRADING FREQUENCY is what loses the money.
    Requiring the signal to hold for N bars before acting tests that directly:
    if trades and costs fall while return improves, the diagnosis was right.

    Printed as a curve, not a best pick. One good value proves nothing - a
    real effect improves smoothly across neighbouring values, and a lone
    spike surrounded by bad ones is the shape of a fitted parameter.
    """
    from dataclasses import replace

    from backtest.crypto_lab import backtest_regime
    from backtest.validate import deflated_sharpe_ratio

    try:
        values = [int(v) for v in spec.split(",") if v.strip()]
    except ValueError:
        print(f"\nbad --confirm-sweep value: {spec!r}")
        return
    if not values:
        return

    uni = btc_only or universe
    label = "BTC-only" if btc_only else f"wide ({len(universe)})"
    print()
    print("=" * 95)
    print(f"CONFIRM-BARS SWEEP on the regime gate, {label}")
    print("A bar is 4h, so confirm=6 means the signal must hold a full day.")
    print("-" * 95)
    print(f"{'confirm':>8} {'return':>10} {'CAGR':>9} {'maxDD':>9} "
          f"{'Sharpe':>8} {'DSR':>7} {'trades':>8} {'costs':>10}")
    print("-" * 95)
    for v in values:
        res = backtest_regime(uni, replace(cfg, confirm_bars=v))
        m = res.metrics
        if not m:
            print(f"{v:>8}  (no result)")
            continue
        dsr_txt = "   n/a"
        try:
            d = deflated_sharpe_ratio(res.equity.pct_change().dropna(),
                                      n_trials=n_trials, periods_per_year=ppy)
            dsr_txt = f"{d.deflated_sharpe:>6.3f}"
        except Exception:
            pass
        print(f"{v:>8} {m.get('total_return_pct', 0):>9.2f}% "
              f"{m.get('cagr_pct', 0):>8.2f}% {m.get('max_drawdown_pct', 0):>8.2f}% "
              f"{m.get('sharpe', 0):>8.2f} {dsr_txt:>7} {m.get('trades', 0):>8} "
              f"${m.get('total_costs', 0):>9,.0f}")
    print("=" * 95)
    print("Reading this: falling trades/costs with rising return means whipsaw")
    print("was the problem. Flat or worse return means the signal itself is")
    print("weak and slowing it down cannot save it. Either way the DSR column")
    print("still has to clear 0.95, and picking the best row here is a new")
    print("trial that belongs in RESEARCH_LOG.md.")


def _stress(rows, universe, cfg) -> None:
    """Stress the best strategy that actually makes decisions.

    Buy-and-hold is excluded deliberately: it has no parameters to plateau
    and one trade to drop, so stressing it measures nothing. The question is
    whether the ACTIVE strategy's margin over hold survives worse costs -
    at 25bps per side a strategy that rebalances often can beat hold on
    paper and lose to it in production.
    """
    from backtest.crypto_lab import backtest_crypto_momentum, backtest_regime
    from backtest.stress import print_report, stress_strategy

    fns = {"regime": backtest_regime, "momentum": backtest_crypto_momentum}
    pick = next((n for n, _, _ in sorted(rows, key=lambda r: -r[1].get("total_return_pct", 0))
                 if any(k in n for k in fns)), None)
    if pick is None:
        print("\nno active strategy to stress (hold-only results).")
        return

    key = "regime" if "regime" in pick else "momentum"
    uni = universe
    if "BTC-only" in pick:
        uni = {k: v for k, v in universe.items() if k == "BTC/USD"}

    # Sweep whatever that strategy actually has to tune. top_n is meaningless
    # on a single-coin universe; confirm_bars is the regime gate's only knob
    # and the one that would ship, so it has to survive a plateau check.
    if key == "momentum":
        params = ("top_n",) if len(uni) > 1 else ()
    else:
        params = ("confirm_bars",)
    print(f"\nstressing best active strategy: {pick}")
    print_report(stress_strategy(fns[key], uni, cfg, name=pick, params=params))


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare crypto strategies")
    ap.add_argument("--years", type=float, default=2.0)
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--cost-bps", type=float, default=25.0,
                    help="per side; Alpaca crypto taker is ~25bps")
    ap.add_argument("--timeframe", type=str, default="4h")
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--stress", action="store_true",
                    help="run cost/robustness stress on the best strategy")
    ap.add_argument("--confirm-sweep", type=str, default="",
                    help="comma-separated confirm_bars values to sweep on the "
                         "regime gate, e.g. 1,2,3,4,6,8")
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
    rows.append(("hold BTC", backtest_hold(universe, cfg)))
    if btc_only:
        rows.append(("regime BTC-only", backtest_regime(btc_only, cfg)))
    rows.append((f"regime wide ({len(universe)})", backtest_regime(universe, cfg)))
    rows.append((f"momentum top{args.top_n}", backtest_crypto_momentum(universe, cfg)))

    rows = [(n, r.metrics, r) for n, r in rows if r.metrics]
    if not rows:
        logger.error("no results")
        return 1

    from backtest.compare import count_trials
    from backtest.validate import deflated_sharpe_ratio, expected_max_sharpe

    # Crypto trades 365 days a year on 4h bars, not 252 days on daily bars.
    # Annualizing with the equity default would multiply every Sharpe here by
    # sqrt(2190/252) ~ 2.9 and turn noise into a headline.
    ppy = int(round(365 * 24 / hours))
    n_trials, source = count_trials()

    bench = rows[0][1].get("benchmark_return_pct", float("nan"))
    print()
    print("=" * 95)
    print(f"{'strategy':<24} {'return':>10} {'CAGR':>9} {'maxDD':>9} "
          f"{'Sharpe':>8} {'DSR':>7} {'trades':>7} {'costs':>10}")
    print("-" * 95)
    for name, m, res in sorted(rows, key=lambda r: -r[1].get("total_return_pct", 0)):
        dsr_txt = "   n/a"
        try:
            bar_returns = res.equity.pct_change().dropna()
            d = deflated_sharpe_ratio(bar_returns, n_trials=n_trials,
                                      periods_per_year=ppy)
            dsr_txt = f"{d.deflated_sharpe:>6.3f}"
        except Exception:
            pass
        print(f"{name:<24} {m.get('total_return_pct', 0):>9.2f}% "
              f"{m.get('cagr_pct', 0):>8.2f}% {m.get('max_drawdown_pct', 0):>8.2f}% "
              f"{m.get('sharpe', 0):>8.2f} {dsr_txt:>7} {m.get('trades', 0):>7} "
              f"${m.get('total_costs', 0):>9,.0f}")
    print("-" * 95)
    print(f"{'BTC buy & hold':<24} {bench:>9.2f}%   <- the bar to clear")
    print("=" * 95)

    floor = expected_max_sharpe(n_trials, args.years)
    print(f"Costs modeled at {args.cost_bps:.0f}bps per side (Alpaca crypto taker).")
    print(f"DSR = probability the edge is real given N={n_trials} trials "
          f"(source: {source}).")
    print(f"     A zero-edge strategy, best of {n_trials} tries on {args.years:g}y, "
          f"is expected to score Sharpe {floor:.2f} by luck alone -")
    print(f"     so any raw Sharpe below {floor:.2f} is not evidence of anything. "
          f"DSR >= 0.95 is the gate.")
    print("Crypto has no survivorship bias here - these coins all still trade.")
    print("NOTE: no delisted-coin history either, so a universe-wide crypto")
    print("      winter that killed alts is under-represented in this window.")

    if args.confirm_sweep:
        _confirm_sweep(args.confirm_sweep, universe, cfg, btc_only, ppy, n_trials)

    if args.stress:
        _stress(rows, universe, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
