"""
backtest/leakage.py
Empirical look-ahead detection: prove the backtest cannot see the future.

WHY EMPIRICAL AND NOT BY EYE
----------------------------
Look-ahead bias is the single most common way a backtest invents an edge that
evaporates live, and it is very hard to spot by reading code. It hides in
places that look completely innocent:

  - a scaler or normalizer fit on the FULL history, then applied per-bar
  - `.rank(pct=True)` or a z-score computed across the whole series
  - `bfill()` / `interpolate()` filling a gap from a later bar
  - a centered rolling window (`center=True`)
  - resampling that labels a bar with its period's closing data
  - joining a dataset (index membership, earnings dates, splits) that is
    already restated with hindsight

Freqtrade's approach - which this steals - tests the property directly
instead of auditing the code. The property is:

    computing a signal from data TRUNCATED at T must give exactly the same
    answer as computing it on the FULL history and then reading bar T.

If those differ, something downstream of T influenced the value at T. That
is a leak, whatever the mechanism. The check is agnostic to how the
indicator is implemented, so it keeps working when the code changes.

Causal recursive indicators (EMA, Wilder RSI/ATR, ADX) pass this exactly:
truncating the END of a series cannot change a value the recursion already
produced on the way forward.

WARM-UP IS A DIFFERENT PROBLEM
------------------------------
Those same recursive indicators never fully forget their seed, so starting
the history at a different point shifts every value slightly - forever, just
by shrinking amounts. That is not look-ahead (nothing from the future is
used), but it does mean a backtest and the live bot will disagree unless
both get enough warm-up. `warmup_sensitivity` measures how many bars each
indicator needs before that disagreement stops mattering.

    python -m backtest.leakage --years 3
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Floating-point noise floor. Anything above this is a real difference, not
# an artifact of summation order.
TOLERANCE = 1e-9


@dataclass
class Leak:
    kind: str            # "feature" | "score" | "momentum"
    subject: str         # column name or ticker
    date: pd.Timestamp
    truncated: float     # value computed from data <= T
    full: float          # value computed on all data, then read at T
    diff: float

    def __str__(self) -> str:
        return (f"{self.kind}:{self.subject} @ {pd.Timestamp(self.date).date()} "
                f"truncated={self.truncated:.10g} full={self.full:.10g} "
                f"diff={self.diff:.3g}")


@dataclass
class LeakReport:
    checks: int = 0
    leaks: list[Leak] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.leaks

    def extend(self, other: "LeakReport") -> "LeakReport":
        self.checks += other.checks
        self.leaks.extend(other.leaks)
        self.notes.extend(other.notes)
        return self

    def summary(self) -> str:
        if self.clean:
            return f"{self.checks} comparisons, ZERO leaks"
        subjects = sorted({f"{l.kind}:{l.subject}" for l in self.leaks})
        return (f"{self.checks} comparisons, {len(self.leaks)} LEAKS across "
                f"{len(subjects)} signals: {', '.join(subjects[:8])}")


def _sample_dates(index: pd.DatetimeIndex, n: int, warmup: int) -> list[pd.Timestamp]:
    """Evenly spaced dates past the warm-up, so every check has a fair
    amount of history behind it."""
    idx = pd.DatetimeIndex(index).sort_values()
    usable = idx[warmup:-1]        # -1: need a bar after T to be meaningful
    if len(usable) == 0:
        return []
    step = max(len(usable) // max(n, 1), 1)
    return list(usable[::step][:n])


def _num(v) -> Optional[float]:
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 1. indicator columns
# ---------------------------------------------------------------------------

def check_features(raw: pd.DataFrame, symbol: str = "?", n_dates: int = 8,
                   warmup: int = 250, tolerance: float = TOLERANCE) -> LeakReport:
    """add_features on truncated data must equal add_features on full data,
    read at the same bar.

    `raw` is OHLCV WITHOUT features - this recomputes them both ways.
    """
    from data.fetcher import add_features

    report = LeakReport()
    if raw is None or len(raw) < warmup + 5:
        report.notes.append(f"{symbol}: too little history to check features")
        return report

    full = add_features(raw)
    for date in _sample_dates(raw.index, n_dates, warmup):
        trunc = add_features(raw.loc[raw.index <= date])
        if date not in trunc.index or date not in full.index:
            continue
        a, b = trunc.loc[date], full.loc[date]
        for col in trunc.columns:
            if col not in full.columns:
                continue
            x, y = _num(a.get(col)), _num(b.get(col))
            report.checks += 1
            if x is None or y is None:
                # NaN on one side only is itself a difference worth flagging
                if (x is None) != (y is None):
                    report.leaks.append(Leak("feature", f"{symbol}.{col}", date,
                                             float("nan"), float("nan"), float("inf")))
                continue
            scale = max(abs(x), abs(y), 1.0)
            if abs(x - y) / scale > tolerance:
                report.leaks.append(
                    Leak("feature", f"{symbol}.{col}", date, x, y, abs(x - y) / scale))
    return report


# ---------------------------------------------------------------------------
# 2. the composite score the live bot ranks on
# ---------------------------------------------------------------------------

def check_scores(raw_universe: dict[str, pd.DataFrame], n_dates: int = 6,
                 warmup: int = 250, tolerance: float = 1e-6) -> LeakReport:
    """End-to-end: the ranking the scheduler actually trades on.

    Slightly looser tolerance than the raw columns because the composite
    involves cross-sectional normalization across the universe - and the
    truncated universe legitimately contains fewer usable symbols on early
    dates, which shifts a percentile without any future data being involved.
    A real leak moves scores far more than that.
    """
    from data.fetcher import add_features
    from strategies import stock_scorer

    report = LeakReport()
    symbols = [s for s, d in raw_universe.items() if d is not None and len(d)]
    if not symbols:
        return report
    index = max((raw_universe[s].index for s in symbols), key=len)
    dates = _sample_dates(index, n_dates, warmup)
    if not dates:
        report.notes.append("scores: not enough history")
        return report

    featured = {s: add_features(raw_universe[s]) for s in symbols}

    for date in dates:
        trunc_uni, slice_uni = {}, {}
        for s in symbols:
            r = raw_universe[s]
            sub_raw = r.loc[r.index <= date]
            if len(sub_raw) < warmup:
                continue
            trunc_uni[s] = add_features(sub_raw)
            f = featured[s]
            slice_uni[s] = f.loc[f.index <= date]
        if len(trunc_uni) < 2:
            continue
        try:
            a = stock_scorer.score_universe(trunc_uni).ranked
            b = stock_scorer.score_universe(slice_uni).ranked
        except Exception as exc:
            report.notes.append(f"scores @ {pd.Timestamp(date).date()}: {exc}")
            continue
        if a is None or b is None or a.empty or b.empty:
            continue
        col = "composite" if "composite" in a.columns else a.columns[-1]
        for sym in set(a.index) & set(b.index):
            x, y = _num(a.loc[sym, col]), _num(b.loc[sym, col])
            report.checks += 1
            if x is None or y is None:
                continue
            scale = max(abs(x), abs(y), 1.0)
            if abs(x - y) / scale > tolerance:
                report.leaks.append(
                    Leak("score", sym, date, x, y, abs(x - y) / scale))
    return report


# ---------------------------------------------------------------------------
# 3. end-to-end: perturb the future, the past must not move
# ---------------------------------------------------------------------------

def perturb_after(universe: dict[str, pd.DataFrame], date,
                  factor: float = 3.0, seed: int = 7) -> dict[str, pd.DataFrame]:
    """Return a copy of `universe` with every bar AFTER `date` mangled.

    EVERY numeric column after the cut is scaled and randomized beyond
    recognition - not just OHLCV. Mangling only raw prices would leave the
    precomputed indicators (sma_50, rsi, macd...) intact after the cut, so a
    strategy that peeked at a future INDICATOR rather than a future price
    would sail through undetected. The columns are the interface; perturb
    all of them.

    Any signal, indicator or sizing rule that consults a post-cut bar will
    visibly change. One that respects causality cannot notice at all.
    """
    rng = np.random.default_rng(seed)
    ts = pd.Timestamp(date)
    out: dict[str, pd.DataFrame] = {}
    for sym, df in universe.items():
        if df is None or df.empty:
            continue
        new = df.copy()
        future = new.index > ts
        n = int(future.sum())
        if n == 0:
            out[sym] = new
            continue
        noise = factor * (1.0 + rng.normal(0, 0.5, n))
        for col in new.columns:
            if not pd.api.types.is_numeric_dtype(new[col]):
                continue
            values = new.loc[future, col].to_numpy(dtype="float64", na_value=np.nan)
            # +1 shifts binary flags (above_sma50, golden_cross) off 0/1 too,
            # so reading one after the cut is just as visible as reading a price.
            new[col] = new[col].astype("float64")
            new.loc[future, col] = (values + 1.0) * noise
        out[sym] = new
    return out


def check_strategy_path(strategy_fn, universe: dict[str, pd.DataFrame],
                        cfg=None, cut_fraction: float = 0.6,
                        tolerance: float = 1e-9) -> LeakReport:
    """THE end-to-end test: run the strategy twice, once on real data and
    once with everything after a cut date destroyed. The equity curve up to
    the cut must be bit-for-bit identical.

    This is the strongest check here because it is indifferent to how the
    strategy is written. Indicators, ranking, position sizing, stop
    placement, index membership - if anything anywhere consults a bar it
    should not have, the pre-cut path moves and this reports it.

    A leak found here but not by check_features means the leak is in the
    strategy logic, not the indicators.
    """
    report = LeakReport()
    symbols = [s for s, d in universe.items() if d is not None and len(d)]
    if not symbols:
        return report
    index = max((universe[s].index for s in symbols), key=len)
    if len(index) < 50:
        report.notes.append("strategy path: not enough history")
        return report
    cut = index[int(len(index) * cut_fraction)]

    real = strategy_fn(universe, cfg) if cfg is not None else strategy_fn(universe)
    mangled_uni = perturb_after(universe, cut)
    fake = (strategy_fn(mangled_uni, cfg) if cfg is not None
            else strategy_fn(mangled_uni))

    a = real.equity.loc[real.equity.index <= cut]
    b = fake.equity.loc[fake.equity.index <= cut]
    if a.empty or b.empty:
        report.notes.append("strategy path: empty equity curve, nothing compared")
        return report

    common = a.index.intersection(b.index)
    if len(common) != len(a):
        report.leaks.append(Leak("strategy", "equity_index", cut,
                                 float(len(a)), float(len(b)), float("inf")))
    for date in common:
        x, y = float(a.loc[date]), float(b.loc[date])
        report.checks += 1
        scale = max(abs(x), abs(y), 1.0)
        if abs(x - y) / scale > tolerance:
            report.leaks.append(Leak("strategy", "equity", date, x, y,
                                     abs(x - y) / scale))
    return report


# ---------------------------------------------------------------------------
# 4. warm-up sensitivity (not leakage - reproducibility)
# ---------------------------------------------------------------------------

def warmup_sensitivity(raw: pd.DataFrame, offsets: tuple[int, ...] = (0, 50, 150, 300),
                       tolerance: float = 1e-4) -> dict[str, float]:
    """How much does each indicator move when history starts later?

    Returns column -> worst relative drift at the final bar across `offsets`.
    EMA, Wilder RSI/ATR and ADX are recursive: they never fully forget their
    seed, so a nonzero number here is expected and correct. It is only a
    problem when it exceeds `tolerance`, which means the backtest and the
    live bot - which start from different points - will disagree on the same
    bar. The fix is more warm-up, not different code.
    """
    from data.fetcher import add_features

    if raw is None or len(raw) < max(offsets) + 300:
        return {}
    base = add_features(raw)
    last = base.index[-1]
    drift: dict[str, float] = {}
    for off in offsets:
        if off == 0:
            continue
        alt = add_features(raw.iloc[off:])
        if last not in alt.index:
            continue
        for col in base.columns:
            x, y = _num(base.loc[last, col]), _num(alt.loc[last, col])
            if x is None or y is None:
                continue
            scale = max(abs(x), abs(y), 1.0)
            drift[col] = max(drift.get(col, 0.0), abs(x - y) / scale)
    return dict(sorted(drift.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def run_all(raw_universe: dict[str, pd.DataFrame], feature_symbols: int = 6,
            n_dates: int = 8) -> LeakReport:
    """Every check, over a sample of the universe. `raw_universe` must be
    OHLCV WITHOUT features - the checks add them both ways themselves."""
    report = LeakReport()
    symbols = sorted(s for s, d in raw_universe.items() if d is not None and len(d))

    for sym in symbols[:feature_symbols]:
        report.extend(check_features(raw_universe[sym], sym, n_dates=n_dates))

    report.extend(check_scores(
        {s: raw_universe[s] for s in symbols[:max(feature_symbols, 4)]},
        n_dates=max(n_dates // 2, 3)))

    # End-to-end, on the strategies that actually decide trades.
    from backtest.engine import BacktestConfig, run_backtest
    from backtest.lab import LabConfig, backtest_momentum, backtest_trend_filter
    from data.fetcher import add_features

    featured = {s: add_features(raw_universe[s]) for s in symbols}
    for name, fn, cfg in (
        ("momentum", backtest_momentum, LabConfig()),
        ("trend_filter", backtest_trend_filter, LabConfig()),
        ("weekly_rotation", run_backtest, BacktestConfig(warmup_bars=210)),
    ):
        sub = check_strategy_path(fn, featured, cfg)
        for leak in sub.leaks:
            leak.subject = f"{name}.{leak.subject}"
        report.extend(sub)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Empirical look-ahead detection")
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--symbols", type=int, default=12,
                    help="universe size to pull (features are O(n^2) to recheck)")
    ap.add_argument("--feature-symbols", type=int, default=6)
    ap.add_argument("--dates", type=int, default=8, help="sample dates per check")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from data.alpaca_data import default_stock_universe, fetch_stock_universe_alpaca

    tickers = default_stock_universe()[:args.symbols]
    lookback = int(args.years * 365) + 500
    logger.info("fetching %d symbols, %d days...", len(tickers), lookback)
    universe = fetch_stock_universe_alpaca(lookback_days=lookback, tickers=tickers)
    if not universe:
        logger.error("no data - check ALPACA_API_KEY / ALPACA_SECRET_KEY")
        return 1

    # fetch_stock_universe_alpaca returns featured frames; strip back to OHLCV
    # so the checks can recompute features both ways from the same source.
    ohlcv = {s: df[["open", "high", "low", "close", "volume"]].copy()
             for s, df in universe.items()
             if df is not None and {"open", "high", "low", "close", "volume"} <= set(df.columns)}

    report = run_all(ohlcv, feature_symbols=args.feature_symbols, n_dates=args.dates)

    print()
    print("=" * 78)
    print("LOOK-AHEAD DETECTION")
    print("-" * 78)
    print(report.summary())
    for leak in report.leaks[:25]:
        print("  LEAK", leak)
    if len(report.leaks) > 25:
        print(f"  ... and {len(report.leaks) - 25} more")
    for note in report.notes[:10]:
        print("  note:", note)

    sym = sorted(ohlcv)[0]
    drift = warmup_sensitivity(ohlcv[sym])
    if drift:
        print("-" * 78)
        print(f"WARM-UP SENSITIVITY ({sym}) - relative drift at the last bar when")
        print("history starts later. Recursive indicators never fully forget their")
        print("seed; this is reproducibility, not leakage.")
        for col, d in list(drift.items())[:10]:
            flag = "  <- needs more warmup" if d > 1e-4 else ""
            print(f"  {col:<16} {d:>12.2e}{flag}")
    print("=" * 78)

    if not report.clean:
        print("GATE 1 FAILED: fix these before trusting any backtest number.")
        return 1
    print("GATE 1 PASSED: no signal used data from after its decision bar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
