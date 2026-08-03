"""
backtest/validate.py
Statistical validation - does a backtest result mean anything?

Every strategy we tried looked plausible before it was measured, and the one
that looked best (momentum, 5y Sharpe 0.81) was selected as the winner of
~12-15 variants tried against the same data. That selection is itself the
problem this module quantifies.

Core result (Bailey & Lopez de Prado, "Pseudo-Mathematics and Financial
Charlatanism", Notices of the AMS 2014): when you test N variants of a
strategy family with ZERO true edge, the expected best-of-N Sharpe is not 0.
On 5 years of data it is ~0.70 at N=10 and ~0.85 at N=20 - so a backtest
Sharpe of 0.81 after 15 trials is exactly what pure noise produces. Testing
more than ~45 configurations on 5 years essentially guarantees an in-sample
Sharpe of 1.0 with a true out-of-sample Sharpe of zero.

Tools here:
  expected_max_sharpe   the noise floor for N trials
  deflated_sharpe_ratio probability the edge is real given N, skew, kurtosis
  permutation_test      does the rule beat itself on shuffled returns?
  bootstrap_drawdown    the drawdown distribution you must be willing to hold

Count N honestly: it is your whole research history - every variant, every
threshold, every abandoned idea - not the size of the final grid.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EULER_MASCHERONI = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (Acklam's rational approximation, ~1e-9)."""
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def expected_max_sharpe(n_trials: int, years: float) -> float:
    """Expected best annualized Sharpe from N trials of a ZERO-edge strategy.

    E[max SR] ~= sigma_SR * [(1-g)*Z^-1(1-1/N) + g*Z^-1(1-1/(N*e))]
    with sigma_SR ~= 1/sqrt(years). This is the bar a real edge must clear -
    not zero.
    """
    n = max(int(n_trials), 1)
    if n == 1:
        return 0.0
    sigma = 1.0 / math.sqrt(max(float(years), 1e-9))
    g = EULER_MASCHERONI
    return sigma * ((1 - g) * _norm_ppf(1 - 1.0 / n)
                    + g * _norm_ppf(1 - 1.0 / (n * math.e)))


@dataclass
class DSRResult:
    sharpe: float
    deflated_sharpe: float      # probability the true Sharpe > threshold
    expected_max_noise: float
    n_trials: int
    years: float
    verdict: str

    def summary(self) -> str:
        return (f"Sharpe {self.sharpe:.2f} | noise floor for {self.n_trials} "
                f"trials {self.expected_max_noise:.2f} | DSR "
                f"{self.deflated_sharpe:.3f} -> {self.verdict}")


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    periods_per_year: int = 252,
    benchmark_sharpe: Optional[float] = None,
) -> DSRResult:
    """Probability the strategy's true Sharpe exceeds the noise floor.

    Accounts for trial count, sample length, skew and kurtosis - negative
    skew and fat tails (typical of trend and short-vol strategies) make a
    given Sharpe less trustworthy, not more.

    DSR < 0.95 means you cannot reject "this is the luckiest of N coin flips".
    """
    r = pd.Series(returns).dropna()
    n = len(r)
    if n < 20 or r.std() == 0:
        return DSRResult(0.0, 0.0, 0.0, n_trials, 0.0, "insufficient data")

    years = n / float(periods_per_year)
    sr = float(r.mean() / r.std()) * math.sqrt(periods_per_year)   # annualized
    sr_star = (benchmark_sharpe if benchmark_sharpe is not None
               else expected_max_sharpe(n_trials, years))

    skew = float(r.skew())
    kurt = float(r.kurtosis()) + 3.0        # pandas gives excess kurtosis
    sr_p = sr / math.sqrt(periods_per_year)  # per-period for the moment terms
    denom = 1.0 - skew * sr_p + ((kurt - 1.0) / 4.0) * (sr_p ** 2)
    denom = max(denom, 1e-12)
    z = ((sr - sr_star) * math.sqrt(max(n - 1, 1))
         / (math.sqrt(periods_per_year) * math.sqrt(denom)))
    dsr = _norm_cdf(z)

    if dsr >= 0.95:
        verdict = "survives deflation (edge plausibly real)"
    elif dsr >= 0.80:
        verdict = "borderline - not significant"
    else:
        verdict = "INDISTINGUISHABLE FROM NOISE"
    return DSRResult(sr, dsr, sr_star, n_trials, years, verdict)


def permutation_test(
    strategy_fn: Callable[[dict], float],
    universe: dict,
    n_permutations: int = 200,
    seed: int = 42,
) -> dict:
    """Does the rule beat itself on SHUFFLED returns?

    Shuffling daily returns destroys serial structure (trend, momentum) while
    preserving each asset's mean and volatility. If the strategy scores as
    well on shuffled data, it was harvesting drift or volatility, not the
    pattern it claims to trade.

    strategy_fn takes a universe dict and returns a scalar score (e.g. total
    return). Returns the real score, the null distribution and a p-value.
    """
    rng = np.random.default_rng(seed)
    real = float(strategy_fn(universe))

    null: list[float] = []
    for _ in range(int(n_permutations)):
        shuffled = {}
        for sym, df in universe.items():
            if df is None or df.empty or "close" not in df.columns:
                continue
            close = pd.to_numeric(df["close"], errors="coerce").dropna()
            if len(close) < 10:
                continue
            # copy: pandas can hand back a read-only view, and shuffle is in-place
            rets = np.array(close.pct_change().dropna().to_numpy(), copy=True)
            rng.shuffle(rets)
            path = float(close.iloc[0]) * np.cumprod(1.0 + rets)
            new = df.iloc[1:len(path) + 1].copy()
            if len(new) != len(path):
                path = path[:len(new)]
            new["close"] = path
            for col in ("open", "high", "low"):
                if col in new.columns:
                    new[col] = path
            shuffled[sym] = new
        try:
            null.append(float(strategy_fn(shuffled)))
        except Exception:
            continue

    if not null:
        return {"real": real, "p_value": float("nan"), "n": 0}
    arr = np.array(null)
    p = float((arr >= real).sum() + 1) / (len(arr) + 1)
    return {
        "real": real,
        "null_mean": float(arr.mean()),
        "null_p95": float(np.percentile(arr, 95)),
        "p_value": p,
        "n": len(arr),
        "verdict": ("beats shuffled data" if p < 0.05
                    else "NOT distinguishable from shuffled data"),
    }


def bootstrap_drawdown(equity: pd.Series, n_samples: int = 1000,
                       block: int = 21, seed: int = 42) -> dict:
    """Block-bootstrap the return path to get the drawdown DISTRIBUTION.

    The backtest shows one path. The question that decides whether you can
    actually run a strategy is the 95th-percentile drawdown - the one you
    have to sit through without turning the bot off.
    """
    eq = pd.Series(equity).dropna()
    if len(eq) < 50:
        return {}
    rets = eq.pct_change().dropna().to_numpy()
    rng = np.random.default_rng(seed)
    n_blocks = max(int(np.ceil(len(rets) / block)), 1)

    worst: list[float] = []
    for _ in range(int(n_samples)):
        starts = rng.integers(0, max(len(rets) - block, 1), size=n_blocks)
        path = np.concatenate([rets[s:s + block] for s in starts])[:len(rets)]
        curve = np.cumprod(1.0 + path)
        dd = curve / np.maximum.accumulate(curve) - 1.0
        worst.append(float(dd.min()))

    arr = np.array(worst)
    return {
        "observed_max_dd_pct": float((eq / eq.cummax() - 1).min()) * 100,
        "median_max_dd_pct": float(np.percentile(arr, 50)) * 100,
        "p95_max_dd_pct": float(np.percentile(arr, 5)) * 100,   # 5th = worst tail
        "worst_max_dd_pct": float(arr.min()) * 100,
        "n_samples": len(arr),
    }
