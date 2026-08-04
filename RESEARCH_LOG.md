# Research log

Every strategy variant ever evaluated against this data, one row each.

## Why this file exists

When you test N variants of a strategy family with **zero** true edge, the
expected best-of-N Sharpe is not zero. On five years of data it is about 0.70
at N=10 and 0.85 at N=20. So a backtest Sharpe of 0.81 chosen as the winner of
15 attempts is exactly what pure luck produces — and past roughly 45
configurations on five years, an in-sample Sharpe of 1.0 with a true
out-of-sample Sharpe of zero is close to guaranteed.

That correction only works if N is honest. N is the **whole research history** —
every variant, every threshold, every idea that was tried and quietly dropped —
not the size of the final parameter grid. The number is only ever allowed to go
up. Deleting a row because the variant lost is the exact behaviour the deflated
Sharpe exists to defend against.

`backtest/compare.py` counts the rows below at runtime and feeds the count to
`deflated_sharpe_ratio`, so every reported DSR is priced against the real
search. See `backtest/validate.py` for the maths (Bailey & López de Prado,
"Pseudo-Mathematics and Financial Charlatanism", *Notices of the AMS*, 2014).

## How to add a row

One row per distinct thing tried. A parameter sweep counts as one row per value
actually evaluated and compared, not one row for the sweep. Append; never edit
or remove.

## Trials

| # | date | strategy / variant | outcome |
|---|------|--------------------|---------|
| 1 | 2026-07 | weekly rotation, ML-scored, default keep-rank — the original live strategy | −20.26% over 3.3y while SPY rose. Falsified. |
| 2 | 2026-07 | weekly rotation, keep_rank=20 | still negative |
| 3 | 2026-07 | weekly rotation, keep_rank=100 | −1.6% over 3.3y vs SPY +79%. Falsified. |
| 4 | 2026-07 | buy & hold SPY (the bar everything must clear) | benchmark, not a candidate |
| 5 | 2026-07 | trend filter, 200-day MA | fewer trades, smaller drawdown, no excess return |
| 6 | 2026-07 | cross-sectional momentum 12-1, top 10, equal weight | +202% / 5y, Sharpe 0.81 — selected as the live strategy |
| 7 | 2026-07 | momentum 12-1, top 15 | comparable |
| 8 | 2026-07 | momentum, rank weighting (linear decay to the leader) | tested for concentration |
| 9 | 2026-07 | momentum, score weighting (proportional to momentum) | tested for concentration |
| 10 | 2026-07 | crypto, BTC-only buy & hold | baseline |
| 11 | 2026-07 | crypto, regime-gated (EMA stack + MACD) | gate telemetry added after the btc_only bug |
| 12 | 2026-07 | crypto, momentum across the wide universe | not yet compared head-to-head |
| 13 | 2026-07 | ML: absolute up/down labels, XGBoost | AUC 0.487 — no signal |
| 14 | 2026-07 | ML: relative (cross-sectional median) labels | AUC 0.509 — no signal |
| 15 | 2026-07 | ML: + cross-sectional percentile features (17 total) | rejected by the 0.52 AUC gate |
| 16 | 2026-07 | ML: purged walk-forward, 4 folds, with embargo | validation method, still no signal |
| 17 | 2026-07 | stock exit mode: Friday-close vs rank-based rotation | rotation kept |
| 18 | 2026-08 | momentum 12-1 with point-in-time index membership | +121% / 5y (was +202% biased), Sharpe 0.63, DSR 0.33 vs 0.95 gate. Fails deflation. Stress gates 4/5/6 all pass — robust in shape, level indistinguishable from luck. |

**Current N = 18.**

## The number that matters

At N=18 on five years of history, the noise floor — the Sharpe a strategy with
**no edge at all** is expected to produce as the best of 18 tries — is
**0.83**. Our best result was **0.81**.

Edge above noise: **−0.02**. The winner underperforms luck.

(At the N=15 we were using before this log was written honestly, the floor was
0.79 and the edge looked like +0.02. Counting the ML variants — which were
tried against the same data and dropped — moved it below zero. That sign flip
is the entire argument for keeping this file.)

That is the finding that stopped strategy work and started the measurement
work. Nothing here gets funded until the deflated Sharpe clears 0.95 against
this count.
