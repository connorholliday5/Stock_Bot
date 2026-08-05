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

| 19 | 2026-08 | crypto: hold BTC (the bar the others must clear) | −3.90% / 2y, Sharpe 0.06, maxDD −53.57%, $25 costs. Benchmark, not a candidate. |
| 20 | 2026-08 | crypto: regime, BTC only | −15.79% / 2y, Sharpe −0.13, DSR 0.007. maxDD −28.05% (half of hold's) but $2,714 costs on 112 trades. Loses to hold. |
| 21 | 2026-08 | crypto: regime, wide universe (7 coins) | −76.89% / 2y, Sharpe −0.55, DSR 0.000, $3,763 costs on 372 trades. Worst result in this log. |
| 22 | 2026-08 | crypto: momentum top-3 | −52.63% / 2y, Sharpe −0.11, DSR 0.008, $2,937 costs on 407 trades. Loses to hold. |

**Current N = 22.**

Rows 19-22 were logged **before** their results, on purpose. The crypto
strategies were written and iterated on months ago against this same history;
counting them only once they produce a good number is precisely the bias this
file exists to prevent. Adding them moves the noise floor up for every future
result, including the equity ones — which is correct, because the trials were
real whether or not anyone wrote them down.

### What the crypto run settled (2026-08, 2y, 4h bars, 25bps/side)

Every active crypto strategy lost to doing nothing, and the more it traded the
worse it did:

| | return | trades | costs | costs as % of capital |
|---|---|---|---|---|
| hold BTC | −3.90% | 1 | $25 | 0.3% |
| regime BTC-only | −15.79% | 112 | $2,714 | 27% |
| momentum top-3 | −52.63% | 407 | $2,937 | 29% |
| regime wide (7) | −76.89% | 372 | $3,763 | 38% |

Three findings:

1. **Widening the universe is actively destructive.** BTC-only −15.79% vs the
   7-coin universe −76.89%, same strategy. This is the question
   `crypto_compare.py` was written to answer. `CRYPTO_BTC_ONLY=true` was
   already the default; it is now an evidence-backed setting rather than a
   cautious guess.
2. **The trend filter works; the costs eat it.** Regime BTC-only cut max
   drawdown roughly in half (−28.05% vs hold's −53.57%) — that is exactly the
   job it was given, and it did it. But 112 trades at 25bps per side burned
   27% of the account, which is more than the protection was worth. The signal
   is not obviously worthless; the *trading frequency at this cost level* is.
3. **Nothing survives deflation.** Best DSR among the active strategies is
   0.008 against a 0.95 gate. On a 2-year window with N=22 the noise floor is
   Sharpe 1.37, and the best raw Sharpe here is negative.

Caveat worth keeping: this window was itself bad for crypto — BTC hold lost
money with a 53.57% drawdown. These strategies have not been observed in a
crypto bull market, so "loses to hold" is established, "has no edge ever" is
not. The gates cannot be run on a window that has not happened yet.

Stress on the best active strategy (regime BTC-only) failed gates 4 and 6:
−36.35% at 2× costs, and removing the 5 best days takes it to −30.26%.

## The number that matters

At N=22 on five years of history, the noise floor — the Sharpe a strategy with
**no edge at all** is expected to produce as the best of 22 tries — is
**0.87**. Our best result was **0.81**.

Edge above noise: **−0.06**. The winner underperforms luck.

(At the N=15 we were using before this log was written honestly, the floor was
0.79 and the edge looked like +0.02. Counting the ML variants — which were
tried against the same data and dropped — moved it below zero. That sign flip
is the entire argument for keeping this file. Adding the four crypto trials
pushed it from −0.02 to −0.06: the equity result got worse without a single
equity number changing, because the trial count is a property of the search,
not of the strategy.)

That is the finding that stopped strategy work and started the measurement
work. Nothing here gets funded until the deflated Sharpe clears 0.95 against
this count.
