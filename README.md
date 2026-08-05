# Stock Bot

An automated trading bot for a single Alpaca account, with a built-in web
dashboard. It runs two strategies side by side from the same pot of money:

- **Stocks (monthly momentum, default):** ranks the S&P 500 on 12-month
  return skipping the most recent month, holds the top N equal-weight, and
  rebalances on the first Monday of each month. No stop-losses — momentum
  exits by falling out of the ranking, and a tight stop on equity noise
  sells dips into recoveries.

  **Do not trust the headline backtest.** It reported +202% over 5 years
  (Sharpe 0.81 vs SPY's 0.70), but that number was selected as the best of
  18 logged trials, and the best of 18 zero-edge strategies scores Sharpe
  **0.83** on five years by luck alone — so the winner does not clear its
  own noise floor. MTUM, the real tradeable momentum ETF, returned 10.3%
  CAGR against this backtest's 24.9%. See `RESEARCH_LOG.md` and the
  validation gates below. The strategy is running on paper for engineering
  reasons, not because it is known to work.

  The legacy **weekly rotation** (`STOCK_STRATEGY=rotation`) is kept for
  comparison only: it backtested **−1.6% over 3.3 years against SPY +79%**.
- **Crypto (24/7):** a 4-hour trend-regime strategy — long while the trend
  is intact (price above EMA200, EMAs stacked, ADX trending), flat when it
  breaks, with stops re-checked every 15 minutes. Scans BTC/ETH/SOL/LTC/
  DOGE/LINK/AVAX; entries stay BTC-only until it proves itself
  (`CRYPTO_BTC_ONLY=false` unlocks the rest). A reserved slice of equity
  (`CRYPTO_ALLOCATION_PCT`, default 25%) guarantees crypto always has cash
  to trade even while stocks are deployed.

The money already in your Alpaca account **is** the bot's capital: equity and
cash are read live from the broker (cached 60 s) and every position is sized
against them.

## Quick start

```bash
cp .env.example .env        # add your two Alpaca keys
pip install -r requirements.txt
python main.py
```

Open **http://localhost:8000** — the dashboard shows account equity, the
equity curve, open positions, trades, signals, and every scheduled job with a
"Run now" button. `/api/docs` has the full JSON API.

The only required configuration is `ALPACA_API_KEY` / `ALPACA_SECRET_KEY`.
With nothing else set, the bot uses a local SQLite database, logs alerts
instead of sending Telegram messages, and pulls stock data from Alpaca's free
IEX feed over the full built-in S&P 500 list (a Polygon key switches to live
index constituents; `STOCK_UNIVERSE` overrides with your own tickers).

### Docker

```bash
cp .env.example .env        # add your keys
docker compose up -d --build
```

Data (SQLite, logs, reports, models) persists in mounted folders.

## Execution modes — the safety ladder

| Mode | Settings | What happens |
|---|---|---|
| **sim** (default) | `ENVIRONMENT=paper` | fills are simulated locally; **no orders are ever sent** |
| **paper** | `ENVIRONMENT=live`, `ALPACA_PAPER=true` | real orders to Alpaca's *paper* API — full rehearsal |
| **live** | `ENVIRONMENT=live`, `ALPACA_PAPER=false` | real money |

Run each rung until you trust it, then move up. In every mode the dashboard
and the DB record positions, trades, and signals identically.

## Configuration

Everything lives in `.env` (see `.env.example` for the full list):

| Variable | Default | Purpose |
|---|---|---|
| `CAPITAL_SOURCE` | `broker` | `broker` = live Alpaca equity/cash; `static` = `STARTING_CAPITAL` |
| `CRYPTO_EXCHANGE` | `alpaca` | `alpaca` (same account, `BTC/USD`) or `binance` (needs keys) |
| `WEB_PORT` | `8000` | dashboard/API port |
| `WEB_AUTH_TOKEN` | *(empty)* | set it to require a Bearer token on pause/resume/run-job |
| `DATABASE_URL` / `DB_*` | *(empty)* | Postgres; blank = SQLite in `DATA_DIR` |
| `POLYGON_API_KEY` | *(empty)* | optional: full S&P 500 scan universe |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | *(empty)* | optional trade/heartbeat alerts |
| `STARTING_CAPITAL` | `1000` | drawdown baseline (and equity in `static` mode) |

## Schedule (America/New_York)

| Job | When |
|---|---|
| Sunday stock scan (+ ML retrain) | Sun 20:00 / 21:00 |
| Monday stock buys | Mon 09:45 |
| Mid-week stop-loss monitor | Mon–Fri, every 30 min, 09:00–16:30 |
| Friday stock sells | Fri 15:45 |
| Crypto trend cycle | every 4 hours, 24/7 |
| Crypto stop monitor | every 15 min, 24/7 (only calls out while positions are open) |
| Weekly P&L report | Fri 16:30 |
| Daily heartbeat | 08:00 |

## Risk controls

- Fixed-fractional risk per trade (2% stocks / 1.5% crypto), capped by
  half-Kelly — a weak edge shrinks positions, never enlarges them.
- Portfolio heat caps, per-position notional caps, and a minimum position
  floor that adapts to account size (`MIN_POSITION_SIZE`, capped at 10% of
  equity, never below $10) so small accounts aren't locked out.
- Crypto stops are re-checked every 15 minutes between the 4-hour cycles.
- Monthly drawdown circuit breaker: −15% halts all new entries.
- **Pause button** in the dashboard stops new entries instantly; stop-loss
  monitoring and scheduled exits keep running so open positions stay
  protected.

## Backtesting

Replay the live strategy over real history before trusting a change:

```bash
python -m backtest.run --years 3
python -m backtest.run --years 5 --top-n 5 --cost-bps 15 --csv equity.csv
```

The engine calls the **same** scorer, rotation rule and position sizer the
scheduler calls, so it tests the strategy you actually run. Decisions on bar
T use data through T only and execute at T+1's open; entry/exit costs, the
stop (with gap-down fills) and the no-margin cash constraint are modeled.

Candidates are filtered to **point-in-time index membership**
(`data/index_membership.py`, 1996–2026), so the engine cannot rank a company
before it joined the S&P 500. That is a *partial* survivorship correction:
Alpaca has no bars for names delisted years ago, so the losers still cannot
be traded in replay and absolute returns remain optimistic. Only OHLC is
known intrabar, so a bar touching both stop and target resolves stop-first.

### Validating a result

A backtest number on its own means nothing. Three tools decide whether one
is worth acting on:

```bash
python -m backtest.leakage --years 3            # look-ahead detection
python -m backtest.compare --years 5            # side by side, with deflated Sharpe
python -m backtest.stress --strategy momentum   # 2x costs + robustness
```

The crypto strategies go through the same two gates:

```bash
python -m backtest.leakage --crypto --years 2
python -m backtest.crypto_compare --years 2 --stress
```

Crypto annualizes on 4h bars over 365 days (2190 periods/year), not 252 —
using the equity default would inflate every crypto Sharpe by ~2.9×.

- **`leakage`** proves the strategy cannot see the future. It recomputes
  every indicator from truncated data, and re-runs each strategy with all
  bars after a cut date randomized — the equity curve before the cut must
  not move. Four tests plant a deliberate leak to prove the detector works.
- **`compare`** reports a **deflated Sharpe** next to the raw one, priced
  against the trial count in `RESEARCH_LOG.md`. Raw Sharpe is meaningless
  without N: the more variants you try, the higher a score luck alone
  produces. DSR ≥ 0.95 is the gate.
- **`stress`** re-runs at 1×/2×/3× modeled costs, removes the best trades
  and the best calendar year, and sweeps each parameter ±25% to check the
  surface is a plateau rather than a spike. Exits non-zero when a gate fails.

**Gates before any real money** — all must pass, in order: zero leaks →
DSR ≥ 0.95 → permutation test p < 0.05 → survives 2× costs → parameter
plateau → survives removing the top 5 trades → bootstrapped 95th-percentile
drawdown written down in advance → 1–3 months of paper trading judged on
tracking error vs a shadow backtest, **not** on P&L. (`t = Sharpe × √years`
means confirming a Sharpe-0.5 edge from returns alone takes 16 years — paper
trading tests the engineering, not the edge.)

## Keeping it running (Windows)

`start_bot.bat` launches the bot with its venv. To survive reboots, register
it in Task Scheduler with a "When the computer starts" trigger — the file's
header comments have the exact steps. Also set Power → Sleep → Never; a
sleeping PC stops watching your stops.

**Do not keep this repo in OneDrive, Dropbox or Google Drive.** Sync races
rewrite files underneath processes that are still using them, which breaks
things in two ways that are hard to diagnose:

- Git updates refs with a compare-and-swap. When the sync client restores a
  ref mid-update, `git pull` fails with `incorrect old value provided` — and
  it fails *after* printing the commit range, so it looks like it worked
  while the working tree stays on old code.
- `data_store/stockbot.db` is written live by the scheduler. A sync client
  copying it mid-transaction can corrupt your trade history.

Put it somewhere local (`C:\Programming\stock_bot`), or exclude the folder in
OneDrive → Settings → Choose folders.

## Development

```bash
python -m pytest tests/ -q     # full offline suite (no network, no broker)
```

Project layout: `config/` settings · `data/` market data + features ·
`strategies/` signal generation and sizing · `risk/` portfolio risk engine ·
`execution/` broker order routing + account layer · `scheduler.py` job wiring ·
`webapp/` FastAPI dashboard · `main.py` entry point.
