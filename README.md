# Stock Bot

An automated trading bot for a single Alpaca account, with a built-in web
dashboard. It runs two strategies side by side from the same pot of money:

- **Stocks (weekly swing):** scans and ranks a large-cap universe Sunday
  night, buys the top-ranked names Monday 9:45 AM ET, monitors stop-losses
  mid-week, and sells everything Friday 3:45 PM ET. The weekly cadence keeps
  API usage tiny — far below Alpaca's rate limits.
- **Crypto (24/7):** a 4-hour trend-following cycle on BTC (BTC-only until it
  proves 60 profitable days), trading around the clock through Alpaca's
  crypto market with the same account balance.

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

## Development

```bash
python -m pytest tests/ -q     # full offline suite (no network, no broker)
```

Project layout: `config/` settings · `data/` market data + features ·
`strategies/` signal generation and sizing · `risk/` portfolio risk engine ·
`execution/` broker order routing + account layer · `scheduler.py` job wiring ·
`webapp/` FastAPI dashboard · `main.py` entry point.
