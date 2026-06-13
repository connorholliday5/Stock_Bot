# CONTEXT - Phase 4 (Crypto 24/7 Strategy)

Append this section to your existing CONTEXT.md.

## Status: Phase 4 COMPLETE (sandbox-tested), pending one wiring step

28/28 tests pass in tests/test_phase4.py. All strategy/indicator/sizing/persistence
logic verified offline (no network, real DB schema on in-memory SQLite). Final gate
is your local `pytest` on Windows.

## What changed

**Bug fixed (was latent, would crash on first real signal write):**
`persist_signals` wrote `direction = SignalType.value` ("BUY"/"SELL") into a
`Column(Enum(SignalDirection))` whose only valid values are long/short/flat.
Verified: "BUY" raises LookupError on read-back. Phase 3's 80/80 passed only
because the DB was mocked.
Fix: `_DIRECTION_MAP` translates BUY->long, SELL->flat, HOLD->flat.
SELL maps to FLAT, not SHORT - this is a spot bot, an exit is flat. SHORT stays
reserved for a future margin strategy.
Note: `asset_type` was fine - "stock"/"crypto" are valid AssetType values
(SQLAlchemy 2.0 accepts the value-string). Only `direction` was broken.

**data/fetcher.py** - added compute_ema, compute_macd (12/26/9), compute_adx
(Wilder), compute_realized_vol; `add_features` now also emits ema_9/21/200, macd,
macd_signal, macd_hist, adx, realized_vol (Phase 2 features unchanged). Added
`CryptoFetcher.get_funding_rate` (defensive, returns 0.0 on any failure) and
`fetch_crypto_funding`. `fetch_crypto_universe(timeframe="4h", limit=300)` already
accepted timeframe, so 4H needed no new fetch path.

**data/validator.py** - one change: `validate_crypto_df(..., max_staleness_hours=None)`
override so 4H frames are not falsely failed by the 1H (2h) staleness rule.

**strategies/signals.py** - direction-map fix (above); `SignalEvent` gained an
`adx` field so Phase 4 populates the reserved `Signal.adx` column; dedup filter
now queries the mapped direction value.

**strategies/crypto_24h.py** - NEW. Long-only spot 4H trend strategy.
Entry gates (all required): close > EMA200; regime == TRENDING; bullish EMA9/21
cross within 3 bars; MACD confirm (macd>signal and hist>0); volume_ratio >= 1.5;
not already held; open crypto positions < 3; BTC-only flag (default True).
Regime: HIGH_VOL (realized_vol >= 0.10) takes precedence, then TRENDING (ADX>=25),
MEAN_REVERTING (ADX<=20), else UNKNOWN.
Sizing: effective risk = min(1.5% fixed, half-Kelly) so a weak/negative edge
shrinks or zeroes the trade, never enlarges it; stop = entry - 2*ATR; TP at 3R;
negative funding applies a 0.5 haircut; funding <= -0.001 skips; floored at $50;
**capped at 35% of equity per position** (new risk control - see decisions); capped
at available cash (no margin).

**execution/ccxt_crypto.py** - NEW. CryptoExecutor, paper mode default True (no real
orders in Phase 1). open_long / close_long write Position + Trade via get_db using
enum members (AssetType.CRYPTO, OrderSide.BUY/SELL, OrderStatus.FILLED). Backstops:
min $50 notional, max 3 positions, no duplicate symbol. Live path routes through
ccxt create_order.

**tests/test_phase4.py** - NEW, 28 tests. Network never touched (synthetic frames).
DB tests run the REAL schema on in-memory SQLite, so they exercise the actual Enum
binding and prove the direction fix (BUY->long, SELL->flat) end to end.

## Decisions made this phase (engineering judgment, per the autonomy note)

1. SELL -> SignalDirection.FLAT (spot exit, not a short).
2. Added a 35%-of-equity per-position notional cap. A tight ATR stop with
   fixed-fractional 1.5% risk can otherwise buy ~70%+ of capital in one name;
   with max 3 positions and no margin, that concentration is undesirable. The cap
   only ever reduces exposure. Tune MAX_POSITION_NOTIONAL_PCT in crypto_24h.py.
3. Reused validate_crypto_df with a staleness override rather than writing a second
   validator - one validation path, correct freshness semantics for 4H.
4. half-Kelly is a CAP on the 1.5% risk, not a multiplier on top of it.
   Defaults win_rate=0.5, win_loss_ratio=1.5 are placeholders until live stats
   calibrate them (Phase 5+ from WeeklyPerformance/ModelPerformance).

## Open / heads-up

- scheduler.py: crypto_cycle() delivered as a drop-in (scheduler_crypto_cycle.py).
  Not folded into scheduler.py because that file was not provided. Paste current
  scheduler.py next session for a complete-file delivery + job registration.
- PAPER_CASH in the cycle is hardcoded to 10000. Wire to your real capital/accounting.
- BTC-only gate is hardcoded True; flip via WeeklyPerformance once 60 days profitable.
- Minor pre-existing (not Phase 4): db.py health_check uses db.execute("SELECT 1");
  SQLAlchemy 2.0 needs text("SELECT 1"). Left untouched (out of scope).