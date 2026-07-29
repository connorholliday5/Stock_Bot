"""scheduler.py - production job wiring.

APScheduler BackgroundScheduler. main.py owns the process; this module builds
and returns a configured scheduler with every job registered, and exposes the
job callables (JOBS) so the web UI can trigger any of them manually.

Capital: get_account_state() reads the LIVE Alpaca account (equity/cash) via
execution.account when CAPITAL_SOURCE=broker - the money in the account is the
capital base. It degrades to the DB-reconstructed NAV when the broker is
unreachable or CAPITAL_SOURCE=static, so nothing here ever blocks on the API.

Rate-limit posture: the account read is TTL-cached (60s), stock data arrives
in a handful of batched requests once a week (Sunday scan / Monday buys /
Friday sells) plus one small mark-refresh batch per 30-min monitor tick, and
crypto data uses Alpaca's keyless crypto endpoint every 4h - comfortably below
Alpaca's ~200 req/min budget by design.

Locked-strategy notes honored here:
  - Stocks: buy Mon 9:45 ET, sell Fri 3:45 ET, Sunday-night scan.
  - Mid-week: stop-loss monitoring only. No over-managing.
  - Crypto: 24/7 4H cycle through the SAME Alpaca account, BTC-only until
    60 days profitable (BTC_ONLY flag below).
  - Every entry job is gated on RiskManager.is_halted() AND the web UI pause
    switch (runtime.bot_state). Pause never blocks risk-reducing jobs.
"""

from __future__ import annotations

import functools
import threading
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from config.settings import settings
from database import db
from risk.manager import RiskManager, MONTHLY_DRAWDOWN_HALT, effective_min_position
from execution.alpaca import StockExecutor
from execution.account import get_account_snapshot
from execution.alpaca_crypto import crypto_executor_from_settings, to_alpaca_symbol
from runtime import bot_state
from strategies import stock_scorer, stock_weekly, crypto_24h, momentum_monthly

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
except Exception:  # pragma: no cover - apscheduler present on target machine
    BackgroundScheduler = None
    CronTrigger = None

MARKET_TZ = "America/New_York"  # ET; APScheduler resolves EST/EDT.


def _btc_only() -> bool:
    """Locked strategy: BTC-only entries until 60 profitable days, then flip
    CRYPTO_BTC_ONLY=false in .env (the scan universe is already wider)."""
    return bool(getattr(settings, "crypto_btc_only", True))

# Trained weekly stock model artifact (booster at MODEL_PATH + ".json",
# metadata at MODEL_PATH + ".meta.json"). TrainedModel.save creates the dir.
MODEL_PATH = "models/weekly_stock"

# A retrained model must beat this out-of-sample AUC to earn its blend seat;
# below it the artifact is quarantined and the scan stays rule-only. 0.5 is
# a coin flip - blending a sub-0.52 model just adds noise to the rankings.
ML_MIN_AUC = 0.52


@dataclass
class AccountState:
    equity: float
    cash: float
    open_positions: list


# ============================================================
# ACCOUNT / POSITION ADAPTERS
# ============================================================

def build_risk_manager() -> RiskManager:
    return RiskManager(starting_capital=settings.starting_capital,
                       monthly_halt=MONTHLY_DRAWDOWN_HALT)


def _rm_halted(rm) -> bool:
    """RiskManager.is_halted is a method; tolerate property-style fakes too."""
    halted = rm.is_halted
    return bool(halted() if callable(halted) else halted)


def _open_positions(asset_type: str) -> list:
    return list(db.get_open_positions(asset_type))


def _position_dicts(positions) -> list[dict]:
    """ORM Position rows -> the dict shape the strategy layer consumes.
    Includes both 'symbol' and 'ticker' keys (stock signals read 'ticker',
    crypto reads 'symbol')."""
    out = []
    for p in positions:
        symbol = getattr(p, "symbol", None) or ""
        out.append({
            "symbol": symbol,
            "ticker": symbol,
            "quantity": float(getattr(p, "quantity", 0.0) or 0.0),
            "entry_price": float(getattr(p, "entry_price", 0.0) or 0.0),
            "current_price": getattr(p, "current_price", None),
            "stop_loss": float(getattr(p, "stop_loss", 0.0) or 0.0),
            "take_profit": float(getattr(p, "take_profit", 0.0) or 0.0),
            "week_number": getattr(p, "week_number", None),
        })
    return out


def _invested(open_positions: list) -> float:
    total = 0.0
    for p in open_positions:
        mark = getattr(p, "current_price", None) or getattr(p, "entry_price", 0.0)
        total += float(getattr(p, "quantity", 0.0) or 0.0) * float(mark or 0.0)
    return total


# Fraction of broker cash held back when sizing, so a real fill a few cents
# above the sizing reference can never trip "insufficient buying power".
BROKER_CASH_BUFFER = 0.01


def _allocation_cap(asset_type: str, equity: float) -> Optional[float]:
    """Remaining budget for this book under CRYPTO_ALLOCATION_PCT.

    With a reservation configured (> 0), crypto may deploy at most
    pct*equity and stocks at most (1-pct)*equity, so Monday's stock buys can
    never starve the crypto book of cash (observed live: stocks took 99% of
    the account and BTC had $2.31 to trade with all week). None = no cap.
    """
    pct = float(getattr(settings, "crypto_allocation_pct", 0.0) or 0.0)
    if pct <= 0 or equity <= 0:
        return None
    pct = min(pct, 1.0)
    if asset_type == "crypto":
        budget = equity * pct - _invested(_open_positions("crypto"))
    else:
        budget = equity * (1.0 - pct) - _invested(_open_positions("stock"))
    return max(0.0, budget)


def get_account_state(rm: RiskManager, asset_type: str) -> AccountState:
    """Equity/cash for sizing. Broker snapshot when configured; DB fallback.
    Cash is additionally capped by the per-book allocation budget."""
    positions = _open_positions(asset_type)
    snap = get_account_snapshot(rm)
    if snap.source == "broker":
        cash = max(0.0, snap.cash * (1.0 - BROKER_CASH_BUFFER))
    else:
        cash = max(0.0, snap.equity - _invested(positions))
    cap = _allocation_cap(asset_type, snap.equity)
    if cap is not None:
        cash = min(cash, cap)
    return AccountState(equity=snap.equity, cash=cash, open_positions=positions)


# ============================================================
# DATA ADAPTERS
# ============================================================

def _crypto_symbols() -> list[str]:
    syms = list(getattr(settings, "crypto_universe", None) or ["BTC/USD"])
    if getattr(settings, "crypto_exchange", "alpaca") == "alpaca":
        syms = [to_alpaca_symbol(s) for s in syms]
    # dedupe, preserve order (BTC/USDT and BTC/USD both map to BTC/USD)
    return list(dict.fromkeys(syms))


def _load_crypto_universe() -> dict:
    """{symbol: 4H feature DataFrame}. Alpaca keyless data by default; the
    ccxt/Binance public path when CRYPTO_EXCHANGE=binance. {} on failure."""
    timeframe = getattr(settings, "crypto_timeframe", "4h")
    if getattr(settings, "crypto_exchange", "alpaca") == "alpaca":
        from data.alpaca_data import fetch_crypto_universe_alpaca
        return fetch_crypto_universe_alpaca(symbols=_crypto_symbols(), timeframe=timeframe)
    try:
        from data.fetcher import fetch_crypto_universe
        return fetch_crypto_universe(timeframe=timeframe) or {}
    except Exception as exc:
        logger.exception("_load_crypto_universe failed: {}", exc)
        return {}


def _load_crypto_funding() -> dict:
    """Perp funding haircut inputs. Alpaca is SPOT - no funding, so {} (the
    strategy treats missing symbols as 0.0). Binance path fetches real rates."""
    if getattr(settings, "crypto_exchange", "alpaca") == "alpaca":
        return {}
    try:
        from data.fetcher import fetch_crypto_funding
        return fetch_crypto_funding(_crypto_symbols()) or {}
    except Exception as exc:
        logger.warning("_load_crypto_funding failed ({}); proceeding with empty.", exc)
        return {}


def _load_feature_universe(lookback_days: int | None = None) -> dict:
    """Stock scan universe {ticker: daily feature DataFrame}.
    Polygon (full S&P 500) when a key is configured; otherwise Alpaca IEX
    daily bars over the S&P 500 list. {} on failure.

    lookback_days overrides the fetch window - the Sunday retrain passes a
    multi-year value (ML_LOOKBACK_DAYS) for more training history; the weekly
    scan uses the fetcher's shorter default."""
    if settings.polygon_api_key and not settings.polygon_api_key.startswith("your_"):
        try:
            universe = db.get_feature_universe()
            if universe:
                return dict(universe)
            logger.warning("_load_feature_universe: Polygon path returned empty; trying Alpaca data")
        except Exception as exc:
            logger.exception("_load_feature_universe (polygon) failed: {}", exc)
    try:
        from data.alpaca_data import fetch_stock_universe_alpaca
        if lookback_days is not None:
            return fetch_stock_universe_alpaca(lookback_days=lookback_days)
        return fetch_stock_universe_alpaca()
    except Exception as exc:
        logger.exception("_load_feature_universe (alpaca) failed: {}", exc)
        return {}


def _load_stock_frames(tickers: list[str], lookback_days: int = 90) -> dict:
    """Small per-symbol daily frames for exits/marks - one batched request."""
    if not tickers:
        return {}
    try:
        from data.alpaca_data import fetch_stock_universe_alpaca
        return fetch_stock_universe_alpaca(lookback_days=lookback_days, tickers=tickers)
    except Exception as exc:
        logger.warning("_load_stock_frames failed ({})", exc)
        return {}


def _latest_closes(universe: dict) -> dict:
    prices = {}
    for symbol, df in (universe or {}).items():
        try:
            close = float(df["close"].iloc[-1])
            if close > 0:
                prices[symbol] = close
        except Exception:
            continue
    return prices


# ----- ML adapter chokepoints -----------------------------------

def _load_ml_scorer():
    """Load the trained ML model for the composite blend, or None for pure
    rule scoring (no model on disk yet / load failure)."""
    try:
        from ml.model import MLScorer
        booster = Path(MODEL_PATH).with_suffix(".json")
        if not booster.exists():
            logger.info("_load_ml_scorer: no model at {} (rule-only scoring).", booster)
            return None
        return MLScorer.from_path(MODEL_PATH)
    except Exception as exc:
        logger.warning("_load_ml_scorer failed ({}); rule-only scoring.", exc)
        return None


def _write_model_performance(metrics: dict) -> None:
    """Persist retrain metrics to ModelPerformance. Best-effort."""
    writer = getattr(db, "persist_model_performance", None)
    if writer is None:
        logger.info("_write_model_performance: no DB writer wired; metrics={}", metrics)
        return
    try:
        writer(metrics)
    except Exception as exc:
        logger.exception("_write_model_performance failed: {}", exc)


# ============================================================
# JOB GUARD - records every run for the web UI, never raises
# ============================================================

# Jobs suppressed by the UI pause switch. Risk-REDUCING jobs are never
# suppressed (midweek monitor, friday sells keep protecting open positions).
ENTRY_JOBS = {"monday_stock_buys", "crypto_cycle"}


# One lock per job: a job can never overlap itself. Observed live: a
# double-clicked "Run now" launched two concurrent buy runs that raced each
# other into duplicate orders at the broker (contained by the backstops, but
# noisy). Second invocation now records "skipped: already running".
_job_locks: dict[str, threading.Lock] = {}


def _guarded(job_id: str):
    lock = _job_locks.setdefault(job_id, threading.Lock())

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if job_id in ENTRY_JOBS and bot_state.paused:
                logger.warning("{}: skipped - bot paused from web UI", job_id)
                run = bot_state.job_started(job_id)
                bot_state.job_finished(run, ok=True, detail="skipped: paused")
                return None
            if not lock.acquire(blocking=False):
                logger.warning("{}: skipped - previous run still in progress", job_id)
                run = bot_state.job_started(job_id)
                bot_state.job_finished(run, ok=True, detail="skipped: already running")
                return None
            run = bot_state.job_started(job_id)
            try:
                result = fn(*args, **kwargs)
                bot_state.job_finished(run, ok=True, detail=str(result or "")[:200])
                return result
            except Exception as exc:  # never let a job crash the scheduler thread
                logger.exception("{} failed: {}", job_id, exc)
                bot_state.job_finished(run, ok=False, detail=str(exc))
                return None
            finally:
                lock.release()
        wrapper.__wrapped_job_id__ = job_id
        return wrapper
    return decorate


# ============================================================
# JOBS
# ============================================================

def _score_and_persist(universe: dict):
    """Score a feature universe (rule + optional ML blend) and persist the
    ranking. Shared by the Sunday scan and the Friday rotation (which needs a
    FRESH ranking to decide what still deserves its slot)."""
    ml_scorer = _load_ml_scorer()
    ml_weight = stock_scorer.DEFAULT_ML_WEIGHT if ml_scorer is not None else 0.0
    result = stock_scorer.score_universe(universe, ml_scorer=ml_scorer, ml_weight=ml_weight)
    ranked = getattr(result, "ranked", result)
    db.persist_scored_universe(ranked)
    return ranked


@_guarded("sunday_stock_scan")
def sunday_stock_scan() -> None:
    logger.info("sunday_stock_scan: scoring stock universe")
    universe = _load_feature_universe()
    if not universe:
        logger.warning("sunday_stock_scan: empty feature universe. Skipping scan.")
        return
    ranked = _score_and_persist(universe)
    n = 0 if ranked is None else len(ranked)
    logger.info("sunday_stock_scan: persisted {} scored symbols", n)
    return f"scored={n}"


@_guarded("sunday_ml_retrain")
def sunday_ml_retrain() -> None:
    logger.info("sunday_ml_retrain: rolling retrain of weekly stock model")
    try:
        from ml.retrain import run_rolling_retrain
    except Exception as exc:
        logger.warning("sunday_ml_retrain: ml stack unavailable ({}); skipping.", exc)
        return "skipped: ml unavailable"
    # Multi-year history for the retrain (more training examples); the weekly
    # scan uses the shorter default window.
    universe = _load_feature_universe(lookback_days=int(getattr(settings, "ml_lookback_days", 1095)))
    if not universe:
        logger.warning("sunday_ml_retrain: empty feature universe. Skipping retrain.")
        return
    result = run_rolling_retrain(
        universe, model_path=MODEL_PATH, performance_writer=_write_model_performance,
    )
    if not result.ok:
        logger.warning("sunday_ml_retrain: skipped ({})", result.reason)
        return f"skipped: {result.reason}"
    # Judge on the walk-forward MEAN when we have folds: a single holdout can
    # clear the bar by luck, and a model promoted on luck quietly degrades the
    # rankings for a week. Fall back to the holdout AUC only if folds are
    # unavailable (short history).
    metrics = result.metrics
    folds = int(metrics.get("n_folds", 0) or 0)
    auc = metrics.get("auc_mean") if folds >= 2 else metrics.get("auc")
    basis = f"walk-forward mean of {folds} folds" if folds >= 2 else "holdout"
    if auc is not None and float(auc) < ML_MIN_AUC:
        _quarantine_model(float(auc))
        return (f"rejected auc={float(auc):.3f} < {ML_MIN_AUC} "
                f"({basis}, rule-only scoring)")
    logger.info("sunday_ml_retrain: ok train={} test={} auc={} ({}) -> {}",
                result.n_train, result.n_test, auc, basis, MODEL_PATH)
    spread = metrics.get("auc_std")
    extra = ""
    try:
        if spread is not None and float(spread) == float(spread):   # not NaN
            extra = f" std={float(spread):.3f}"
    except (TypeError, ValueError):
        pass
    return f"retrained auc={float(auc):.3f} ({basis}{extra})" if auc is not None else "retrained"


def _quarantine_model(auc: float) -> None:
    """Move a below-threshold model artifact aside so _load_ml_scorer cannot
    pick it up. The scan then runs rule-only until a retrain clears the bar."""
    logger.warning("sunday_ml_retrain: model REJECTED (auc={:.3f} < {}); "
                   "quarantining artifact - scan stays rule-only.", auc, ML_MIN_AUC)
    for suffix in (".json", ".meta.json"):
        path = Path(str(MODEL_PATH) + suffix)
        try:
            if path.exists():
                path.replace(path.with_suffix(path.suffix + ".rejected"))
        except Exception as exc:
            logger.warning("_quarantine_model: could not move {} ({})", path, exc)


@_guarded("monday_stock_buys")
def monday_stock_buys() -> None:
    # The weekly rotation was falsified by backtest (-1.6% over 3.3y vs SPY
    # +79%). It stays in the codebase for comparison but must not trade
    # alongside momentum, or the two strategies fight over the same cash.
    if getattr(settings, "stock_strategy", "momentum") != "rotation":
        return "skipped: STOCK_STRATEGY != rotation"
    rm = build_risk_manager()
    if _rm_halted(rm):
        logger.warning("monday_stock_buys: RiskManager halted (monthly DD). No entries.")
        return "halted"
    scored = db.get_latest_scored_universe()
    if scored is None or len(scored) == 0:
        logger.warning("monday_stock_buys: no scored universe. Run sunday_stock_scan first.")
        return "no scored universe"
    universe = _load_feature_universe()
    if not universe:
        logger.warning("monday_stock_buys: no feature universe available. No entries.")
        return "no feature universe"
    state = get_account_state(rm, "stock")
    executor = StockExecutor.from_settings()
    executor.paper_cash = state.cash
    executor.min_position_usd = effective_min_position(state.equity)
    result = stock_weekly.run_stock_weekly_buys(
        scored_df=scored,
        universe=universe,
        equity=state.equity,
        available_cash=state.cash,
        executor=executor,
        risk_manager=rm,
        open_positions=_position_dicts(state.open_positions),
    )
    filled = sum(1 for r in getattr(result, "executed", []) if r.get("status") == "filled")
    logger.info("monday_stock_buys: complete -> {} entries, {} filled",
                len(getattr(result, "entries", [])), filled)
    return f"entries={len(getattr(result, 'entries', []))} filled={filled}"


@_guarded("midweek_stock_monitor")
def midweek_stock_monitor() -> None:
    # Locked strategy: stop-loss monitoring only. No over-managing.
    positions = _open_positions("stock")
    if not positions:
        return "no open positions"
    symbols = [getattr(p, "symbol", "") for p in positions if getattr(p, "symbol", "")]
    frames = _load_stock_frames(symbols, lookback_days=10)
    prices = _latest_closes(frames)
    if prices:
        db.update_position_marks(prices)
    executor = StockExecutor.from_settings()
    closed = 0
    for p in positions:
        symbol = getattr(p, "symbol", "")
        entry = float(getattr(p, "entry_price", 0.0) or 0.0)
        stop = float(getattr(p, "stop_loss", 0.0) or 0.0)
        mark = prices.get(symbol) or getattr(p, "current_price", None)
        if entry <= 0 or not mark:
            continue
        mark = float(mark)
        stop_level = stop if stop > 0 else entry * (1.0 - settings.stock_stop_loss)
        if mark <= stop_level:
            logger.warning("midweek stop hit {} entry={} stop={} mark={}",
                           symbol, entry, stop_level, mark)
            result = executor.close_long(symbol, exit_price=mark, reason="stop_loss")
            if result.get("status") == "filled":
                closed += 1
    if closed:
        logger.info("midweek_stock_monitor: closed {} stopped-out position(s)", closed)
    return f"marks={len(prices)} closed={closed}"


def _synthesize_missing_marks(universe: dict, positions: list) -> dict:
    """Fallback marks so an exit is never priced at 0 when data is missing:
    synthesize a one-row frame from the last known mark (or entry)."""
    import pandas as pd
    for p in positions:
        symbol = getattr(p, "symbol", "")
        if symbol and symbol not in universe:
            mark = float(getattr(p, "current_price", None)
                         or getattr(p, "entry_price", 0.0) or 0.0)
            if mark > 0:
                universe[symbol] = pd.DataFrame({"close": [mark]})
    return universe


@_guarded("friday_stock_sells")
def friday_stock_sells() -> None:
    """Friday 3:45 exit pass (rotation strategy only).

    Two modes (STOCK_EXIT_MODE):

    rotate (default) - re-score the universe NOW and sell only positions
      that fell out of the top ROTATION_KEEP_RANK; still-ranked winners ride
      into next week (stop-losses keep protecting them, weekends included via
      Monday-morning marks). Cuts turnover and lets momentum compound.
    liquidate - legacy: sell everything, flat over the weekend.
    """
    if getattr(settings, "stock_strategy", "momentum") != "rotation":
        return "skipped: STOCK_STRATEGY != rotation"
    positions = _open_positions("stock")
    if not positions:
        logger.info("friday_stock_sells: no open stock positions.")
        return "no open positions"
    mode = getattr(settings, "stock_exit_mode", "rotate")

    if mode == "rotate":
        universe = _load_feature_universe()
        ranked = _score_and_persist(universe) if universe else None
        to_sell = stock_weekly.select_rotation_exits(
            _position_dicts(positions),
            ranked,
            keep_rank=int(getattr(settings, "rotation_keep_rank", 20)),
        )
        held = len(positions) - len(to_sell)
        if not to_sell:
            logger.info("friday_stock_sells (rotate): holding all {} position(s)", len(positions))
            return f"mode=rotate held={held} sold=0"
        universe = _synthesize_missing_marks(universe or {}, positions)
        executor = StockExecutor.from_settings()
        result = stock_weekly.run_stock_weekly_sells(
            open_positions=to_sell, universe=universe, executor=executor,
        )
        filled = sum(1 for r in getattr(result, "executed", []) if r.get("status") == "filled")
        logger.info("friday_stock_sells (rotate): held {} / sold {} ({} filled)",
                    held, len(to_sell), filled)
        return f"mode=rotate held={held} sold={len(to_sell)} filled={filled}"

    symbols = [getattr(p, "symbol", "") for p in positions if getattr(p, "symbol", "")]
    universe = _synthesize_missing_marks(_load_stock_frames(symbols, lookback_days=90), positions)
    executor = StockExecutor.from_settings()
    result = stock_weekly.run_stock_weekly_sells(
        open_positions=_position_dicts(positions),
        universe=universe,
        executor=executor,
    )
    filled = sum(1 for r in getattr(result, "executed", []) if r.get("status") == "filled")
    logger.info("friday_stock_sells: complete -> {} exits, {} filled",
                len(getattr(result, "exits", [])), filled)
    return f"exits={len(getattr(result, 'exits', []))} filled={filled}"


@_guarded("crypto_cycle")
def crypto_cycle() -> None:
    rm = build_risk_manager()
    if _rm_halted(rm):
        logger.warning("crypto_cycle: RiskManager halted (monthly DD). No crypto entries.")
        return "halted"
    universe = _load_crypto_universe()
    if not universe:
        logger.warning("crypto_cycle: empty crypto universe (data fetch failed). Skipping.")
        return "no data"
    prices = _latest_closes(universe)
    if prices:
        db.update_position_marks(prices)
    state = get_account_state(rm, "crypto")
    funding = _load_crypto_funding()
    result = crypto_24h.run_crypto_24h_pipeline(
        universe=universe,
        funding_rates=funding,
        open_positions=_position_dicts(state.open_positions),
        equity=state.equity,
        available_cash=state.cash,
        btc_only=_btc_only(),
        risk_manager=rm,
        entry_mode=getattr(settings, "crypto_entry_mode", "regime"),
    )
    executor = crypto_executor_from_settings()
    executor.paper_cash = state.cash
    executor.min_position_usd = effective_min_position(state.equity)
    closed = opened = 0
    for plan in getattr(result, "exits", []):
        exit_price = float(plan.close_price or 0.0) or prices.get(plan.symbol, 0.0)
        if exit_price <= 0:
            logger.warning("crypto_cycle: no price for exit {}; skipping this cycle", plan.symbol)
            continue
        r = executor.close_long(plan.symbol, exit_price=exit_price, reason=plan.reason)
        closed += 1 if r.get("status") == "filled" else 0
    for plan in getattr(result, "entries", []):
        r = executor.open_long(
            symbol=plan.symbol,
            units=plan.units,
            entry_price=plan.entry_price,
            stop_loss=plan.stop_price,
            take_profit=plan.take_profit,
        )
        opened += 1 if r.get("status") == "filled" else 0
    logger.info("crypto_cycle: complete -> {} entries ({} filled), {} exits ({} filled)",
                len(getattr(result, "entries", [])), opened,
                len(getattr(result, "exits", [])), closed)
    # Always say WHY a cycle opened nothing: a silent "opened=0" hid a
    # symbol-matching bug that made crypto entries impossible for two weeks.
    summary = ""
    try:
        summary = result.gate_summary()
    except Exception:
        pass
    if summary:
        logger.info("crypto_cycle: gates -> {}", summary)
    detail = f"opened={opened} closed={closed}"
    return f"{detail} [{summary}]" if summary else detail


def _latest_crypto_prices(symbols: list[str]) -> dict:
    """{position symbol: latest price}. One keyless Alpaca request; symbols
    are mapped to /USD pairs for the query and keyed back to the originals
    (positions may carry Binance-style /USDT names)."""
    if not symbols:
        return {}
    try:
        from data.alpaca_data import fetch_latest_crypto_prices_alpaca
        mapped = {s: to_alpaca_symbol(s) for s in symbols}
        fetched = fetch_latest_crypto_prices_alpaca(list(set(mapped.values())))
        return {orig: fetched[alp] for orig, alp in mapped.items() if alp in fetched}
    except Exception as exc:
        logger.warning("_latest_crypto_prices failed ({})", exc)
        return {}


@_guarded("crypto_stop_monitor")
def crypto_stop_monitor() -> None:
    """Risk-reducing: between 4h cycles, check crypto stops every 15 minutes
    so a sharp move can't run unprotected for hours. Never paused. Makes no
    API call at all while there are no open crypto positions."""
    positions = _open_positions("crypto")
    if not positions:
        return "no open positions"
    symbols = [getattr(p, "symbol", "") for p in positions if getattr(p, "symbol", "")]
    prices = _latest_crypto_prices(symbols)
    if prices:
        db.update_position_marks(prices)
    executor = crypto_executor_from_settings()
    closed = 0
    for p in positions:
        symbol = getattr(p, "symbol", "")
        stop = float(getattr(p, "stop_loss", 0.0) or 0.0)
        mark = prices.get(symbol) or getattr(p, "current_price", None)
        if stop <= 0 or not mark:
            continue
        if float(mark) <= stop:
            logger.warning("crypto stop hit {} stop={} mark={}", symbol, stop, mark)
            r = executor.close_long(symbol, exit_price=float(mark), reason="stop_loss")
            closed += 1 if r.get("status") == "filled" else 0
    if closed:
        logger.info("crypto_stop_monitor: closed {} stopped-out position(s)", closed)
    return f"marks={len(prices)} closed={closed}"


@_guarded("monthly_momentum_rebalance")
def monthly_momentum_rebalance() -> None:
    """The live momentum strategy: rank on 12-1 momentum, hold the top N
    equal-weight, rebalance monthly. Runs only when STOCK_STRATEGY=momentum
    (the default); the legacy weekly rotation jobs no-op in that mode."""
    if getattr(settings, "stock_strategy", "momentum") != "momentum":
        return "skipped: STOCK_STRATEGY != momentum"
    rm = build_risk_manager()
    if _rm_halted(rm):
        logger.warning("monthly_momentum_rebalance: drawdown halt active. No entries.")
        return "halted"
    # Momentum needs ~13 months of history to rank, so pull a wider window
    # than the weekly scan ever needed.
    universe = _load_feature_universe(lookback_days=600)
    if not universe:
        logger.warning("monthly_momentum_rebalance: no data. Skipping.")
        return "no data"
    state = get_account_state(rm, "stock")
    executor = StockExecutor.from_settings()
    executor.paper_cash = state.cash
    executor.min_position_usd = effective_min_position(state.equity)
    result = momentum_monthly.run_monthly_rebalance(
        universe=universe,
        open_positions=_position_dicts(state.open_positions),
        equity=state.equity,
        available_cash=state.cash,
        executor=executor,
        top_n=int(getattr(settings, "momentum_top_n", 10)),
        lookback=int(getattr(settings, "momentum_lookback", 252)),
        skip=int(getattr(settings, "momentum_skip", 21)),
        min_position_usd=executor.min_position_usd,
    )
    logger.info("monthly_momentum_rebalance: {}", result)
    return f"sold={result.get('sold', 0)} bought={result.get('bought', 0)}"


@_guarded("weekly_performance_report")
def weekly_performance_report() -> None:
    logger.info("weekly_performance_report: building weekly P&L report")
    from reporting import build_weekly_report, write_report
    rm = build_risk_manager()
    snap = get_account_snapshot(rm)
    report = build_weekly_report(
        nav=snap.equity,
        stock_positions=_open_positions("stock"),
        crypto_positions=_open_positions("crypto"),
        environment=getattr(settings, "environment", "paper"),
    )
    path = write_report(report)
    logger.info("weekly_performance_report: wrote {} (net={} trades={} win_rate={})",
                path, report.net_pnl, report.total_trades, report.win_rate)
    try:
        from monitoring import alert_manager
        alert_manager.send_weekly_summary(
            week=report.week_index,
            net_pnl=report.net_pnl,
            win_rate=report.win_rate,
            total_trades=report.total_trades,
        )
    except Exception as exc:
        logger.warning("weekly_performance_report: telegram summary failed: {}", exc)
    return f"net={report.net_pnl}"


@_guarded("daily_heartbeat")
def daily_heartbeat() -> None:
    rm = build_risk_manager()
    snap = get_account_snapshot(rm)
    stock_n = len(_open_positions("stock"))
    crypto_n = len(_open_positions("crypto"))
    status = "HALTED" if _rm_halted(rm) else ("PAUSED" if bot_state.paused else "OK")
    msg = ("*Heartbeat* {}\nNAV: ${:,.2f} ({})\nOpen: {} stock / {} crypto").format(
        status, snap.equity, snap.source, stock_n, crypto_n)
    try:
        from monitoring import alert_manager
        alert_manager.send(msg)
    except Exception as exc:
        logger.warning("daily_heartbeat: telegram send failed: {}", exc)
    logger.info("daily_heartbeat: {} nav={} ({})", status, snap.equity, snap.source)
    return f"{status} nav={snap.equity:.2f}"


# ============================================================
# REGISTRATION
# ============================================================

JOBS = {
    "sunday_stock_scan": sunday_stock_scan,
    "sunday_ml_retrain": sunday_ml_retrain,
    "monday_stock_buys": monday_stock_buys,
    "midweek_stock_monitor": midweek_stock_monitor,
    "friday_stock_sells": friday_stock_sells,
    "monthly_momentum_rebalance": monthly_momentum_rebalance,
    "crypto_cycle": crypto_cycle,
    "crypto_stop_monitor": crypto_stop_monitor,
    "weekly_performance_report": weekly_performance_report,
    "daily_heartbeat": daily_heartbeat,
}


def register_jobs(scheduler) -> None:
    tz = MARKET_TZ
    scheduler.add_job(sunday_stock_scan,
                      CronTrigger(day_of_week="sun", hour=20, minute=0, timezone=tz),
                      id="sunday_stock_scan", replace_existing=True)
    scheduler.add_job(sunday_ml_retrain,
                      CronTrigger(day_of_week="sun", hour=21, minute=0, timezone=tz),
                      id="sunday_ml_retrain", replace_existing=True)
    scheduler.add_job(monday_stock_buys,
                      CronTrigger(day_of_week="mon", hour=9, minute=45, timezone=tz),
                      id="monday_stock_buys", replace_existing=True)
    scheduler.add_job(midweek_stock_monitor,
                      CronTrigger(day_of_week="mon-fri", hour="9-16", minute="*/30", timezone=tz),
                      id="midweek_stock_monitor", replace_existing=True)
    scheduler.add_job(friday_stock_sells,
                      CronTrigger(day_of_week="fri", hour=15, minute=45, timezone=tz),
                      id="friday_stock_sells", replace_existing=True)
    # First Monday of each month, just after the open.
    scheduler.add_job(monthly_momentum_rebalance,
                      CronTrigger(day="1-7", day_of_week="mon", hour=9, minute=45,
                                  timezone=tz),
                      id="monthly_momentum_rebalance", replace_existing=True)
    scheduler.add_job(crypto_cycle,
                      CronTrigger(hour="*/4", minute=0, timezone=tz),
                      id="crypto_cycle", replace_existing=True)
    scheduler.add_job(crypto_stop_monitor,
                      CronTrigger(minute="*/15", timezone=tz),
                      id="crypto_stop_monitor", replace_existing=True)
    scheduler.add_job(weekly_performance_report,
                      CronTrigger(day_of_week="fri", hour=16, minute=30, timezone=tz),
                      id="weekly_performance_report", replace_existing=True)
    scheduler.add_job(daily_heartbeat,
                      CronTrigger(hour=8, minute=0, timezone=tz),
                      id="daily_heartbeat", replace_existing=True)


def build_scheduler():
    if BackgroundScheduler is None:
        raise RuntimeError("APScheduler not installed")
    scheduler = BackgroundScheduler(timezone=MARKET_TZ)
    register_jobs(scheduler)
    logger.info("scheduler: {} jobs registered", len(scheduler.get_jobs()))
    return scheduler
