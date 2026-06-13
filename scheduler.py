"""scheduler.py - Phase 8: weekly reporting + daily heartbeat wired in.

APScheduler BackgroundScheduler. main.py owns the process loop and watchdog;
this module builds and returns a configured scheduler with every job registered.

Account state (equity / cash / open_positions) is reconstructed from the DB plus
RiskManager (option b). Broker reads are deferred to the Phase 10 account layer.
Everything funnels through the ADAPTER LAYER below, so swapping to broker or hybrid
is a one-section change that never touches job logic.

Phase 7 additions (ADAPTER LAYER, jobs degrade to safe no-ops until Phase 2 wired):
  - _load_feature_universe(), _load_ml_scorer(), _write_model_performance().
  - sunday_stock_scan ML-blended scan; sunday_ml_retrain rolling retrain.

Phase 8 additions (jobs only; reporting + alerts imported lazily so scheduler
import stays bulletproof):
  - weekly_performance_report: builds the weekly P&L report from the reporting
    package, writes a static HTML file, sends a Telegram weekly summary.
  - daily_heartbeat: one-line Telegram NAV/positions heartbeat. Best-effort.

Locked-strategy notes honored here:
  - Stocks: buy Mon 9:45 ET, sell Fri 3:45 ET, Sunday-night scan.
  - Mid-week: stop-loss monitoring only. No over-managing (SPY breaker is log-only).
  - Crypto: 24/7 4H cycle, BTC-only until 60 days profitable.
  - Every trading job is gated on RiskManager.is_halted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from config.settings import settings
from database import db
from risk.manager import RiskManager, MONTHLY_DRAWDOWN_HALT
from execution.alpaca import StockExecutor
from strategies import stock_scorer, stock_weekly, crypto_24h

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
except Exception:  # pragma: no cover - apscheduler present on target machine
    BackgroundScheduler = None
    CronTrigger = None

MARKET_TZ = "America/New_York"  # ET; APScheduler resolves EST/EDT. Strategy says EST.
STOCK_STOP_LOSS = 0.05
CRYPTO_UNIVERSE = ["BTC/USDT"]  # BTC-only gate (TODO #3) until 60 days profitable.

# Phase 7: trained weekly stock model artifact (booster at MODEL_PATH + ".json",
# metadata at MODEL_PATH + ".meta.json"). TrainedModel.save creates the dir.
MODEL_PATH = "models/weekly_stock"


@dataclass
class AccountState:
    equity: float
    cash: float
    open_positions: list


# ============================================================
# ADAPTER LAYER - runtime account state (option b: DB reconstruct)
# Swap to broker (a) or hybrid (c) here in Phase 10. Jobs below never change.
#
# Assumed db.py surface - rename these to match your Phase 1 accessors if needed:
#   db.get_open_positions(asset_type)   -> list of Position(.symbol .quantity .entry_price .current_price)
#   db.get_latest_scored_universe()     -> DataFrame persisted by the Sunday scan (or None)
#   db.persist_scored_universe(df)      -> None
#   db.get_feature_universe()           -> {ticker: DataFrame} Phase 2 validated features (or None)
#   db.persist_model_performance(dict)  -> None  (writes a ModelPerformance row)
#
# RiskManager is assumed to self-load NAV/heat from the same DB on construction
# (Phase 5: NAV = capital + closed-trade PnL + open marks). If yours must be fed
# trades/positions, add that seed call inside build_risk_manager() only.
# ============================================================

def build_risk_manager() -> RiskManager:
    return RiskManager(starting_capital=settings.starting_capital,
                       monthly_halt=MONTHLY_DRAWDOWN_HALT)


def _open_positions(asset_type: str) -> list:
    return list(db.get_open_positions(asset_type))


def _available_cash(rm: RiskManager, open_positions: list) -> float:
    invested = 0.0
    for p in open_positions:
        mark = getattr(p, "current_price", None) or getattr(p, "entry_price", 0.0)
        invested += float(getattr(p, "quantity", 0.0)) * float(mark or 0.0)
    return max(0.0, float(rm.current_nav) - invested)


def get_account_state(rm: RiskManager, asset_type: str) -> AccountState:
    positions = _open_positions(asset_type)
    cash = _available_cash(rm, positions)
    return AccountState(equity=float(rm.current_nav), cash=cash, open_positions=positions)


def _symbols(scored) -> list:
    """Extract the symbol list from the persisted scored universe.

    The scorer's ranked frame uses a 'ticker' column; older/persisted shapes may
    use 'symbol'. Accept either. Falls back to iterating a bare sequence.
    """
    if scored is None:
        return []
    cols = getattr(scored, "columns", None)
    if cols is not None:
        cols = list(cols)
        for key in ("ticker", "symbol"):
            if key in cols:
                return list(scored[key])
    try:
        return list(scored)
    except TypeError:
        return []


# ----- Phase 7 ML adapter chokepoints -----------------------------------

def _load_feature_universe() -> dict:
    """ADAPTER: Phase 2 validated feature universe as {ticker: DataFrame}.

    This is the single place to wire the Phase 2 pipeline output that both the
    scan and the retrain consume. Returns {} (safe no-op) until that accessor is
    wired, so neither job can crash the scheduler thread before Phase 2 lands.
    """
    getter = getattr(db, "get_feature_universe", None)
    if getter is None:
        logger.warning("_load_feature_universe: no Phase 2 feature source wired. "
                       "Returning empty universe (scan/retrain will no-op).")
        return {}
    try:
        universe = getter()
        return dict(universe) if universe else {}
    except Exception as exc:
        logger.exception("_load_feature_universe failed: {}", exc)
        return {}


def _load_ml_scorer():
    """ADAPTER: load the trained ML model for the composite blend.

    Returns an ml.model.MLScorer, or None for pure rule scoring when no model is
    on disk yet (e.g. before the first successful retrain) or load fails.
    """
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
    """ADAPTER: persist retrain metrics to the ModelPerformance table.

    Best-effort; never raises into the retrain job. Assumes
    db.persist_model_performance(dict). Wire to the real accessor if the name
    differs. No schema changes (ModelPerformance already exists - Phase 1).
    """
    writer = getattr(db, "persist_model_performance", None)
    if writer is None:
        logger.info("_write_model_performance: no DB writer wired; metrics={}", metrics)
        return
    try:
        writer(metrics)
    except Exception as exc:
        logger.exception("_write_model_performance failed: {}", exc)


# ============================================================
# JOBS
# ============================================================

def sunday_stock_scan() -> None:
    logger.info("sunday_stock_scan: scoring S&P 500 universe")
    try:
        universe = _load_feature_universe()
        if not universe:
            logger.warning("sunday_stock_scan: empty feature universe. Skipping scan.")
            return
        ml_scorer = _load_ml_scorer()
        ml_weight = stock_scorer.DEFAULT_ML_WEIGHT if ml_scorer is not None else 0.0
        result = stock_scorer.score_universe(
            universe,
            ml_scorer=ml_scorer,
            ml_weight=ml_weight,
        )
        ranked = getattr(result, "ranked", result)
        db.persist_scored_universe(ranked)
        n = 0 if ranked is None else len(ranked)
        logger.info("sunday_stock_scan: persisted {} scored symbols (ml_blend={})",
                    n, ml_scorer is not None)
    except Exception as exc:  # never let a scan crash the scheduler thread
        logger.exception("sunday_stock_scan failed: {}", exc)


def sunday_ml_retrain() -> None:
    logger.info("sunday_ml_retrain: rolling retrain of weekly stock model")
    try:
        from ml.retrain import run_rolling_retrain
        universe = _load_feature_universe()
        if not universe:
            logger.warning("sunday_ml_retrain: empty feature universe. Skipping retrain.")
            return
        result = run_rolling_retrain(
            universe,
            model_path=MODEL_PATH,
            performance_writer=_write_model_performance,
        )
        if result.ok:
            logger.info("sunday_ml_retrain: ok train={} test={} auc={} -> {}",
                        result.n_train, result.n_test,
                        result.metrics.get("auc"), MODEL_PATH)
        else:
            logger.warning("sunday_ml_retrain: skipped ({})", result.reason)
    except Exception as exc:  # never let retrain crash the scheduler thread
        logger.exception("sunday_ml_retrain failed: {}", exc)


def monday_stock_buys() -> None:
    rm = build_risk_manager()
    if rm.is_halted:
        logger.warning("monday_stock_buys: RiskManager halted (monthly DD). No entries.")
        return
    scored = db.get_latest_scored_universe()
    if scored is None or len(scored) == 0:
        logger.warning("monday_stock_buys: no scored universe. Run sunday_stock_scan first.")
        return
    state = get_account_state(rm, "stock")
    executor = StockExecutor.from_settings()
    result = stock_weekly.run_stock_weekly_buys(
        scored_df=scored,
        universe=_symbols(scored),
        equity=state.equity,
        cash=state.cash,
        executor=executor,
        risk_manager=rm,
    )
    logger.info("monday_stock_buys: complete -> {}", result)


def midweek_stock_monitor() -> None:
    # Locked strategy: stop-loss monitoring only. No over-managing.
    positions = _open_positions("stock")
    if not positions:
        return
    executor = StockExecutor.from_settings()
    closed = 0
    for p in positions:
        entry = float(getattr(p, "entry_price", 0.0) or 0.0)
        mark = getattr(p, "current_price", None)
        if entry <= 0 or mark is None:
            continue
        if float(mark) <= entry * (1.0 - STOCK_STOP_LOSS):
            logger.warning("midweek stop hit {} entry={} mark={}",
                           getattr(p, "symbol", "?"), entry, mark)
            executor.close_long(getattr(p, "symbol", None))
            closed += 1
    _spy_breaker_check()
    if closed:
        logger.info("midweek_stock_monitor: closed {} stopped-out position(s)", closed)


def _spy_breaker_check() -> None:
    # Optional market circuit breaker. Log-only by design: locked strategy says
    # "no over-managing" mid-week. Promote to a hard flatten only if Phase 9
    # validation supports it.
    return


def friday_stock_sells() -> None:
    rm = build_risk_manager()  # constructed for symmetry / future logging
    positions = _open_positions("stock")
    if not positions:
        logger.info("friday_stock_sells: no open stock positions.")
        return
    executor = StockExecutor.from_settings()
    result = stock_weekly.run_stock_weekly_sells(open_positions=positions, executor=executor)
    logger.info("friday_stock_sells: complete -> {}", result)


def crypto_cycle() -> None:
    rm = build_risk_manager()
    if rm.is_halted:
        logger.warning("crypto_cycle: RiskManager halted (monthly DD). No crypto entries.")
        return
    state = get_account_state(rm, "crypto")
    funding = {}
    if hasattr(crypto_24h, "get_funding_rates"):
        try:
            funding = crypto_24h.get_funding_rates(CRYPTO_UNIVERSE)
        except Exception as exc:
            logger.warning("crypto_cycle: funding fetch failed ({}); proceeding with empty.", exc)
    result = crypto_24h.run_crypto_24h_pipeline(
        universe=CRYPTO_UNIVERSE,
        funding=funding,
        open_positions=state.open_positions,
        equity=state.equity,
        cash=state.cash,
        risk_manager=rm,
    )
    logger.info("crypto_cycle: complete -> {}", result)


def weekly_performance_report() -> None:
    logger.info("weekly_performance_report: building weekly P&L report")
    try:
        from reporting import build_weekly_report, write_report
        rm = build_risk_manager()
        report = build_weekly_report(
            nav=rm.current_nav,
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
    except Exception as exc:  # never let reporting crash the scheduler thread
        logger.exception("weekly_performance_report failed: {}", exc)


def daily_heartbeat() -> None:
    logger.info("daily_heartbeat: sending status heartbeat")
    try:
        rm = build_risk_manager()
        stock_n = len(_open_positions("stock"))
        crypto_n = len(_open_positions("crypto"))
        status = "HALTED" if rm.is_halted else "OK"
        msg = ("*Heartbeat* {}\nNAV: ${:,.2f}\nOpen: {} stock / {} crypto").format(
            status, float(rm.current_nav), stock_n, crypto_n)
        try:
            from monitoring import alert_manager
            alert_manager.send(msg)
        except Exception as exc:
            logger.warning("daily_heartbeat: telegram send failed: {}", exc)
        logger.info("daily_heartbeat: {} nav={}", status, rm.current_nav)
    except Exception as exc:  # never let the heartbeat crash the scheduler thread
        logger.exception("daily_heartbeat failed: {}", exc)


# ============================================================
# REGISTRATION
# ============================================================

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
    scheduler.add_job(crypto_cycle,
                      CronTrigger(hour="*/4", minute=0, timezone=tz),
                      id="crypto_cycle", replace_existing=True)
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