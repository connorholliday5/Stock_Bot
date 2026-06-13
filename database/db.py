# ============================================================
# database/db.py
# PostgreSQL connection + full schema
# All tables defined here upfront - no schema changes later
# ============================================================

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    DateTime, Boolean, Text, Enum, UniqueConstraint, select, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.sql import func
from contextlib import contextmanager
from loguru import logger
import enum

from config import settings

Base = declarative_base()
engine = create_engine(settings.database_url, pool_pre_ping=True, pool_size=5)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# ============================================================
# Enums
# ============================================================

class AssetType(str, enum.Enum):
    STOCK = "stock"
    CRYPTO = "crypto"

class OrderSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(str, enum.Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class SignalDirection(str, enum.Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

class MarketRegime(str, enum.Enum):
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOL = "high_vol"
    UNKNOWN = "unknown"


# ============================================================
# Tables
# ============================================================

class Trade(Base):
    """Every executed trade, both stocks and crypto"""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_type = Column(Enum(AssetType), nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(Enum(OrderSide), nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    status = Column(Enum(OrderStatus), nullable=False, default=OrderStatus.PENDING)
    pnl = Column(Float, nullable=True)
    pnl_pct = Column(Float, nullable=True)
    broker_order_id = Column(String(100), nullable=True)
    opened_at = Column(DateTime(timezone=True), nullable=True)
    closed_at = Column(DateTime(timezone=True), nullable=True)
    close_reason = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Signal(Base):
    """Every generated signal - logged regardless of whether trade is placed"""
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_type = Column(Enum(AssetType), nullable=False)
    symbol = Column(String(20), nullable=False)
    direction = Column(Enum(SignalDirection), nullable=False)
    composite_score = Column(Float, nullable=True)
    rsi = Column(Float, nullable=True)
    sma_50 = Column(Float, nullable=True)
    sma_200 = Column(Float, nullable=True)
    adx = Column(Float, nullable=True)
    volume_ratio = Column(Float, nullable=True)
    momentum_20d = Column(Float, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    ml_prediction = Column(String(10), nullable=True)
    ml_confidence = Column(Float, nullable=True)
    acted_on = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Position(Base):
    """Current open positions"""
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_type = Column(Enum(AssetType), nullable=False)
    symbol = Column(String(20), nullable=False, unique=True)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, nullable=True)
    unrealized_pnl_pct = Column(Float, nullable=True)
    opened_at = Column(DateTime(timezone=True), nullable=False)
    week_number = Column(Integer, nullable=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class WeeklyPerformance(Base):
    """Weekly P&L summary for stocks"""
    __tablename__ = "weekly_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    week_number = Column(Integer, nullable=False)
    year = Column(Integer, nullable=False)
    asset_type = Column(Enum(AssetType), nullable=False)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    gross_pnl = Column(Float, default=0.0)
    net_pnl = Column(Float, default=0.0)
    win_rate = Column(Float, nullable=True)
    sharpe = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    starting_capital = Column(Float, nullable=True)
    ending_capital = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (UniqueConstraint("week_number", "year", "asset_type"),)


class ModelPerformance(Base):
    """ML model accuracy tracking"""
    __tablename__ = "model_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(50), nullable=False)
    asset_type = Column(Enum(AssetType), nullable=False)
    version = Column(String(20), nullable=False)
    train_accuracy = Column(Float, nullable=True)
    val_accuracy = Column(Float, nullable=True)
    live_accuracy = Column(Float, nullable=True)
    feature_count = Column(Integer, nullable=True)
    training_rows = Column(Integer, nullable=True)
    is_active = Column(Boolean, default=False)
    trained_at = Column(DateTime(timezone=True), server_default=func.now())


class BotLog(Base):
    """System events, errors, and key decisions"""
    __tablename__ = "bot_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    level = Column(String(10), nullable=False)
    module = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ============================================================
# Connection Utilities
# ============================================================

@contextmanager
def get_db() -> Session:
    """Context manager for database sessions"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        db.close()


def init_db():
    """Create all tables - run once on startup"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def health_check() -> bool:
    """Verify database connection is alive"""
    try:
        with get_db() as db:
            db.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


# ============================================================
# Query accessors (Phase 9) - READS
# Read-only. These DO NOT use get_db() because get_db() commits on
# exit and the sessionmaker uses the default expire_on_commit=True;
# that would expire every attribute and the detached rows returned
# here would raise DetachedInstanceError on first attribute read.
# Instead: open a session, load all columns, expunge (detach with
# loaded values intact), then close without committing. Callers
# (reporting/pnl.py) read plain columns via getattr, which is safe
# on a detached instance once the columns are already loaded.
# ============================================================

def get_closed_trades(since=None):
    """Closed trades, optionally on/after `since` (a datetime). A trade is
    'closed' when closed_at is set. Returns a list of detached Trade rows;
    reporting re-sorts and re-filters, so order does not matter here."""
    db = SessionLocal()
    try:
        stmt = select(Trade).where(Trade.closed_at.isnot(None))
        if since is not None:
            stmt = stmt.where(Trade.closed_at >= since)
        rows = list(db.scalars(stmt).all())
        db.expunge_all()
        return rows
    finally:
        db.close()


def get_open_positions(asset_type=None):
    """Open positions (the positions table holds only open rows), optionally
    filtered by asset_type. Accepts an AssetType or the raw 'stock'/'crypto'
    string. Returns a list of detached Position rows."""
    if isinstance(asset_type, str):
        try:
            asset_type = AssetType(asset_type)
        except ValueError:
            pass
    db = SessionLocal()
    try:
        stmt = select(Position)
        if asset_type is not None:
            stmt = stmt.where(Position.asset_type == asset_type)
        rows = list(db.scalars(stmt).all())
        db.expunge_all()
        return rows
    finally:
        db.close()


def get_model_performance():
    """All ModelPerformance rows. Reporting limits/sorts (newest-first, 8).
    Returns a list of detached ModelPerformance rows."""
    db = SessionLocal()
    try:
        rows = list(db.scalars(select(ModelPerformance)).all())
        db.expunge_all()
        return rows
    finally:
        db.close()


def get_feature_universe():
    """{ticker: DataFrame} of Phase 2 validated features for the Sunday scorer
    and ML retrain (HANDOFF TODO 6). Served FRESH from the fetcher so no schema
    change or new table is introduced (respects 'no schema changes until
    Phase 10').

    Returns {} on any failure so sunday_stock_scan / sunday_ml_retrain no-op
    cleanly instead of killing the scheduler thread.

    WARNING (Phase 9): this fires a live, full-universe Polygon pull with retry
    backoff. If the Polygon key is unauthorized it is slow and yields nothing.
    The dry_run harness can skip the data jobs with --skip scan,retrain."""
    try:
        from data.fetcher import fetch_stock_universe
        # CONFIRM-ME: fetch_stock_universe() is called zero-arg. If it requires
        # a ticker list or lookback window, pass them here.
        return fetch_stock_universe() or {}
    except Exception as e:
        logger.warning(f"get_feature_universe: fetch failed ({e}); returning empty universe")
        return {}


# ============================================================
# Query accessors (Phase 9) - WRITES + scored-universe round-trip
# persist_scored_universe / get_latest_scored_universe close the
# Sunday-scan -> Monday-buy round-trip. Persisted LOSSLESSLY as JSON in
# a BotLog row (module="scored_universe") rather than the Signal table,
# because run_stock_weekly_buys needs the full scored frame (e.g. close
# for sizing) which Signal does not store. This respects the invariant
# "no schema changes until Phase 10"; Phase 10 should replace it with a
# dedicated scored_universe table.
# Writes use get_db() (which commits); reads use a non-committing session.
# ============================================================

SCORED_UNIVERSE_MARKER = "scored_universe"


def _as_float(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _as_int(v):
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def _utcstamp() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d_%H%M%S")


def persist_scored_universe(ranked) -> None:
    """Persist the Sunday-scan ranked DataFrame so monday_stock_buys can read it
    back. Lossless JSON (orient='split') into a BotLog row. Best-effort.

    NOTE: orient='split' round-trips column order + values faithfully for the
    scorer's numeric/string columns. If the frame ever carries datetime columns,
    confirm dtype on read (epoch-ms vs datetime)."""
    if ranked is None:
        logger.warning("persist_scored_universe: nothing to persist (ranked is None)")
        return
    try:
        payload = ranked.to_json(orient="split")
    except Exception as exc:
        logger.exception("persist_scored_universe: serialize failed: {}", exc)
        return
    try:
        with get_db() as db:
            db.add(BotLog(level="DATA", module=SCORED_UNIVERSE_MARKER, message=payload))
        logger.info("persist_scored_universe: persisted {} rows", len(ranked))
    except Exception as exc:
        logger.exception("persist_scored_universe: write failed: {}", exc)


def get_latest_scored_universe():
    """Return the most recent scored universe as a DataFrame, or None if none
    has been persisted. Reads the newest BotLog row written by
    persist_scored_universe and deserializes it."""
    db = SessionLocal()
    try:
        stmt = (
            select(BotLog)
            .where(BotLog.module == SCORED_UNIVERSE_MARKER)
            .order_by(BotLog.created_at.desc(), BotLog.id.desc())
            .limit(1)
        )
        row = db.scalars(stmt).first()
        payload = row.message if row is not None else None
    finally:
        db.close()
    if not payload:
        return None
    try:
        from io import StringIO
        import pandas as pd
        return pd.read_json(StringIO(payload), orient="split")
    except Exception as exc:
        logger.exception("get_latest_scored_universe: parse failed: {}", exc)
        return None


def persist_model_performance(metrics: dict) -> None:
    """Write a ModelPerformance row from retrain metrics. Tolerant: maps known
    keys, ignores the rest. ModelPerformance has no free-text column, so metrics
    without a column (e.g. 'auc') are not stored. Best-effort; the caller wraps
    this, but we also guard here."""
    m = dict(metrics or {})
    version = (str(m.get("version", "")) or _utcstamp())
    try:
        row = ModelPerformance(
            model_name=str(m.get("model_name", "weekly_stock")),
            asset_type=AssetType.STOCK,
            version=version,
            train_accuracy=_as_float(m.get("train_accuracy")),
            val_accuracy=_as_float(m.get("val_accuracy", m.get("accuracy"))),
            live_accuracy=_as_float(m.get("live_accuracy")),
            feature_count=_as_int(m.get("feature_count")),
            training_rows=_as_int(m.get("training_rows", m.get("n_train"))),
            is_active=bool(m.get("is_active", False)),
        )
        with get_db() as db:
            db.add(row)
        # log the local, NOT row.version: after commit the instance is detached
        # and expired (expire_on_commit=True), so attribute access would raise.
        logger.info("persist_model_performance: wrote row version={}", version)
    except Exception as exc:
        logger.exception("persist_model_performance: write failed: {}", exc)
