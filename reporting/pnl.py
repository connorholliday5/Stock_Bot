"""
reporting/pnl.py
Phase 8 - Monitoring and reporting.

Computation functions take plain data and return structured results. An ADAPTER
section pulls inputs from the DB through getattr-guarded accessors that degrade
to empty on any failure (same pattern as scheduler Phase 7: never crash the
scheduler thread, no schema changes, single chokepoint to wire real accessors).

Assumed db.py surface (rename in the ADAPTER if yours differs):
  db.get_closed_trades(since)        -> iterable of Trade-like rows
  db.get_open_positions(asset_type)  -> iterable of Position-like rows
  db.get_model_performance()         -> iterable of ModelPerformance-like rows

All row fields are read tolerantly (multiple candidate attribute names) so a
column rename cannot crash reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

from loguru import logger

from database import db

MARKET_TZ = ZoneInfo("America/New_York")
RECENT_TRADES_LIMIT = 15
MODEL_TREND_LIMIT = 8

_EPOCH = datetime.min.replace(tzinfo=MARKET_TZ)


# ----- tolerant access ---------------------------------------------------

def _attr(obj: Any, *names: str, default=None):
    if isinstance(obj, dict):
        for n in names:
            if n in obj and obj[n] is not None:
                return obj[n]
        return default
    for n in names:
        val = getattr(obj, n, None)
        if val is not None:
            return val
    return default


def _f(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _as_et(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=MARKET_TZ)


# ----- time windows ------------------------------------------------------

def current_week_window(now: Optional[datetime] = None) -> tuple[datetime, datetime]:
    """Monday 00:00 ET .. Sunday end-of-day ET for the week containing now."""
    if now is None:
        now = datetime.now(MARKET_TZ)
    now = _as_et(now).astimezone(MARKET_TZ)
    monday = (now - timedelta(days=now.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0)
    end = monday + timedelta(days=7) - timedelta(microseconds=1)
    return monday, end


def _trade_time(t) -> Optional[datetime]:
    raw = _attr(t, "closed_at", "exit_time", "exit_at", "updated_at", "timestamp")
    if isinstance(raw, datetime):
        return _as_et(raw)
    return None


def filter_week(trades, start: datetime, end: datetime) -> list:
    """Keep dated trades inside [start, end]. Undated trades pass through
    (caller is assumed to have scoped them)."""
    out = []
    for t in trades or []:
        ts = _trade_time(t)
        if ts is None:
            out.append(t)
            continue
        if start <= ts.astimezone(MARKET_TZ) <= end:
            out.append(t)
    return out


# ----- aggregation -------------------------------------------------------

def summarize_trades(trades) -> dict:
    trades = list(trades or [])
    wins = losses = 0
    gross_win = gross_loss = 0.0
    by_asset: dict[str, dict] = {}
    for t in trades:
        pnl = _f(_attr(t, "pnl", "realized_pnl", "net_pnl", default=0.0))
        asset = str(_attr(t, "asset_type", "asset", default="stock")).lower()
        a = by_asset.setdefault(asset, {"net_pnl": 0.0, "trades": 0, "wins": 0})
        a["net_pnl"] += pnl
        a["trades"] += 1
        if pnl > 0:
            wins += 1
            gross_win += pnl
            a["wins"] += 1
        elif pnl < 0:
            losses += 1
            gross_loss += pnl
    total = len(trades)
    return {
        "net_pnl": gross_win + gross_loss,
        "gross_win": gross_win,
        "gross_loss": gross_loss,
        "win_rate": (wins / total * 100.0) if total else 0.0,
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "by_asset": by_asset,
    }


def normalize_positions(positions, asset_type: Optional[str] = None) -> list[dict]:
    out = []
    for p in positions or []:
        qty = _f(_attr(p, "qty", "quantity", "shares", default=0.0))
        entry = _f(_attr(p, "entry_price", "avg_entry", "entry", default=0.0))
        mark = _f(_attr(p, "current_price", "mark", "last_price"), entry)
        upnl = (mark - entry) * qty if (entry and qty) else 0.0
        out.append({
            "symbol": str(_attr(p, "symbol", "ticker", default="?")),
            "asset_type": str(_attr(p, "asset_type", "asset", default=asset_type or "")).lower(),
            "qty": qty,
            "entry": entry,
            "mark": mark,
            "upnl": upnl,
        })
    return out


def recent_trades(trades, limit: int = RECENT_TRADES_LIMIT) -> list[dict]:
    rows = []
    for t in trades or []:
        rows.append({
            "symbol": str(_attr(t, "symbol", "ticker", default="?")),
            "asset_type": str(_attr(t, "asset_type", "asset", default="")).lower(),
            "pnl": _f(_attr(t, "pnl", "realized_pnl", "net_pnl", default=0.0)),
            "pnl_pct": _attr(t, "pnl_pct", "return_pct"),
            "reason": str(_attr(t, "reason", "exit_reason", default="")),
            "closed_at": _trade_time(t),
        })
    rows.sort(key=lambda r: (r["closed_at"] is not None, r["closed_at"] or _EPOCH),
              reverse=True)
    return rows[:limit]


def model_trend(rows, limit: int = MODEL_TREND_LIMIT) -> list[dict]:
    out = []
    for r in rows or []:
        out.append({
            "when": _attr(r, "created_at", "trained_at", "timestamp", "date"),
            "auc": _attr(r, "auc", "roc_auc"),
            "accuracy": _attr(r, "accuracy", "acc"),
            "model": str(_attr(r, "model_name", "name", default="weekly_stock")),
        })
    out.sort(
        key=lambda x: (isinstance(x["when"], datetime),
                       x["when"] if isinstance(x["when"], datetime) else _EPOCH),
        reverse=True,
    )
    return out[:limit]


# ============================================================
# ADAPTER LAYER - DB reads (getattr-guarded, degrade to empty)
# ============================================================

def _call(name: str, *args, default):
    getter = getattr(db, name, None)
    if getter is None:
        logger.info("reporting: db.{} not wired; using default.", name)
        return default
    try:
        return getter(*args)
    except TypeError:
        try:
            return getter()
        except Exception as exc:
            logger.warning("reporting: db.{}() failed: {}", name, exc)
            return default
    except Exception as exc:
        logger.warning("reporting: db.{} failed: {}", name, exc)
        return default


def load_closed_trades(since) -> list:
    return list(_call("get_closed_trades", since, default=[]) or [])


def load_open_positions(asset_type: str) -> list:
    return list(_call("get_open_positions", asset_type, default=[]) or [])


def load_model_rows() -> list:
    return list(_call("get_model_performance", default=[]) or [])


# ----- top-level report builder ------------------------------------------

@dataclass
class WeeklyReport:
    week_index: int
    week_start: datetime
    week_end: datetime
    generated_at: datetime
    environment: str
    nav: float
    net_pnl: float
    gross_win: float
    gross_loss: float
    win_rate: float
    total_trades: int
    wins: int
    losses: int
    by_asset: dict
    open_positions: list
    recent_trades: list
    model_trend: list


def build_weekly_report(
    now: Optional[datetime] = None,
    nav: float = 0.0,
    stock_positions=None,
    crypto_positions=None,
    trades=None,
    model_rows=None,
    environment: str = "paper",
) -> WeeklyReport:
    start, end = current_week_window(now)
    gen = _as_et(now).astimezone(MARKET_TZ) if isinstance(now, datetime) else datetime.now(MARKET_TZ)

    if trades is None:
        trades = load_closed_trades(start)
    trades = filter_week(trades, start, end)

    if model_rows is None:
        model_rows = load_model_rows()
    if stock_positions is None:
        stock_positions = load_open_positions("stock")
    if crypto_positions is None:
        crypto_positions = load_open_positions("crypto")

    summary = summarize_trades(trades)
    positions = (normalize_positions(stock_positions, "stock")
                 + normalize_positions(crypto_positions, "crypto"))

    return WeeklyReport(
        week_index=start.isocalendar().week,
        week_start=start,
        week_end=end,
        generated_at=gen,
        environment=str(environment),
        nav=_f(nav),
        net_pnl=summary["net_pnl"],
        gross_win=summary["gross_win"],
        gross_loss=summary["gross_loss"],
        win_rate=summary["win_rate"],
        total_trades=summary["total_trades"],
        wins=summary["wins"],
        losses=summary["losses"],
        by_asset=summary["by_asset"],
        open_positions=positions,
        recent_trades=recent_trades(trades),
        model_trend=model_trend(model_rows),
    )
