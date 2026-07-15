"""
webapp/api.py
FastAPI application: JSON API + the single-page dashboard.

Read endpoints are open (intended for localhost / private LAN / behind a
reverse proxy). Control endpoints (pause / resume / run-job) additionally
require "Authorization: Bearer <WEB_AUTH_TOKEN>" whenever WEB_AUTH_TOKEN is
set - set it for anything reachable from the internet.

The app is built by create_app(scheduler); main.py passes the running
APScheduler instance so /api/status can report real next-run times and
/api/jobs/{id}/run can fire any job on demand (in a worker thread - jobs do
network + DB work and must never block the event loop).
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from config import settings
from database import db
from execution.account import get_account_snapshot
from runtime import bot_state

UTC = timezone.utc

STATIC_DIR = Path(__file__).parent / "static"

START_TIME = datetime.now(UTC)


def _require_control_auth(request: Request) -> None:
    token = settings.web_auth_token
    if not token:
        return
    header = request.headers.get("authorization", "")
    if header != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="invalid or missing bearer token")


def _iso(dt) -> Optional[str]:
    if dt is None:
        return None
    try:
        # SQLite returns naive datetimes even for timezone-aware columns; all
        # bot writes are UTC, so stamp naive values as UTC. Without the offset
        # the browser parses them as *local* time and shows times hours off.
        if getattr(dt, "tzinfo", None) is None and hasattr(dt, "replace"):
            dt = dt.replace(tzinfo=UTC)
        return dt.isoformat()
    except Exception:
        return str(dt)


def _position_row(p) -> dict:
    return {
        "symbol": getattr(p, "symbol", ""),
        "asset_type": str(getattr(getattr(p, "asset_type", None), "value", getattr(p, "asset_type", ""))),
        "quantity": getattr(p, "quantity", None),
        "entry_price": getattr(p, "entry_price", None),
        "current_price": getattr(p, "current_price", None),
        "stop_loss": getattr(p, "stop_loss", None),
        "take_profit": getattr(p, "take_profit", None),
        "unrealized_pnl": getattr(p, "unrealized_pnl", None),
        "unrealized_pnl_pct": getattr(p, "unrealized_pnl_pct", None),
        "opened_at": _iso(getattr(p, "opened_at", None)),
    }


def _trade_row(t) -> dict:
    return {
        "id": getattr(t, "id", None),
        "symbol": getattr(t, "symbol", ""),
        "asset_type": str(getattr(getattr(t, "asset_type", None), "value", getattr(t, "asset_type", ""))),
        "side": str(getattr(getattr(t, "side", None), "value", getattr(t, "side", ""))),
        "quantity": getattr(t, "quantity", None),
        "entry_price": getattr(t, "entry_price", None),
        "exit_price": getattr(t, "exit_price", None),
        "pnl": getattr(t, "pnl", None),
        "pnl_pct": getattr(t, "pnl_pct", None),
        "close_reason": getattr(t, "close_reason", None),
        "opened_at": _iso(getattr(t, "opened_at", None)),
        "closed_at": _iso(getattr(t, "closed_at", None)),
    }


def _signal_row(s) -> dict:
    return {
        "id": getattr(s, "id", None),
        "symbol": getattr(s, "symbol", ""),
        "asset_type": str(getattr(getattr(s, "asset_type", None), "value", getattr(s, "asset_type", ""))),
        "direction": str(getattr(getattr(s, "direction", None), "value", getattr(s, "direction", ""))),
        "composite_score": getattr(s, "composite_score", None),
        "acted_on": getattr(s, "acted_on", None),
        "created_at": _iso(getattr(s, "created_at", None)),
    }


def create_app(scheduler=None) -> FastAPI:
    app = FastAPI(title="Stock Bot", version="1.0.0", docs_url="/api/docs",
                  openapi_url="/api/openapi.json")
    app.state.scheduler = scheduler

    # ---------------- dashboard ----------------

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        index = STATIC_DIR / "index.html"
        if not index.exists():
            return "<h1>Stock Bot</h1><p>Dashboard asset missing; API at /api/docs</p>"
        return index.read_text(encoding="utf-8")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
               '<text y="13" font-size="13">📈</text></svg>')
        from fastapi.responses import Response
        return Response(content=svg, media_type="image/svg+xml")

    # ---------------- reads ----------------

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": db.health_check(), "time": datetime.now(UTC).isoformat()}

    @app.get("/api/status")
    def status() -> dict:
        sched = app.state.scheduler
        jobs = []
        if sched is not None:
            try:
                for job in sched.get_jobs():
                    jobs.append({
                        "id": job.id,
                        "next_run": _iso(getattr(job, "next_run_time", None)),
                    })
            except Exception:
                jobs = []
        halted = False
        try:
            from scheduler import build_risk_manager, _rm_halted
            halted = _rm_halted(build_risk_manager())
        except Exception:
            pass
        return {
            "environment": settings.environment,
            "execution_mode": settings.execution_mode,
            "capital_source": settings.capital_source,
            "crypto_exchange": settings.crypto_exchange,
            "paused": bot_state.paused,
            "halted": halted,
            "started_at": bot_state.started_at.isoformat(),
            "scheduler_running": bool(sched is not None and getattr(sched, "running", False)),
            "jobs": jobs,
            "last_runs": bot_state.last_runs(),
        }

    @app.get("/api/account")
    def account() -> dict:
        snap = get_account_snapshot()
        positions = db.get_open_positions()
        stock = [p for p in positions
                 if str(getattr(getattr(p, "asset_type", None), "value", "")) == "stock"]
        crypto = [p for p in positions
                  if str(getattr(getattr(p, "asset_type", None), "value", "")) == "crypto"]
        return {
            **snap.as_dict(),
            "open_stock_positions": len(stock),
            "open_crypto_positions": len(crypto),
        }

    @app.get("/api/positions")
    def positions() -> list[dict]:
        return [_position_row(p) for p in db.get_open_positions()]

    @app.get("/api/trades")
    def trades(limit: int = 100) -> list[dict]:
        limit = max(1, min(int(limit), 500))
        return [_trade_row(t) for t in db.get_recent_trades(limit=limit)]

    @app.get("/api/signals")
    def signals(limit: int = 100) -> list[dict]:
        limit = max(1, min(int(limit), 500))
        return [_signal_row(s) for s in db.get_recent_signals(limit=limit)]

    @app.get("/api/equity-curve")
    def equity_curve() -> dict:
        """Realized equity walk: starting capital + cumulative closed-trade
        P&L, one point per close. Anchored to the live account equity when the
        broker is the capital source."""
        closed = sorted(
            db.get_closed_trades(),
            key=lambda t: getattr(t, "closed_at", None) or datetime.min.replace(tzinfo=UTC),
        )
        realized = sum(float(getattr(t, "pnl", 0.0) or 0.0) for t in closed)
        snap = get_account_snapshot()
        # Work backwards so "now" always equals the account's current equity.
        base = snap.equity - realized
        points = [{"t": None, "equity": round(base, 2)}]
        running = base
        for t in closed:
            running += float(getattr(t, "pnl", 0.0) or 0.0)
            points.append({"t": _iso(getattr(t, "closed_at", None)), "equity": round(running, 2)})
        return {"source": snap.source, "current_equity": round(snap.equity, 2), "points": points}

    @app.get("/api/performance")
    def performance() -> dict:
        closed = db.get_closed_trades()
        def bucket(trades):
            pnls = [float(getattr(t, "pnl", 0.0) or 0.0) for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            return {
                "trades": len(pnls),
                "net_pnl": round(sum(pnls), 2),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": round(100.0 * len(wins) / len(pnls), 1) if pnls else 0.0,
                "avg_win": round(sum(wins) / len(wins), 2) if wins else 0.0,
                "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0.0,
            }
        stock = [t for t in closed
                 if str(getattr(getattr(t, "asset_type", None), "value", "")) == "stock"]
        crypto = [t for t in closed
                  if str(getattr(getattr(t, "asset_type", None), "value", "")) == "crypto"]
        return {"all": bucket(closed), "stock": bucket(stock), "crypto": bucket(crypto)}

    # ---------------- controls ----------------

    @app.post("/api/control/pause", dependencies=[Depends(_require_control_auth)])
    def pause() -> dict:
        bot_state.pause()
        return {"paused": True,
                "note": "New entries stopped. Stop-loss monitoring and scheduled exits keep running."}

    @app.post("/api/control/resume", dependencies=[Depends(_require_control_auth)])
    def resume() -> dict:
        bot_state.resume()
        return {"paused": False}

    @app.post("/api/jobs/{job_id}/run", dependencies=[Depends(_require_control_auth)])
    def run_job(job_id: str) -> JSONResponse:
        from scheduler import JOBS
        fn = JOBS.get(job_id)
        if fn is None:
            raise HTTPException(status_code=404, detail=f"unknown job '{job_id}'")
        thread = threading.Thread(target=fn, name=f"manual-{job_id}", daemon=True)
        thread.start()
        return JSONResponse({"started": job_id,
                             "note": "running in background; watch last_runs in /api/status"})

    @app.get("/api/job-history")
    def job_history(limit: int = 50) -> list[dict]:
        return bot_state.history(limit=max(1, min(int(limit), 200)))

    return app
