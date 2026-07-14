"""
runtime/state.py
In-process bot state shared between the scheduler (writer) and the web UI
(reader + pause/resume control). Thread-safe; APScheduler executes jobs on
worker threads while uvicorn serves requests on its own.

Pause semantics (deliberate): pausing blocks NEW ENTRIES only. Risk-reducing
jobs - the mid-week stop-loss monitor and the Friday scheduled exit - keep
running, so pausing can never strand an open position without protection.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

UTC = timezone.utc


@dataclass
class JobRun:
    job_id: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    ok: Optional[bool] = None
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "ok": self.ok,
            "detail": self.detail,
        }


class BotState:
    def __init__(self, history_size: int = 200) -> None:
        self._lock = threading.Lock()
        self.started_at = datetime.now(UTC)
        self._paused = False
        self._history: deque[JobRun] = deque(maxlen=history_size)
        self._last: dict[str, JobRun] = {}

    # -- pause / resume -----------------------------------------------------

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    # -- job accounting -------------------------------------------------------

    def job_started(self, job_id: str) -> JobRun:
        run = JobRun(job_id=job_id, started_at=datetime.now(UTC))
        with self._lock:
            self._history.append(run)
            self._last[job_id] = run
        return run

    def job_finished(self, run: JobRun, ok: bool, detail: str = "") -> None:
        with self._lock:
            run.finished_at = datetime.now(UTC)
            run.ok = ok
            run.detail = detail[:500]

    # -- snapshots ------------------------------------------------------------

    def last_runs(self) -> dict[str, dict]:
        with self._lock:
            return {job_id: run.as_dict() for job_id, run in self._last.items()}

    def history(self, limit: int = 50) -> list[dict]:
        with self._lock:
            runs = list(self._history)[-limit:]
        return [r.as_dict() for r in reversed(runs)]

    def snapshot(self) -> dict:
        return {
            "started_at": self.started_at.isoformat(),
            "paused": self.paused,
            "last_runs": self.last_runs(),
        }


bot_state = BotState()
