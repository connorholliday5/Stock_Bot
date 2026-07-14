"""
execution/account.py
Production account layer - the live Alpaca account is the source of truth
for capital ("use the money in my account as the starting amount").

Responsibilities:
  - AlpacaAccount: thin, TTL-cached wrapper over TradingClient.get_account().
    The cache (default 60s) keeps every consumer (scheduler jobs, web UI,
    heartbeat) far below Alpaca's ~200 req/min rate limit no matter how often
    they poll.
  - get_account_snapshot(): the single chokepoint the scheduler and web UI
    call. Uses the broker when CAPITAL_SOURCE=broker and keys are configured;
    otherwise (or on any broker failure) falls back to the DB-reconstructed
    NAV from RiskManager, so the bot keeps functioning offline and in tests.

The alpaca-py SDK is imported lazily so this module (and everything that
imports it) loads with no SDK installed - tests run fully offline.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from config import settings

UTC = timezone.utc

DEFAULT_TTL_SECONDS = 60.0


@dataclass
class AccountSnapshot:
    """Normalized view of tradable capital, wherever it came from."""
    equity: float
    cash: float
    buying_power: float = 0.0
    currency: str = "USD"
    source: str = "broker"          # "broker" | "static"
    is_paper: bool = True
    fetched_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def as_dict(self) -> dict:
        return {
            "equity": round(self.equity, 2),
            "cash": round(self.cash, 2),
            "buying_power": round(self.buying_power, 2),
            "currency": self.currency,
            "source": self.source,
            "is_paper": self.is_paper,
            "fetched_at": self.fetched_at.isoformat(),
        }


class AlpacaAccount:
    """TTL-cached reader for the Alpaca trading account."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: Optional[bool] = None,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        client=None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.alpaca_api_key
        self.secret_key = secret_key if secret_key is not None else settings.alpaca_secret_key
        self.paper = settings.is_paper if paper is None else paper
        self.ttl_seconds = ttl_seconds
        self._client = client            # injectable for tests
        self._lock = threading.Lock()
        self._cached: Optional[AccountSnapshot] = None
        self._cached_at: float = 0.0

    @property
    def configured(self) -> bool:
        creds_ok = bool(self.api_key and self.secret_key
                        and not self.api_key.startswith("your_"))
        return creds_ok or self._client is not None

    def _get_client(self):
        if self._client is None:
            from alpaca.trading.client import TradingClient  # lazy: optional dep
            self._client = TradingClient(self.api_key, self.secret_key, paper=self.paper)
        return self._client

    def snapshot(self, force: bool = False) -> Optional[AccountSnapshot]:
        """Cached account snapshot; None when unconfigured or the broker errors."""
        if not self.configured:
            return None
        with self._lock:
            fresh = (time.monotonic() - self._cached_at) < self.ttl_seconds
            if self._cached is not None and fresh and not force:
                return self._cached
            try:
                account = self._get_client().get_account()
            except Exception as exc:
                logger.warning("AlpacaAccount: get_account failed ({}); using stale/fallback", exc)
                return self._cached  # possibly stale, possibly None - caller falls back
            snap = AccountSnapshot(
                equity=float(getattr(account, "equity", 0.0) or 0.0),
                cash=float(getattr(account, "cash", 0.0) or 0.0),
                buying_power=float(getattr(account, "buying_power", 0.0) or 0.0),
                currency=str(getattr(account, "currency", "USD") or "USD"),
                source="broker",
                is_paper=self.paper,
            )
            self._cached = snap
            self._cached_at = time.monotonic()
            return snap


# Module-level singleton so every consumer shares one cache (one rate budget).
_account: Optional[AlpacaAccount] = None
_account_lock = threading.Lock()


def broker_account() -> AlpacaAccount:
    global _account
    with _account_lock:
        if _account is None:
            _account = AlpacaAccount()
        return _account


def _static_snapshot(risk_manager=None) -> AccountSnapshot:
    """DB-reconstructed capital: STARTING_CAPITAL + realized + marked P&L."""
    equity = float(settings.starting_capital)
    if risk_manager is not None:
        try:
            equity = float(risk_manager.current_nav())
        except Exception as exc:
            logger.warning("static snapshot: current_nav failed ({}); using starting_capital", exc)
    return AccountSnapshot(
        equity=equity,
        cash=equity,          # scheduler subtracts open-position notional itself
        buying_power=equity,
        source="static",
        is_paper=settings.is_paper,
    )


def get_account_snapshot(risk_manager=None, force: bool = False) -> AccountSnapshot:
    """Capital chokepoint. Broker first (when enabled), DB/static fallback."""
    if settings.use_broker_capital:
        snap = broker_account().snapshot(force=force)
        if snap is not None:
            return snap
        logger.warning("get_account_snapshot: broker unavailable; falling back to static capital")
    return _static_snapshot(risk_manager)
