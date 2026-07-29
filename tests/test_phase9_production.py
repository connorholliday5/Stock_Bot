"""tests/test_phase9_production.py - production layer:
account snapshot (broker capital), Alpaca crypto executor routing, and the
web API (status / account / data / equity curve / controls / auth).

Fully offline: broker clients are faked, the web app runs on the real schema
bound to in-memory SQLite via FastAPI's TestClient.
"""

from __future__ import annotations

import os

for _k, _v in {
    "ALPACA_API_KEY": "test", "ALPACA_SECRET_KEY": "test",
    "STARTING_CAPITAL": "10000",
}.items():
    os.environ.setdefault(_k, _v)

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import database.db as dbmod
from database.db import (
    AssetType, Base, OrderSide, OrderStatus, Position, Trade,
)

UTC = timezone.utc


# ---- bind the real schema to in-memory SQLite -------------------------------

@pytest.fixture()
def mem_db(monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    monkeypatch.setattr(dbmod, "engine", engine)
    monkeypatch.setattr(dbmod, "SessionLocal", Session)
    yield Session


def _seed(Session):
    now = datetime.now(UTC)
    s = Session()
    s.add(Trade(asset_type=AssetType.STOCK, symbol="AAPL", side=OrderSide.SELL,
                quantity=1.0, entry_price=100.0, exit_price=110.0,
                status=OrderStatus.FILLED, pnl=10.0, pnl_pct=0.1,
                opened_at=now - timedelta(days=5), closed_at=now - timedelta(days=1)))
    s.add(Trade(asset_type=AssetType.CRYPTO, symbol="BTC/USD", side=OrderSide.SELL,
                quantity=0.001, entry_price=60000.0, exit_price=58000.0,
                status=OrderStatus.FILLED, pnl=-2.0, pnl_pct=-0.033,
                opened_at=now - timedelta(days=3), closed_at=now - timedelta(hours=5)))
    s.add(Position(asset_type=AssetType.CRYPTO, symbol="BTC/USD", quantity=0.001,
                   entry_price=61000.0, current_price=62000.0, stop_loss=58000.0,
                   take_profit=67000.0, opened_at=now - timedelta(hours=8)))
    s.commit()
    s.close()


# =============================================================================
# execution/account.py
# =============================================================================

class FakeTradingClient:
    def __init__(self, equity=229.81, cash=120.55, fail=False):
        self._equity, self._cash, self._fail = equity, cash, fail
        self.calls = 0

    def get_account(self):
        self.calls += 1
        if self._fail:
            raise RuntimeError("api down")
        import types
        return types.SimpleNamespace(equity=self._equity, cash=self._cash,
                                     buying_power=self._cash, currency="USD")


def test_snapshot_reads_broker_equity():
    from execution.account import AlpacaAccount
    acct = AlpacaAccount(client=FakeTradingClient())
    snap = acct.snapshot()
    assert snap.source == "broker"
    assert snap.equity == pytest.approx(229.81)   # the money in the account
    assert snap.cash == pytest.approx(120.55)


def test_snapshot_is_ttl_cached():
    from execution.account import AlpacaAccount
    client = FakeTradingClient()
    acct = AlpacaAccount(client=client, ttl_seconds=300)
    acct.snapshot(); acct.snapshot(); acct.snapshot()
    assert client.calls == 1                       # one API hit, many reads
    acct.snapshot(force=True)
    assert client.calls == 2


def test_snapshot_none_when_unconfigured():
    from execution.account import AlpacaAccount
    acct = AlpacaAccount(api_key="", secret_key="")
    assert acct.snapshot() is None


def test_get_account_snapshot_falls_back_to_static(monkeypatch):
    import execution.account as acc
    monkeypatch.setattr(acc.settings, "capital_source", "broker")
    monkeypatch.setattr(acc, "broker_account",
                        lambda: acc.AlpacaAccount(client=FakeTradingClient(fail=True)))

    class RM:
        def current_nav(self):
            return 1234.5

    snap = acc.get_account_snapshot(risk_manager=RM())
    assert snap.source == "static"
    assert snap.equity == pytest.approx(1234.5)


def test_get_account_snapshot_static_mode(monkeypatch):
    import execution.account as acc
    monkeypatch.setattr(acc.settings, "capital_source", "static")
    snap = acc.get_account_snapshot()
    assert snap.source == "static"


# =============================================================================
# execution/alpaca_crypto.py
# =============================================================================

def test_symbol_mapping():
    from execution.alpaca_crypto import to_alpaca_symbol
    assert to_alpaca_symbol("BTC/USDT") == "BTC/USD"
    assert to_alpaca_symbol("ETH/BUSD") == "ETH/USD"
    assert to_alpaca_symbol("SOL/USD") == "SOL/USD"
    assert to_alpaca_symbol("BTCUSD") == "BTCUSD"


def test_alpaca_crypto_paper_fill_and_db_rows(mem_db):
    from execution.alpaca_crypto import AlpacaCryptoExecutor
    ex = AlpacaCryptoExecutor(paper=True, paper_cash=1000.0)
    r = ex.open_long("BTC/USD", units=0.002, entry_price=60000.0,
                     stop_loss=57000.0, take_profit=66000.0)
    assert r["status"] == "filled"
    s = mem_db()
    pos = s.query(Position).filter_by(symbol="BTC/USD").one()
    assert pos.asset_type == AssetType.CRYPTO
    r2 = ex.close_long("BTC/USD", exit_price=63000.0, reason="take_profit")
    assert r2["status"] == "filled"
    assert r2["pnl"] > 0
    s.close()


def test_alpaca_crypto_live_routes_gtc(monkeypatch, mem_db):
    """Live order path sends the mapped symbol with GTC (crypto is 24/7)."""
    from execution.alpaca_crypto import AlpacaCryptoExecutor

    sent = {}

    class FakeOrder:
        filled_avg_price = 60100.0
        filled_qty = 0.002

    class FakeClient:
        def submit_order(self, req):
            sent["symbol"] = req.symbol
            sent["tif"] = str(req.time_in_force)
            return FakeOrder()

    ex = AlpacaCryptoExecutor(paper=False, client=FakeClient())
    r = ex.open_long("BTC/USDT", units=0.002, entry_price=60000.0,
                     stop_loss=57000.0, take_profit=66000.0)
    assert r["status"] == "filled"
    assert sent["symbol"] == "BTC/USD"
    assert "GTC" in sent["tif"].upper()


def test_factory_defaults_to_alpaca(monkeypatch):
    import execution.alpaca_crypto as ac
    monkeypatch.setattr(ac.settings, "crypto_exchange", "alpaca")
    monkeypatch.setattr(ac.settings, "environment", "paper")
    ex = ac.crypto_executor_from_settings()
    assert isinstance(ex, ac.AlpacaCryptoExecutor)
    assert ex.paper is True                        # sim mode by default


# =============================================================================
# webapp
# =============================================================================

@pytest.fixture()
def client(mem_db, monkeypatch):
    from fastapi.testclient import TestClient
    from execution.account import AccountSnapshot
    import webapp.api as api

    _seed(mem_db)
    monkeypatch.setattr(
        api, "get_account_snapshot",
        lambda *a, **k: AccountSnapshot(equity=229.81, cash=120.55,
                                        buying_power=120.55, source="broker"),
    )
    app = api.create_app(scheduler=None)
    return TestClient(app)


def test_dashboard_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Stock Bot" in r.text
    assert "Equity curve" in r.text


def test_status_endpoint(client):
    d = client.get("/api/status").json()
    assert d["paused"] in (True, False)
    assert "execution_mode" in d and "jobs" in d


def test_account_endpoint_uses_broker_money(client):
    d = client.get("/api/account").json()
    assert d["equity"] == pytest.approx(229.81)
    assert d["source"] == "broker"
    assert d["open_crypto_positions"] == 1


def test_positions_trades_signals(client):
    pos = client.get("/api/positions").json()
    assert [p["symbol"] for p in pos] == ["BTC/USD"]
    trades = client.get("/api/trades").json()
    assert len(trades) == 2
    sigs = client.get("/api/signals").json()
    assert sigs == []


def test_equity_curve_anchors_to_account(client):
    d = client.get("/api/equity-curve").json()
    pts = d["points"]
    # last point == current account equity; walk starts at equity - realized
    assert pts[-1]["equity"] == pytest.approx(229.81)
    assert pts[0]["equity"] == pytest.approx(229.81 - 8.0)   # realized = +10 - 2
    assert len(pts) == 3


def test_performance_buckets(client):
    d = client.get("/api/performance").json()
    assert d["all"]["trades"] == 2
    assert d["all"]["net_pnl"] == pytest.approx(8.0)
    assert d["stock"]["wins"] == 1
    assert d["crypto"]["losses"] == 1


def test_pause_resume_roundtrip(client):
    from runtime import bot_state
    bot_state.resume()
    assert client.post("/api/control/pause").json()["paused"] is True
    assert bot_state.paused is True
    assert client.post("/api/control/resume").json()["paused"] is False
    assert bot_state.paused is False


def test_control_auth_enforced(mem_db, monkeypatch):
    from fastapi.testclient import TestClient
    import webapp.api as api
    monkeypatch.setattr(api.settings, "web_auth_token", "s3cret")
    app = api.create_app(scheduler=None)
    c = TestClient(app)
    try:
        assert c.post("/api/control/pause").status_code == 401
        ok = c.post("/api/control/pause", headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200
        # reads stay open
        assert c.get("/api/status").status_code == 200
    finally:
        from runtime import bot_state
        bot_state.resume()


def test_unknown_job_404(client):
    assert client.post("/api/jobs/not_a_job/run").status_code == 404


def test_zero_cash_means_zero_budget():
    """available_cash=0 must gate the trade, never disable the cash ceiling.
    (Observed live: after a batch consumed the cash, remaining hit exactly 0
    and follow-on entries were sized FULL - broker rejected them all.)"""
    from risk.manager import size_position, stock_params
    ps = size_position(stock_params(), equity=10_000, available_cash=0.0,
                       entry_price=100.0, stop_price=95.0)
    assert not ps.tradable
    assert ps.reason == "no_cash"


class _PendingThenFilledClient:
    """Mimics Alpaca: submit returns 'accepted' filled_qty=0; polling the
    order returns the real fill."""

    def __init__(self, fill_qty=0.38, fill_price=228.59, fills_after=1):
        import types
        self._pending = types.SimpleNamespace(
            id="ord-1", status="accepted", filled_qty="0", filled_avg_price=None)
        self._filled = types.SimpleNamespace(
            id="ord-1", status="filled",
            filled_qty=str(fill_qty), filled_avg_price=str(fill_price))
        self._fills_after = fills_after
        self.polls = 0

    def submit_order(self, req):
        return self._pending

    def get_order_by_id(self, order_id):
        self.polls += 1
        return self._filled if self.polls >= self._fills_after else self._pending


def test_order_fill_is_polled_not_trusted(mem_db, monkeypatch):
    """The qty-0 phantom-position bug: the executor must poll until the order
    fills instead of recording the instant 'accepted' snapshot."""
    import execution.alpaca as ea
    monkeypatch.setattr(ea, "FILL_POLL_INTERVAL_S", 0.0)
    ex = ea.StockExecutor(paper=False, client=_PendingThenFilledClient())
    ex.min_position_usd = 10.0
    r = ex.open_long("MS", units=0.38, entry_price=228.24,
                     stop_loss=216.86, take_profit=251.0)
    assert r["status"] == "filled"
    assert r["units"] == pytest.approx(0.38)          # real fill qty, not 0
    assert r["price"] == pytest.approx(228.59)        # real fill price
    s = mem_db()
    pos = s.query(Position).filter_by(symbol="MS").one()
    assert pos.quantity == pytest.approx(0.38)
    s.close()


def test_unfilled_order_falls_back_to_submitted_qty(mem_db, monkeypatch):
    """After-hours orders queue unfilled; record submitted qty at ref price
    rather than a qty-0 phantom."""
    import execution.alpaca as ea
    monkeypatch.setattr(ea, "FILL_POLL_INTERVAL_S", 0.0)
    monkeypatch.setattr(ea, "FILL_POLL_ATTEMPTS", 2)
    client = _PendingThenFilledClient(fills_after=99)   # never fills in window
    ex = ea.StockExecutor(paper=False, client=client)
    ex.min_position_usd = 10.0
    r = ex.open_long("MS", units=0.38, entry_price=228.24,
                     stop_loss=216.86, take_profit=251.0)
    assert r["units"] == pytest.approx(0.38)
    assert r["price"] == pytest.approx(228.24)


def test_naive_timestamps_serialized_as_utc(client):
    """SQLite hands back naive datetimes; the API must stamp them UTC so the
    browser converts to the viewer's local time instead of misreading them."""
    trades = client.get("/api/trades").json()
    assert trades, "seeded trades expected"
    for t in trades:
        assert t["opened_at"].endswith("+00:00") or t["opened_at"].endswith("Z")


# =============================================================================
# database helpers
# =============================================================================

def test_update_position_marks(mem_db):
    _seed(mem_db)
    n = dbmod.update_position_marks({"BTC/USD": 65000.0, "UNKNOWN": 1.0})
    assert n == 1
    s = mem_db()
    pos = s.query(Position).filter_by(symbol="BTC/USD").one()
    assert pos.current_price == 65000.0
    assert pos.unrealized_pnl == pytest.approx((65000.0 - 61000.0) * 0.001)
    s.close()


def test_recent_trades_ordering(mem_db):
    _seed(mem_db)
    rows = dbmod.get_recent_trades(limit=10)
    assert len(rows) == 2


# =============================================================================
# adaptive minimum position floor
# =============================================================================

def test_effective_min_position_scales_with_equity():
    from risk.manager import effective_min_position
    assert effective_min_position(10_000, 50.0) == 50.0     # big account: unchanged
    assert effective_min_position(229.81, 50.0) == pytest.approx(22.981)  # 10% of equity
    assert effective_min_position(50.0, 50.0) == 10.0       # hard $10 floor
    assert effective_min_position(0, 50.0) == 50.0          # unknown equity: configured


def test_configured_position_count_is_actually_achievable():
    """MAX_STOCK_POSITIONS was fiction: 2% risk / 5% stop = a 40% position,
    so cash died after ~2.5 and the bot ran a concentrated 2-3 name book
    while reporting a max of 8. The notional cap makes the config real."""
    from risk.manager import size_position, stock_params
    params = stock_params(max_positions=8)
    equity = cash = 10_000.0
    filled = 0
    for _ in range(12):
        ps = size_position(params, equity=equity, available_cash=cash,
                           entry_price=100.0, stop_price=95.0)
        if not ps.tradable:
            break
        filled += 1
        cash -= ps.notional
        assert ps.notional <= equity * 0.126     # ~1/8 of equity
    assert filled == 8


def test_position_cap_honours_explicit_setting(monkeypatch):
    import risk.manager as rm
    monkeypatch.setattr(rm.settings, "stock_max_position_pct", 0.25, raising=False)
    assert rm.stock_position_cap(8) == pytest.approx(0.25)
    monkeypatch.setattr(rm.settings, "stock_max_position_pct", 0.0, raising=False)
    assert rm.stock_position_cap(4) == pytest.approx(0.25)   # auto = 1/4


def test_small_account_can_size_a_position():
    """A $230 account with a 5% stop must produce a tradable position - the
    old fixed $50 floor plus haircuts used to lock small accounts out."""
    from risk.manager import size_position, stock_params
    ps = size_position(
        stock_params(), equity=229.81, available_cash=229.81,
        entry_price=100.0, stop_price=95.0,
    )
    assert ps.tradable
    assert ps.notional >= 22.9


def test_executor_floor_is_instance_configurable(mem_db):
    from execution.alpaca_crypto import AlpacaCryptoExecutor
    ex = AlpacaCryptoExecutor(paper=True, paper_cash=230.0)
    ex.min_position_usd = 23.0
    r = ex.open_long("BTC/USD", units=0.0005, entry_price=60000.0,   # $30 notional
                     stop_loss=57000.0, take_profit=66000.0)
    assert r["status"] == "filled"


# =============================================================================
# stock universe
# =============================================================================

def test_default_stock_universe_is_full_sp500(monkeypatch):
    import data.alpaca_data as ad
    monkeypatch.setattr(ad.settings, "stock_universe", "", raising=False)
    universe = ad.default_stock_universe()
    assert len(universe) > 450
    assert "AAPL" in universe and "BRK.B" in universe


def test_stock_universe_env_override(monkeypatch):
    import data.alpaca_data as ad
    monkeypatch.setattr(ad.settings, "stock_universe", "aapl, msft ,NVDA", raising=False)
    assert ad.default_stock_universe() == ["AAPL", "MSFT", "NVDA"]


# =============================================================================
# pause persistence
# =============================================================================

def test_pause_state_survives_restart(tmp_path, monkeypatch):
    import runtime.state as rs
    monkeypatch.setattr(rs, "_state_file", lambda: tmp_path / "bot_state.json")
    first = rs.BotState()
    assert first.paused is False
    first.pause()
    reborn = rs.BotState()           # simulates a process restart
    assert reborn.paused is True
    reborn.resume()
    assert rs.BotState().paused is False
