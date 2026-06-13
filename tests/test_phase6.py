
"""tests/test_phase6.py - scheduler wiring verification.

Covers: job registration + triggers, DB-reconstruct account state, RiskManager
halt gating on both trading sides, the Monday/Friday/crypto dispatch contracts,
mid-week stop-loss closing logic, and clean no-op behavior of the ML/report stubs.

Every external dependency is monkeypatched, so this suite runs with no DB, no
broker, and no network.
"""

import types

import pytest

import scheduler as S


# --------------------------- fakes ---------------------------

class FakeRM:
    def __init__(self, nav=1000.0, halted=False):
        self._nav = nav
        self._halted = halted

    @property
    def current_nav(self):
        return self._nav

    @property
    def is_halted(self):
        return self._halted


class FakePos:
    def __init__(self, symbol, qty, entry_price, current_price=None):
        self.symbol = symbol
        self.qty = qty
        self.entry_price = entry_price
        self.current_price = current_price


class FakeExecutor:
    def __init__(self):
        self.closed = []

    @classmethod
    def from_settings(cls):
        return cls()

    def close_long(self, symbol):
        self.closed.append(symbol)
        return {"symbol": symbol, "action": "close_long"}


# --------------------------- registration ---------------------------

def test_eight_jobs_registered():
    sched = S.build_scheduler()
    ids = sorted(j.id for j in sched.get_jobs())
    assert ids == sorted([
        "sunday_stock_scan", "sunday_ml_retrain", "monday_stock_buys",
        "midweek_stock_monitor", "friday_stock_sells", "crypto_cycle",
        "weekly_performance_report", "daily_heartbeat",
    ])


def _trigger_str(sched, job_id):
    job = next(j for j in sched.get_jobs() if j.id == job_id)
    return str(job.trigger)


def test_stock_triggers_match_strategy():
    sched = S.build_scheduler()
    buys = _trigger_str(sched, "monday_stock_buys")
    sells = _trigger_str(sched, "friday_stock_sells")
    scan = _trigger_str(sched, "sunday_stock_scan")
    assert "day_of_week='mon'" in buys and "hour='9'" in buys and "minute='45'" in buys
    assert "day_of_week='fri'" in sells and "hour='15'" in sells and "minute='45'" in sells
    assert "day_of_week='sun'" in scan and "hour='20'" in scan


def test_crypto_trigger_is_4h():
    sched = S.build_scheduler()
    assert "hour='*/4'" in _trigger_str(sched, "crypto_cycle")


# --------------------------- account state ---------------------------

def test_account_state_db_reconstruct(monkeypatch):
    rm = FakeRM(nav=1000.0)
    positions = [FakePos("AAPL", 2, 100.0, 110.0), FakePos("MSFT", 1, 50.0, None)]
    monkeypatch.setattr(S.db, "get_open_positions", lambda a: positions, raising=False)
    st = S.get_account_state(rm, "stock")
    # invested = 2*110 + 1*50 (falls back to entry when current is None) = 270
    assert st.equity == 1000.0
    assert st.cash == pytest.approx(730.0)
    assert len(st.open_positions) == 2


# --------------------------- halt gating ---------------------------

def test_monday_buys_blocked_when_halted(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(S, "build_risk_manager", lambda: FakeRM(halted=True))
    monkeypatch.setattr(S.stock_weekly, "run_stock_weekly_buys",
                        lambda **k: called.__setitem__("n", called["n"] + 1))
    S.monday_stock_buys()
    assert called["n"] == 0


def test_crypto_blocked_when_halted(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(S, "build_risk_manager", lambda: FakeRM(halted=True))
    monkeypatch.setattr(S.crypto_24h, "run_crypto_24h_pipeline",
                        lambda **k: called.__setitem__("n", called["n"] + 1))
    S.crypto_cycle()
    assert called["n"] == 0


# --------------------------- dispatch contracts ---------------------------

def test_monday_buys_passes_risk_manager(monkeypatch):
    rm = FakeRM(nav=1000.0, halted=False)
    captured = {}
    scored = types.SimpleNamespace(columns=["symbol"], __len__=lambda s: 3)

    # SimpleNamespace can't be len()'d or indexed; use a tiny stand-in instead.
    class Scored:
        columns = ["symbol"]
        def __len__(self):
            return 3
        def __getitem__(self, k):
            return ["AAPL", "MSFT", "NVDA"]
    scored = Scored()

    monkeypatch.setattr(S, "build_risk_manager", lambda: rm)
    monkeypatch.setattr(S.db, "get_latest_scored_universe", lambda: scored, raising=False)
    monkeypatch.setattr(S.db, "get_open_positions", lambda a: [], raising=False)
    monkeypatch.setattr(S.StockExecutor, "from_settings", classmethod(lambda cls: FakeExecutor()))
    monkeypatch.setattr(S.stock_weekly, "run_stock_weekly_buys",
                        lambda **k: captured.update(k) or {"orders": 0})
    S.monday_stock_buys()
    assert captured["risk_manager"] is rm
    assert captured["equity"] == 1000.0
    assert captured["universe"] == ["AAPL", "MSFT", "NVDA"]


def test_monday_buys_skips_without_scan(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(S, "build_risk_manager", lambda: FakeRM(halted=False))
    monkeypatch.setattr(S.db, "get_latest_scored_universe", lambda: None, raising=False)
    monkeypatch.setattr(S.stock_weekly, "run_stock_weekly_buys",
                        lambda **k: called.__setitem__("n", called["n"] + 1))
    S.monday_stock_buys()
    assert called["n"] == 0


def test_friday_sells_dispatch(monkeypatch):
    positions = [FakePos("AAPL", 2, 100.0, 105.0)]
    captured = {}
    monkeypatch.setattr(S, "build_risk_manager", lambda: FakeRM())
    monkeypatch.setattr(S.db, "get_open_positions", lambda a: positions, raising=False)
    monkeypatch.setattr(S.StockExecutor, "from_settings", classmethod(lambda cls: FakeExecutor()))
    monkeypatch.setattr(S.stock_weekly, "run_stock_weekly_sells",
                        lambda **k: captured.update(k) or {"sold": 1})
    S.friday_stock_sells()
    assert captured["open_positions"] == positions


def test_crypto_dispatch_passes_btc_only(monkeypatch):
    rm = FakeRM(nav=1000.0, halted=False)
    captured = {}
    monkeypatch.setattr(S, "build_risk_manager", lambda: rm)
    monkeypatch.setattr(S.db, "get_open_positions", lambda a: [], raising=False)
    monkeypatch.setattr(S.crypto_24h, "run_crypto_24h_pipeline",
                        lambda **k: captured.update(k) or {"orders": 0})
    S.crypto_cycle()
    assert captured["universe"] == ["BTC/USDT"]
    assert captured["risk_manager"] is rm


# --------------------------- mid-week stop loss ---------------------------

def test_midweek_closes_only_stopped_out(monkeypatch):
    ex = FakeExecutor()
    # AAPL down 6% -> stop hit; MSFT down 2% -> hold; NVDA no mark -> skip
    positions = [
        FakePos("AAPL", 1, 100.0, 94.0),
        FakePos("MSFT", 1, 100.0, 98.0),
        FakePos("NVDA", 1, 100.0, None),
    ]
    monkeypatch.setattr(S.db, "get_open_positions", lambda a: positions, raising=False)
    monkeypatch.setattr(S.StockExecutor, "from_settings", classmethod(lambda cls: ex))
    S.midweek_stock_monitor()
    assert ex.closed == ["AAPL"]


# --------------------------- stubs ---------------------------

def test_stubs_run_clean():
    # Should not raise.
    S.sunday_ml_retrain()
    S.weekly_performance_report()
