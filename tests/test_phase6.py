"""tests/test_phase6.py - scheduler wiring verification (production wiring).

Covers: job registration + triggers, broker/static account state, RiskManager
halt gating on both trading sides, the Monday/Friday/crypto dispatch contracts
against the REAL strategy signatures (universe dicts of DataFrames, position
dicts, priced exits), mid-week stop-loss closing, the web-UI pause gate, and
clean no-op behavior of the ML/report stubs.

Every external dependency is monkeypatched, so this suite runs with no DB, no
broker, and no network.
"""

import os

for _k, _v in {
    "ALPACA_API_KEY": "test", "ALPACA_SECRET_KEY": "test",
    "STARTING_CAPITAL": "10000",
}.items():
    os.environ.setdefault(_k, _v)

import pandas as pd
import pytest

import scheduler as S
from execution.account import AccountSnapshot
from runtime import bot_state


# --------------------------- fakes ---------------------------

class FakeRM:
    """Mirrors the real RiskManager surface: is_halted() and current_nav()
    are METHODS (the old scheduler treated them as attributes - that bug made
    every entry job think the bot was halted)."""

    def __init__(self, nav=1000.0, halted=False):
        self._nav = nav
        self._halted = halted

    def current_nav(self):
        return self._nav

    def is_halted(self):
        return self._halted


class FakePos:
    def __init__(self, symbol, quantity, entry_price, current_price=None, stop_loss=0.0):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.current_price = current_price
        self.stop_loss = stop_loss
        self.take_profit = 0.0
        self.week_number = None


class FakeExecutor:
    def __init__(self):
        self.closed = []
        self.opened = []
        self.paper_cash = 0.0

    @classmethod
    def from_settings(cls):
        return cls()

    def close_long(self, symbol, exit_price=0.0, units=None, reason=""):
        self.closed.append((symbol, exit_price, reason))
        return {"status": "filled", "symbol": symbol}

    def open_long(self, symbol, units, entry_price, stop_loss, take_profit, week_number=None):
        self.opened.append(symbol)
        return {"status": "filled", "symbol": symbol}


def _static_snap(equity):
    return AccountSnapshot(equity=equity, cash=equity, source="static")


def _df(close=100.0):
    return pd.DataFrame({"close": [close], "atr": [1.0]})


@pytest.fixture(autouse=True)
def _unpaused():
    bot_state.resume()
    yield
    bot_state.resume()


# --------------------------- registration ---------------------------

def test_nine_jobs_registered():
    sched = S.build_scheduler()
    ids = sorted(j.id for j in sched.get_jobs())
    assert ids == sorted([
        "sunday_stock_scan", "sunday_ml_retrain", "monday_stock_buys",
        "midweek_stock_monitor", "friday_stock_sells", "crypto_cycle",
        "crypto_stop_monitor", "weekly_performance_report", "daily_heartbeat",
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


def test_crypto_stop_monitor_is_15m():
    sched = S.build_scheduler()
    assert "minute='*/15'" in _trigger_str(sched, "crypto_stop_monitor")


def test_crypto_stop_monitor_closes_breached(monkeypatch):
    ex = FakeExecutor()
    positions = [
        FakePos("BTC/USD", 0.001, 61000.0, stop_loss=58000.0),   # breached at 57500
        FakePos("ETH/USD", 0.01, 3000.0, stop_loss=2800.0),      # safe at 2900
    ]
    monkeypatch.setattr(S.db, "get_open_positions", lambda a: positions, raising=False)
    monkeypatch.setattr(S.db, "update_position_marks", lambda p: 0, raising=False)
    monkeypatch.setattr(S, "_latest_crypto_prices",
                        lambda syms: {"BTC/USD": 57500.0, "ETH/USD": 2900.0})
    monkeypatch.setattr(S, "crypto_executor_from_settings", lambda paper=None: ex)
    S.crypto_stop_monitor()
    assert ex.closed == [("BTC/USD", 57500.0, "stop_loss")]


def test_crypto_stop_monitor_runs_while_paused(monkeypatch):
    """Risk-reducing job: the UI pause must never suppress it."""
    ex = FakeExecutor()
    positions = [FakePos("BTC/USD", 0.001, 61000.0, stop_loss=58000.0)]
    monkeypatch.setattr(S.db, "get_open_positions", lambda a: positions, raising=False)
    monkeypatch.setattr(S.db, "update_position_marks", lambda p: 0, raising=False)
    monkeypatch.setattr(S, "_latest_crypto_prices", lambda syms: {"BTC/USD": 57000.0})
    monkeypatch.setattr(S, "crypto_executor_from_settings", lambda paper=None: ex)
    bot_state.pause()
    S.crypto_stop_monitor()
    assert len(ex.closed) == 1


def test_jobs_dict_exposes_every_registered_job():
    sched = S.build_scheduler()
    assert set(S.JOBS) == {j.id for j in sched.get_jobs()}


# --------------------------- account state ---------------------------

def test_account_state_static_reconstruct(monkeypatch):
    rm = FakeRM(nav=1000.0)
    positions = [FakePos("AAPL", 2, 100.0, 110.0), FakePos("MSFT", 1, 50.0, None)]
    monkeypatch.setattr(S.settings, "crypto_allocation_pct", 0.0, raising=False)
    monkeypatch.setattr(S.db, "get_open_positions", lambda a: positions, raising=False)
    monkeypatch.setattr(S, "get_account_snapshot", lambda rm=None, force=False: _static_snap(1000.0))
    st = S.get_account_state(rm, "stock")
    # invested = 2*110 + 1*50 (falls back to entry when current is None) = 270
    assert st.equity == 1000.0
    assert st.cash == pytest.approx(730.0)
    assert len(st.open_positions) == 2


def test_account_state_broker_cash_buffered(monkeypatch):
    rm = FakeRM(nav=999.0)
    monkeypatch.setattr(S.db, "get_open_positions", lambda a: [], raising=False)
    snap = AccountSnapshot(equity=229.81, cash=120.55, source="broker")
    monkeypatch.setattr(S, "get_account_snapshot", lambda rm=None, force=False: snap)
    st = S.get_account_state(rm, "stock")
    assert st.equity == pytest.approx(229.81)   # the money in the Alpaca account
    # sizing cash is held back 1% so real fills can't trip buying-power errors
    assert st.cash == pytest.approx(120.55 * (1.0 - S.BROKER_CASH_BUFFER))


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


# --------------------------- pause gating (web UI) ---------------------------

def test_pause_blocks_entry_jobs_only(monkeypatch):
    called = {"buys": 0, "sells": 0}
    monkeypatch.setattr(S.settings, "stock_exit_mode", "liquidate", raising=False)
    monkeypatch.setattr(S, "build_risk_manager", lambda: FakeRM())
    monkeypatch.setattr(S.stock_weekly, "run_stock_weekly_buys",
                        lambda **k: called.__setitem__("buys", called["buys"] + 1))
    positions = [FakePos("AAPL", 2, 100.0, 105.0)]
    monkeypatch.setattr(S.db, "get_open_positions", lambda a: positions, raising=False)
    monkeypatch.setattr(S, "_load_stock_frames", lambda tickers, lookback_days=90: {"AAPL": _df(105.0)})
    monkeypatch.setattr(S.db, "update_position_marks", lambda p: 0, raising=False)
    monkeypatch.setattr(S.StockExecutor, "from_settings", classmethod(lambda cls: FakeExecutor()))

    captured = {}
    monkeypatch.setattr(S.stock_weekly, "run_stock_weekly_sells",
                        lambda **k: captured.update(k) or None)

    bot_state.pause()
    S.monday_stock_buys()            # entry job: suppressed
    S.friday_stock_sells()           # risk-reducing job: still runs
    assert called["buys"] == 0
    assert captured["open_positions"][0]["symbol"] == "AAPL"


# --------------------------- dispatch contracts ---------------------------

def test_monday_buys_dispatch_contract(monkeypatch):
    rm = FakeRM(nav=1000.0, halted=False)
    captured = {}
    scored = pd.DataFrame({"ticker": ["AAPL", "MSFT", "NVDA"], "score": [3, 2, 1]})
    universe = {"AAPL": _df(), "MSFT": _df(), "NVDA": _df()}

    monkeypatch.setattr(S.settings, "crypto_allocation_pct", 0.0, raising=False)
    monkeypatch.setattr(S, "build_risk_manager", lambda: rm)
    monkeypatch.setattr(S.db, "get_latest_scored_universe", lambda: scored, raising=False)
    monkeypatch.setattr(S.db, "get_open_positions", lambda a: [], raising=False)
    monkeypatch.setattr(S, "get_account_snapshot", lambda rm=None, force=False: _static_snap(1000.0))
    monkeypatch.setattr(S, "_load_feature_universe", lambda: universe)
    monkeypatch.setattr(S.StockExecutor, "from_settings", classmethod(lambda cls: FakeExecutor()))
    monkeypatch.setattr(S.stock_weekly, "run_stock_weekly_buys",
                        lambda **k: captured.update(k) or None)
    S.monday_stock_buys()
    assert captured["risk_manager"] is rm
    assert captured["equity"] == 1000.0
    assert captured["available_cash"] == 1000.0          # real kwarg name
    assert captured["universe"] is universe              # dict of DataFrames
    assert captured["scored_df"] is scored
    assert captured["executor"] is not None


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
    monkeypatch.setattr(S.settings, "stock_exit_mode", "liquidate", raising=False)
    monkeypatch.setattr(S, "build_risk_manager", lambda: FakeRM())
    monkeypatch.setattr(S.db, "get_open_positions", lambda a: positions, raising=False)
    monkeypatch.setattr(S, "_load_stock_frames",
                        lambda tickers, lookback_days=90: {"AAPL": _df(105.0)})
    monkeypatch.setattr(S.StockExecutor, "from_settings", classmethod(lambda cls: FakeExecutor()))
    monkeypatch.setattr(S.stock_weekly, "run_stock_weekly_sells",
                        lambda **k: captured.update(k) or None)
    S.friday_stock_sells()
    # positions are handed over as dicts with both symbol and ticker keys
    assert captured["open_positions"][0]["symbol"] == "AAPL"
    assert captured["open_positions"][0]["ticker"] == "AAPL"
    assert "AAPL" in captured["universe"]


def test_friday_sells_synthesizes_mark_when_data_missing(monkeypatch):
    positions = [FakePos("AAPL", 2, 100.0, 103.0)]
    captured = {}
    monkeypatch.setattr(S.settings, "stock_exit_mode", "liquidate", raising=False)
    monkeypatch.setattr(S, "build_risk_manager", lambda: FakeRM())
    monkeypatch.setattr(S.db, "get_open_positions", lambda a: positions, raising=False)
    monkeypatch.setattr(S, "_load_stock_frames", lambda tickers, lookback_days=90: {})
    monkeypatch.setattr(S.StockExecutor, "from_settings", classmethod(lambda cls: FakeExecutor()))
    monkeypatch.setattr(S.stock_weekly, "run_stock_weekly_sells",
                        lambda **k: captured.update(k) or None)
    S.friday_stock_sells()
    # exit must never be priced at 0: last mark is synthesized into the universe
    assert float(captured["universe"]["AAPL"]["close"].iloc[-1]) == 103.0


def test_crypto_dispatch_contract(monkeypatch):
    rm = FakeRM(nav=1000.0, halted=False)
    captured = {}
    universe = {"BTC/USD": _df(60000.0)}
    monkeypatch.setattr(S, "build_risk_manager", lambda: rm)
    monkeypatch.setattr(S.db, "get_open_positions", lambda a: [], raising=False)
    monkeypatch.setattr(S.db, "update_position_marks", lambda p: 0, raising=False)
    monkeypatch.setattr(S, "get_account_snapshot", lambda rm=None, force=False: _static_snap(1000.0))
    monkeypatch.setattr(S, "_load_crypto_universe", lambda: universe)
    monkeypatch.setattr(S, "crypto_executor_from_settings", lambda paper=None: FakeExecutor())
    monkeypatch.setattr(S.crypto_24h, "run_crypto_24h_pipeline",
                        lambda **k: captured.update(k) or None)
    S.crypto_cycle()
    assert captured["universe"] is universe              # dict of DataFrames
    assert captured["funding_rates"] == {}               # alpaca spot: no funding
    assert captured["risk_manager"] is rm
    assert captured["btc_only"] is True
    assert captured["entry_mode"] == "regime"            # posture-based default
    # crypto book budget = 25% of equity under the default allocation
    assert captured["available_cash"] == pytest.approx(250.0)


def test_friday_rotation_sells_only_laggards(monkeypatch):
    import pandas as pd
    positions = [FakePos("AAPL", 2, 100.0, 105.0),   # still ranked -> hold
                 FakePos("XYZ", 1, 50.0, 48.0)]      # fell out -> sell
    ranked = pd.DataFrame({"ticker": ["AAPL", "MSFT", "NVDA"], "composite": [3, 2, 1]})
    captured = {}
    monkeypatch.setattr(S.settings, "stock_exit_mode", "rotate", raising=False)
    monkeypatch.setattr(S, "build_risk_manager", lambda: FakeRM())
    monkeypatch.setattr(S.db, "get_open_positions", lambda a: positions, raising=False)
    monkeypatch.setattr(S, "_load_feature_universe", lambda: {"AAPL": _df(105.0)})
    monkeypatch.setattr(S, "_score_and_persist", lambda universe: ranked)
    monkeypatch.setattr(S.StockExecutor, "from_settings", classmethod(lambda cls: FakeExecutor()))
    monkeypatch.setattr(S.stock_weekly, "run_stock_weekly_sells",
                        lambda **k: captured.update(k) or None)
    S.friday_stock_sells()
    sold = [p["symbol"] for p in captured["open_positions"]]
    assert sold == ["XYZ"]


def test_friday_rotation_holds_all_when_scan_fails(monkeypatch):
    positions = [FakePos("AAPL", 2, 100.0, 105.0)]
    called = {"sells": 0}
    monkeypatch.setattr(S.settings, "stock_exit_mode", "rotate", raising=False)
    monkeypatch.setattr(S, "build_risk_manager", lambda: FakeRM())
    monkeypatch.setattr(S.db, "get_open_positions", lambda a: positions, raising=False)
    monkeypatch.setattr(S, "_load_feature_universe", lambda: {})
    monkeypatch.setattr(S.stock_weekly, "run_stock_weekly_sells",
                        lambda **k: called.__setitem__("sells", called["sells"] + 1))
    S.friday_stock_sells()
    assert called["sells"] == 0      # data glitch must not dump the book


def test_crypto_allocation_reserves_budget(monkeypatch):
    """Stocks capped at (1-pct)*equity; crypto keeps its pct*equity slice."""
    rm = FakeRM(nav=1000.0)
    monkeypatch.setattr(S.settings, "crypto_allocation_pct", 0.25, raising=False)
    monkeypatch.setattr(S.db, "get_open_positions",
                        lambda a: [] if a == "crypto"
                        else [FakePos("AAPL", 5, 100.0, 100.0)],
                        raising=False)
    snap = AccountSnapshot(equity=1000.0, cash=500.0, source="broker")
    monkeypatch.setattr(S, "get_account_snapshot", lambda rm=None, force=False: snap)
    stock = S.get_account_state(rm, "stock")
    crypto = S.get_account_state(rm, "crypto")
    # stock budget: 750 - 500 invested = 250 (below buffered cash 495)
    assert stock.cash == pytest.approx(250.0)
    # crypto budget: 250 - 0 invested (below buffered cash 495)
    assert crypto.cash == pytest.approx(250.0)


def test_crypto_cycle_executes_plans(monkeypatch):
    import types
    rm = FakeRM(nav=1000.0, halted=False)
    universe = {"BTC/USD": _df(60000.0)}
    ex = FakeExecutor()
    entry = types.SimpleNamespace(symbol="BTC/USD", units=0.01, entry_price=60000.0,
                                  stop_price=58000.0, take_profit=66000.0)
    exit_plan = types.SimpleNamespace(symbol="ETH/USD", close_price=3000.0, reason="below_ema200")
    result = types.SimpleNamespace(entries=[entry], exits=[exit_plan], persisted=2)

    monkeypatch.setattr(S, "build_risk_manager", lambda: rm)
    monkeypatch.setattr(S.db, "get_open_positions", lambda a: [], raising=False)
    monkeypatch.setattr(S.db, "update_position_marks", lambda p: 0, raising=False)
    monkeypatch.setattr(S, "get_account_snapshot", lambda rm=None, force=False: _static_snap(1000.0))
    monkeypatch.setattr(S, "_load_crypto_universe", lambda: universe)
    monkeypatch.setattr(S, "crypto_executor_from_settings", lambda paper=None: ex)
    monkeypatch.setattr(S.crypto_24h, "run_crypto_24h_pipeline", lambda **k: result)
    S.crypto_cycle()
    assert ex.opened == ["BTC/USD"]
    assert ex.closed == [("ETH/USD", 3000.0, "below_ema200")]


# --------------------------- mid-week stop loss ---------------------------

def test_midweek_closes_only_stopped_out(monkeypatch):
    ex = FakeExecutor()
    # AAPL down 6% -> default 5% stop hit; MSFT down 2% -> hold; NVDA no mark -> skip
    positions = [
        FakePos("AAPL", 1, 100.0, 94.0),
        FakePos("MSFT", 1, 100.0, 98.0),
        FakePos("NVDA", 1, 100.0, None),
    ]
    monkeypatch.setattr(S.db, "get_open_positions", lambda a: positions, raising=False)
    monkeypatch.setattr(S.db, "update_position_marks", lambda p: 0, raising=False)
    monkeypatch.setattr(S, "_load_stock_frames", lambda tickers, lookback_days=10: {})
    monkeypatch.setattr(S.StockExecutor, "from_settings", classmethod(lambda cls: ex))
    S.midweek_stock_monitor()
    assert [c[0] for c in ex.closed] == ["AAPL"]
    # the close is PRICED (old wiring called close_long with no exit price)
    assert ex.closed[0][1] == 94.0
    assert ex.closed[0][2] == "stop_loss"


def test_midweek_respects_stored_stop(monkeypatch):
    ex = FakeExecutor()
    # stored stop 90: a 6% drop to 94 must NOT close (94 > 90)
    positions = [FakePos("AAPL", 1, 100.0, 94.0, stop_loss=90.0)]
    monkeypatch.setattr(S.db, "get_open_positions", lambda a: positions, raising=False)
    monkeypatch.setattr(S.db, "update_position_marks", lambda p: 0, raising=False)
    monkeypatch.setattr(S, "_load_stock_frames", lambda tickers, lookback_days=10: {})
    monkeypatch.setattr(S.StockExecutor, "from_settings", classmethod(lambda cls: ex))
    S.midweek_stock_monitor()
    assert ex.closed == []


# --------------------------- stubs ---------------------------

def test_stubs_run_clean(monkeypatch):
    # Should not raise even with no data/model/DB.
    monkeypatch.setattr(S, "_load_feature_universe", lambda: {})
    S.sunday_ml_retrain()
    S.weekly_performance_report()


# --------------------------- reentrancy guard ---------------------------

def test_double_trigger_does_not_overlap(monkeypatch):
    """Two simultaneous invocations of the same job (double-clicked Run now)
    must not overlap: the second records 'skipped: already running'."""
    import threading
    entered = threading.Event()
    release = threading.Event()

    def slow_rm():
        entered.set()
        release.wait(timeout=5)
        return FakeRM()

    monkeypatch.setattr(S, "build_risk_manager", slow_rm)
    monkeypatch.setattr(S, "get_account_snapshot",
                        lambda rm=None, force=False: _static_snap(1000.0))
    monkeypatch.setattr(S.db, "get_open_positions", lambda a: [], raising=False)

    t = threading.Thread(target=S.daily_heartbeat, daemon=True)
    t.start()
    assert entered.wait(timeout=5)
    S.daily_heartbeat()                       # overlapping second call
    last = S.bot_state.last_runs()["daily_heartbeat"]
    assert last["detail"] == "skipped: already running"
    release.set()
    t.join(timeout=5)


# --------------------------- ML quality gate ---------------------------

def test_low_auc_model_is_quarantined(monkeypatch, tmp_path):
    """A model that tests at coin-flip AUC must NOT earn the 30% blend seat:
    the artifact is moved aside so the scan stays rule-only."""
    import types
    import ml.retrain as retrain_mod
    import pandas as pd

    model_base = tmp_path / "weekly_stock"
    booster = model_base.with_suffix(".json")
    booster.write_text("{}")
    monkeypatch.setattr(S, "MODEL_PATH", str(model_base))
    monkeypatch.setattr(S, "_load_feature_universe", lambda: {"AAPL": pd.DataFrame({"close": [1.0]})})
    monkeypatch.setattr(retrain_mod, "run_rolling_retrain",
                        lambda *a, **k: types.SimpleNamespace(
                            ok=True, n_train=100, n_test=25,
                            metrics={"auc": 0.487}, reason="ok"))
    result = S.sunday_ml_retrain()
    assert "rejected" in result
    assert not booster.exists()                              # moved aside
    assert booster.with_suffix(".json.rejected").exists()
    assert S._load_ml_scorer() is None                       # rule-only now


def test_good_auc_model_is_kept(monkeypatch, tmp_path):
    import types
    import ml.retrain as retrain_mod
    import pandas as pd

    model_base = tmp_path / "weekly_stock"
    booster = model_base.with_suffix(".json")
    booster.write_text("{}")
    monkeypatch.setattr(S, "MODEL_PATH", str(model_base))
    monkeypatch.setattr(S, "_load_feature_universe", lambda: {"AAPL": pd.DataFrame({"close": [1.0]})})
    monkeypatch.setattr(retrain_mod, "run_rolling_retrain",
                        lambda *a, **k: types.SimpleNamespace(
                            ok=True, n_train=100, n_test=25,
                            metrics={"auc": 0.58}, reason="ok"))
    result = S.sunday_ml_retrain()
    assert "retrained" in result
    assert booster.exists()                                  # kept
