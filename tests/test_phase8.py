# ============================================================
# tests/test_phase8.py
# Phase 8 - monitoring and reporting. Pure computation + HTML render.
# No DB, no Telegram, no scheduler import.
# Run: pytest tests/test_phase8.py -q
# ============================================================

from types import SimpleNamespace
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from reporting.pnl import (
    summarize_trades, normalize_positions, recent_trades, model_trend,
    current_week_window, filter_week, build_weekly_report, WeeklyReport,
)
from reporting.html_report import render_weekly_report, write_report

ET = ZoneInfo("America/New_York")


def _trade(pnl, asset="stock", **kw):
    return SimpleNamespace(pnl=pnl, asset_type=asset, **kw)


def test_summarize_empty():
    s = summarize_trades([])
    assert s["net_pnl"] == 0.0
    assert s["total_trades"] == 0
    assert s["win_rate"] == 0.0


def test_summarize_mixed():
    s = summarize_trades([_trade(100), _trade(-50), _trade(30), _trade(-10)])
    assert s["net_pnl"] == 70.0
    assert s["wins"] == 2
    assert s["losses"] == 2
    assert s["total_trades"] == 4
    assert s["win_rate"] == 50.0
    assert s["gross_win"] == 130.0
    assert s["gross_loss"] == -60.0


def test_summarize_zero_pnl_not_win_or_loss():
    s = summarize_trades([_trade(0), _trade(10)])
    assert s["wins"] == 1
    assert s["losses"] == 0
    assert s["total_trades"] == 2
    assert s["win_rate"] == 50.0


def test_summarize_tolerant_pnl_attr():
    s = summarize_trades([SimpleNamespace(realized_pnl=25, asset_type="stock")])
    assert s["net_pnl"] == 25.0
    assert s["wins"] == 1


def test_summarize_by_asset_split():
    s = summarize_trades([_trade(100, "stock"), _trade(-20, "crypto"), _trade(5, "crypto")])
    assert s["by_asset"]["stock"]["net_pnl"] == 100.0
    assert s["by_asset"]["crypto"]["net_pnl"] == -15.0
    assert s["by_asset"]["crypto"]["trades"] == 2


def test_summarize_all_wins_full_rate():
    s = summarize_trades([_trade(1), _trade(2), _trade(3)])
    assert s["win_rate"] == 100.0


def test_current_week_window_starts_monday_midnight():
    now = datetime(2026, 5, 27, 12, 0, tzinfo=ET)  # Wednesday
    start, end = current_week_window(now)
    assert start.weekday() == 0
    assert (start.hour, start.minute, start.second) == (0, 0, 0)
    assert start <= now <= end


def test_current_week_window_naive_now():
    now = datetime(2026, 5, 27, 12, 0)  # naive -> treated as ET
    start, end = current_week_window(now)
    assert start.weekday() == 0


def test_filter_week_dated_and_undated():
    start, end = current_week_window(datetime(2026, 5, 27, 12, 0, tzinfo=ET))
    inside = _trade(10, closed_at=datetime(2026, 5, 26, 10, 0, tzinfo=ET))
    outside = _trade(10, closed_at=datetime(2026, 5, 1, 10, 0, tzinfo=ET))
    undated = _trade(10)
    kept = filter_week([inside, outside, undated], start, end)
    assert inside in kept
    assert undated in kept
    assert outside not in kept


def test_normalize_positions_tolerant_fields():
    p = SimpleNamespace(ticker="AAPL", quantity=2, avg_entry=100.0, current_price=110.0)
    out = normalize_positions([p], "stock")
    assert out[0]["symbol"] == "AAPL"
    assert out[0]["qty"] == 2.0
    assert out[0]["entry"] == 100.0
    assert out[0]["mark"] == 110.0
    assert out[0]["upnl"] == 20.0
    assert out[0]["asset_type"] == "stock"


def test_normalize_positions_missing_entry_zero_upnl():
    p = SimpleNamespace(symbol="X", qty=0, entry_price=0)
    out = normalize_positions([p])
    assert out[0]["upnl"] == 0.0


def test_recent_trades_limit():
    trades = [_trade(1) for _ in range(40)]
    assert len(recent_trades(trades, limit=15)) == 15


def test_recent_trades_sorted_newest_first():
    a = _trade(1, symbol="A", closed_at=datetime(2026, 5, 20, tzinfo=ET))
    b = _trade(1, symbol="B", closed_at=datetime(2026, 5, 25, tzinfo=ET))
    out = recent_trades([a, b])
    assert out[0]["symbol"] == "B"
    assert out[1]["symbol"] == "A"


def test_model_trend_maps_and_sorts():
    r1 = SimpleNamespace(created_at=datetime(2026, 5, 1, tzinfo=ET), auc=0.70, accuracy=0.65)
    r2 = SimpleNamespace(created_at=datetime(2026, 5, 8, tzinfo=ET), roc_auc=0.76, acc=0.68)
    out = model_trend([r1, r2], limit=8)
    assert out[0]["auc"] == 0.76
    assert out[1]["auc"] == 0.70
    assert out[0]["accuracy"] == 0.68


def test_build_weekly_report_injected():
    pos = [SimpleNamespace(symbol="AAPL", qty=1, entry_price=100, current_price=105)]
    report = build_weekly_report(
        now=datetime(2026, 5, 27, 16, 30, tzinfo=ET),
        nav=1234.56,
        stock_positions=pos,
        crypto_positions=[],
        trades=[_trade(50), _trade(-20)],
        model_rows=[],
        environment="paper",
    )
    assert isinstance(report, WeeklyReport)
    assert report.nav == 1234.56
    assert report.total_trades == 2
    assert report.net_pnl == 30.0
    assert len(report.open_positions) == 1
    assert report.open_positions[0]["symbol"] == "AAPL"


def test_build_weekly_report_degrades_empty():
    report = build_weekly_report(
        now=datetime(2026, 5, 27, 16, 30, tzinfo=ET),
        nav=1000.0,
        stock_positions=[],
        crypto_positions=[],
        trades=[],
        model_rows=[],
    )
    assert report.total_trades == 0
    assert report.net_pnl == 0.0
    assert report.open_positions == []


def test_render_weekly_report_html():
    pos = [SimpleNamespace(symbol="MSFT", qty=1, entry_price=100, current_price=120)]
    report = build_weekly_report(
        now=datetime(2026, 5, 27, 16, 30, tzinfo=ET),
        nav=2000.0,
        stock_positions=pos,
        crypto_positions=[],
        trades=[_trade(75)],
        model_rows=[],
    )
    html_str = render_weekly_report(report)
    assert "<!DOCTYPE html>" in html_str
    assert "MSFT" in html_str
    assert "$2,000.00" in html_str
    assert "Weekly Performance Report" in html_str


def test_write_report_creates_files(tmp_path):
    report = build_weekly_report(
        now=datetime(2026, 5, 27, 16, 30, tzinfo=ET),
        nav=1500.0,
        stock_positions=[],
        crypto_positions=[],
        trades=[_trade(10)],
        model_rows=[],
    )
    path = write_report(report, out_dir=str(tmp_path))
    assert path.exists()
    assert (tmp_path / "latest.html").exists()
    assert "<!DOCTYPE html>" in path.read_text(encoding="utf-8")
