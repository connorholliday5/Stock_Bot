"""
reporting/html_report.py
Phase 8 - static HTML weekly report writer. No external deps, inline CSS,
ASCII only. Writes reports/weekly_<YYYY>-W<WW>.html and reports/latest.html.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

from reporting.pnl import WeeklyReport

DEFAULT_OUT_DIR = "reports"


def _money(x) -> str:
    try:
        return f"${float(x):,.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def _pct(x) -> str:
    try:
        return f"{float(x):.1f}%"
    except (TypeError, ValueError):
        return "-"


def _qty(x) -> str:
    try:
        s = f"{float(x):,.4f}".rstrip("0").rstrip(".")
        return s if s else "0"
    except (TypeError, ValueError):
        return "0"


def _dt(x) -> str:
    if isinstance(x, datetime):
        return x.strftime("%Y-%m-%d %H:%M").strip()
    return "-"


def _e(x) -> str:
    return html.escape(str(x))


def _sign(x) -> str:
    return "pos" if _safe_float(x) >= 0 else "neg"


def _safe_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _table(headers, rows) -> str:
    th = "".join(f"<th>{_e(h)}</th>" for h in headers)
    if not rows:
        body = f"<tr><td colspan='{len(headers)}' class='empty'>None</td></tr>"
    else:
        body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def render_weekly_report(report: WeeklyReport) -> str:
    asset_rows = []
    for asset, a in sorted(report.by_asset.items()):
        asset_rows.append([
            _e(asset.upper()),
            f"<span class='{_sign(a.get('net_pnl', 0))}'>{_money(a.get('net_pnl', 0))}</span>",
            _e(a.get("trades", 0)),
            _e(a.get("wins", 0)),
        ])

    pos_rows = []
    for p in report.open_positions:
        pos_rows.append([
            _e(p["symbol"]),
            _e(p["asset_type"].upper()),
            _qty(p["qty"]),
            _money(p["entry"]),
            _money(p["mark"]),
            f"<span class='{_sign(p['upnl'])}'>{_money(p['upnl'])}</span>",
        ])

    trade_rows = []
    for t in report.recent_trades:
        trade_rows.append([
            _dt(t["closed_at"]),
            _e(t["symbol"]),
            _e(t["asset_type"].upper()),
            f"<span class='{_sign(t['pnl'])}'>{_money(t['pnl'])}</span>",
            _pct(t["pnl_pct"]) if t["pnl_pct"] is not None else "-",
            _e(t["reason"]),
        ])

    model_rows = []
    for m in report.model_trend:
        model_rows.append([
            _dt(m["when"]),
            _e(m["model"]),
            f"{_safe_float(m['auc']):.3f}" if m["auc"] is not None else "-",
            f"{_safe_float(m['accuracy']):.3f}" if m["accuracy"] is not None else "-",
        ])

    net_cls = _sign(report.net_pnl)
    summary_table = _table(
        ["Metric", "Value"],
        [
            ["Net P&L", f"<span class='{net_cls}'>{_money(report.net_pnl)}</span>"],
            ["Gross Win", _money(report.gross_win)],
            ["Gross Loss", _money(report.gross_loss)],
            ["Win Rate", _pct(report.win_rate)],
            ["Trades", _e(report.total_trades)],
            ["Wins / Losses", f"{_e(report.wins)} / {_e(report.losses)}"],
        ],
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weekly Report W{report.week_index:02d}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif;
         background: #0e1116; color: #e6edf3; margin: 0; padding: 24px; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .meta {{ color: #8b949e; font-size: 13px; margin-bottom: 20px; }}
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
          padding: 16px 20px; min-width: 160px; }}
  .card .label {{ color: #8b949e; font-size: 12px; text-transform: uppercase; }}
  .card .value {{ font-size: 24px; font-weight: 600; margin-top: 4px; }}
  h2 {{ font-size: 15px; margin: 24px 0 8px; color: #c9d1d9; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 8px;
          background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }}
  th, td {{ text-align: left; padding: 8px 12px; font-size: 13px;
           border-bottom: 1px solid #21262d; }}
  th {{ background: #1c2230; color: #8b949e; text-transform: uppercase; font-size: 11px; }}
  tr:last-child td {{ border-bottom: none; }}
  .empty {{ color: #6e7681; text-align: center; font-style: italic; }}
  .pos {{ color: #3fb950; }}
  .neg {{ color: #f85149; }}
</style>
</head>
<body>
  <h1>Weekly Performance Report - Week {report.week_index:02d}</h1>
  <div class="meta">
    {_e(report.week_start.strftime('%Y-%m-%d'))} to {_e(report.week_end.strftime('%Y-%m-%d'))}
    &middot; {_e(report.environment.upper())}
    &middot; generated {_e(report.generated_at.strftime('%Y-%m-%d %H:%M'))}
  </div>
  <div class="cards">
    <div class="card"><div class="label">NAV</div><div class="value">{_money(report.nav)}</div></div>
    <div class="card"><div class="label">Net P&L</div><div class="value {net_cls}">{_money(report.net_pnl)}</div></div>
    <div class="card"><div class="label">Win Rate</div><div class="value">{_pct(report.win_rate)}</div></div>
    <div class="card"><div class="label">Trades</div><div class="value">{report.total_trades}</div></div>
  </div>
  <h2>Summary</h2>
  {summary_table}
  <h2>By Asset</h2>
  {_table(["Asset", "Net P&L", "Trades", "Wins"], asset_rows)}
  <h2>Open Positions</h2>
  {_table(["Symbol", "Asset", "Qty", "Entry", "Mark", "Unrealized"], pos_rows)}
  <h2>Recent Trades</h2>
  {_table(["Closed", "Symbol", "Asset", "P&L", "Return", "Reason"], trade_rows)}
  <h2>Model Metric Trend</h2>
  {_table(["Trained", "Model", "AUC", "Accuracy"], model_rows)}
</body>
</html>"""


def write_report(report: WeeklyReport, out_dir: str = DEFAULT_OUT_DIR) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    html_str = render_weekly_report(report)
    name = f"weekly_{report.week_start.strftime('%Y')}-W{report.week_index:02d}.html"
    path = out / name
    path.write_text(html_str, encoding="utf-8")
    (out / "latest.html").write_text(html_str, encoding="utf-8")
    return path
