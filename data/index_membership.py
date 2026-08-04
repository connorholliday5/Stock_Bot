"""
data/index_membership.py
Point-in-time S&P 500 membership: who was actually in the index on date T.

WHY THIS EXISTS
---------------
`data/sp500.py` is TODAY's membership. Ranking that list across history means
every backtest only ever considers companies that survived to the present -
the ones that blew up, got acquired at a discount, or were dropped for poor
performance are invisible. Momentum is hit harder by this than buy-and-hold,
because momentum systematically buys the extended, high-beta names that are
the most likely to later crater and leave the index. A published Nasdaq-100
momentum test scored 46% CAGR on current members and 16.4% once delisted
names were restored, with drawdown going from -41% to -83%.

That is the leading explanation for our own unexplained gap: our momentum
backtest reported 24.9% CAGR while MTUM - a real, tradeable momentum ETF
whose record contains every name that died - returned 10.3%.

DATA
----
`index/sp500_history.csv.gz`  (from github.com/fja05680/sp500)
    `date,tickers` - one full membership snapshot per change date,
    1996-01-02 through 2019-01-11, ~504 names each. Tickers that later left
    the index carry a `-YYYYMM` suffix recording when (e.g. `AAL-199702`,
    `AAMRQ-201312`). Symbology is dot-style for class shares (`BRK.B`),
    matching data/sp500.py and Alpaca.

`index/sp500_changes_since_2019.csv`  (same repo)
    `date,add,remove` - deltas replayed forward from the last snapshot.

Both files are committed rather than fetched, so backtests are offline,
deterministic, and reproducible across machines.

HONEST LIMITATION - READ BEFORE TRUSTING A NUMBER
-------------------------------------------------
This is a PARTIAL correction, not a cure. It stops us ranking companies that
were not in the index yet, which removes one direction of the bias. It does
NOT resurrect the losers: Alpaca has no price history for names that were
delisted years ago, so those bars simply do not exist and cannot be traded
in replay. Expect results to move toward reality without reaching it, and
expect a residual gap versus MTUM. That gap stays a documented caveat - not
evidence that the strategy is good and the ETF is wrong.

One more wrinkle: tickers get reused across decades. Stripping the suffix
maps `AAL-199702` (Alexander & Alexander, delisted 1997) onto the same
symbol as today's AAL (American Airlines). Harmless while price data only
covers recent years - recent dates draw on recent snapshots - but a future
data source with deep history would need the delisting month respected,
not just stripped.
"""

from __future__ import annotations

import csv
import gzip
import logging
import re
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "index"
SNAPSHOT_FILE = DATA_DIR / "sp500_history.csv.gz"
CHANGES_FILE = DATA_DIR / "sp500_changes_since_2019.csv"

# A trailing "-YYYYMM" marks the month the name left the index. Real tickers
# in this dataset never contain a hyphen, so the pattern is unambiguous.
_DELISTED_SUFFIX = re.compile(r"-\d{6}$")


def strip_delisting_suffix(ticker: str) -> str:
    """`AAMRQ-201312` -> `AAMRQ`; `BRK.B` -> `BRK.B`."""
    return _DELISTED_SUFFIX.sub("", (ticker or "").strip())


def delisting_month(ticker: str) -> Optional[str]:
    """`AAMRQ-201312` -> `'201312'`, or None for a name still listed."""
    m = _DELISTED_SUFFIX.search((ticker or "").strip())
    return m.group(0)[1:] if m else None


@dataclass
class Membership:
    """Snapshot dates (ascending) and the membership set at each one."""

    dates: list[pd.Timestamp]
    sets: list[frozenset[str]]

    def __post_init__(self) -> None:
        # members_on binary-searches `dates`, which silently returns the
        # wrong snapshot if the file is ever published out of order. Sorting
        # here costs nothing and removes the failure mode entirely.
        if self.dates != sorted(self.dates):
            order = sorted(range(len(self.dates)), key=lambda i: self.dates[i])
            self.dates = [self.dates[i] for i in order]
            self.sets = [self.sets[i] for i in order]

    def __len__(self) -> int:
        return len(self.dates)

    def members_on(self, date) -> frozenset[str]:
        """Membership in effect on `date`.

        Uses the most recent snapshot at or before `date` - never a later
        one, which would be look-ahead. Dates before the first snapshot fall
        back to the earliest known membership; that is an approximation, but
        an approximation made from the PAST, so it cannot leak the future.
        """
        if not self.dates:
            return frozenset()
        ts = pd.Timestamp(date)
        # Alpaca frames are tz-aware UTC; the snapshot dates are naive.
        # Comparing the two raises, so strip the tz here - a calendar date
        # is a calendar date. Without this, every real-data backtest with
        # the filter on dies (or worse, gets silently swallowed upstream).
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        ts = ts.normalize()
        i = bisect_right(self.dates, ts) - 1
        return self.sets[max(i, 0)]

    @property
    def coverage(self) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        return (self.dates[0], self.dates[-1]) if self.dates else (None, None)


def _read_snapshots(path: Path) -> tuple[list[pd.Timestamp], list[frozenset[str]]]:
    dates: list[pd.Timestamp] = []
    sets: list[frozenset[str]] = []
    with gzip.open(path, "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            raw = (row.get("tickers") or "").strip()
            if not raw:
                continue
            names = {strip_delisting_suffix(t) for t in raw.split(",") if t.strip()}
            names.discard("")
            dates.append(pd.Timestamp(row["date"]).normalize())
            sets.append(frozenset(names))
    return dates, sets


def _split(cell: str) -> list[str]:
    return [t.strip() for t in (cell or "").split(",") if t.strip()]


def _apply_changes(path: Path, dates: list[pd.Timestamp],
                   sets: list[frozenset[str]]) -> None:
    """Replay `date,add,remove` deltas forward from the last snapshot.

    Mutates `dates`/`sets` in place. Rows dated at or before the final
    snapshot are skipped - the snapshot already reflects them.
    """
    if not dates:
        return
    cutoff = dates[-1]
    current = set(sets[-1])
    with path.open("r", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("date") or "").strip()]
    for row in sorted(rows, key=lambda r: r["date"]):
        ts = pd.Timestamp(row["date"]).normalize()
        if ts <= cutoff:
            continue
        for t in _split(row.get("remove", "")):
            current.discard(strip_delisting_suffix(t))
        for t in _split(row.get("add", "")):
            current.add(strip_delisting_suffix(t))
        dates.append(ts)
        sets.append(frozenset(current))


_CACHE: Optional[Membership] = None
_WARNED = False


def load_membership(force: bool = False) -> Membership:
    """Parse both files once and cache. Missing data yields an EMPTY
    Membership rather than an exception, so a checkout without the CSVs
    degrades to the old (biased) behaviour loudly instead of crashing."""
    global _CACHE, _WARNED
    if _CACHE is not None and not force:
        return _CACHE
    if force:
        _WARNED = False
    try:
        if not SNAPSHOT_FILE.exists():
            raise FileNotFoundError(SNAPSHOT_FILE)
        dates, sets = _read_snapshots(SNAPSHOT_FILE)
        if CHANGES_FILE.exists():
            _apply_changes(CHANGES_FILE, dates, sets)
        else:
            logger.warning("index_membership: %s missing; membership frozen at %s",
                           CHANGES_FILE.name, dates[-1].date() if dates else "n/a")
        _CACHE = Membership(dates=dates, sets=sets)
        lo, hi = _CACHE.coverage
        logger.info("index_membership: %d snapshots, %s -> %s",
                    len(_CACHE), lo.date() if lo is not None else "?",
                    hi.date() if hi is not None else "?")
    except Exception as exc:
        if not _WARNED:
            logger.warning("index_membership unavailable (%s); backtests will run "
                           "on TODAY's membership and stay survivorship-biased", exc)
            _WARNED = True
        _CACHE = Membership(dates=[], sets=[])
    return _CACHE


def members_on(date) -> frozenset[str]:
    """Point-in-time S&P 500 membership. Empty set means 'unknown' - callers
    must treat that as "apply no filter", never as "no eligible names"."""
    return load_membership().members_on(date)


_EMPTY_WARNED = False


def filter_to_members(symbols: Iterable[str], date) -> list[str]:
    """Keep only `symbols` that were in the index on `date`.

    Returns the input unchanged when membership data is unavailable, so a
    missing dataset cannot silently empty a backtest's candidate pool.

    Filtering everything away is reported once, loudly: with a custom or
    non-equity universe (crypto, a personal watchlist, synthetic test data)
    the S&P membership list simply does not apply, and a silently empty
    candidate pool looks exactly like a strategy that chose not to trade.
    """
    global _EMPTY_WARNED
    syms = list(symbols)
    members = members_on(date)
    if not members:
        return syms
    kept = [s for s in syms if s in members]
    if syms and not kept and not _EMPTY_WARNED:
        logger.warning(
            "index_membership: none of %d symbols were S&P 500 members on %s "
            "(e.g. %s) - the point-in-time filter is removing everything. If "
            "this is a custom or non-equity universe, disable it with "
            "point_in_time_membership=False.",
            len(syms), pd.Timestamp(date).date(), ", ".join(syms[:5]))
        _EMPTY_WARNED = True
    return kept


def is_available() -> bool:
    """True when point-in-time filtering is actually in effect. Backtest
    reports should print this rather than claiming a correction they did
    not apply."""
    return len(load_membership()) > 0
