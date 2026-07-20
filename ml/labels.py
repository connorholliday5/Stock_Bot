"""
ml/labels.py
Forward return labels for the weekly stock model.

Two label definitions:

  absolute (make_labels) - 1 if a stock's own forward return over `horizon`
    bars clears the round-trip fee. Legacy default. Problem for a rotation
    strategy: it mostly encodes "was the whole market up this week", which
    the bot cannot act on because it picks AMONG stocks, not the market.

  relative (relative_labels) - 1 if a stock's forward return BEATS the
    cross-sectional median that same week. This is the right target for a
    strategy that ranks stocks against each other: the model learns relative
    strength, and classes stay ~balanced (good for AUC). This is the new
    default used by build_training_matrix.

The final `horizon` bars of each series have no forward window and are
labelled NaN (dropped downstream).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_HORIZON = 5        # approx one trading week of daily bars
DEFAULT_FEE_BPS = 10.0     # round trip cost in basis points (0.10 percent)


def forward_return(close: pd.Series, horizon: int = DEFAULT_HORIZON) -> pd.Series:
    """Forward simple return over `horizon` bars: close[t+h] / close[t] - 1."""
    close = pd.to_numeric(pd.Series(close), errors="coerce")
    fwd = close.shift(-horizon) / close - 1.0
    return fwd.rename("forward_return")


def make_labels(
    close: pd.Series,
    horizon: int = DEFAULT_HORIZON,
    fee_bps: float = DEFAULT_FEE_BPS,
) -> pd.Series:
    """
    ABSOLUTE binary label: 1 if forward return clears the round trip fee,
    else 0. NaN where the forward window runs past the end of the series.
    """
    fwd = forward_return(close, horizon)
    threshold = float(fee_bps) / 10000.0
    label = (fwd > threshold).astype("float64")
    label[fwd.isna()] = np.nan
    return label.rename("label")


def relative_labels(
    pooled: pd.DataFrame,
    ret_col: str = "forward_return",
    date_col: str = "date",
) -> pd.Series:
    """
    RELATIVE (cross-sectional) binary label over a POOLED (ticker, bar) frame:
    1 if a row's forward return beats the median forward return of all stocks
    on the same date, else 0. NaN where the forward return is NaN.

    A stock that returns +1% in a week where the median stock returned +2% is
    a *relative* loser and correctly labelled 0 - exactly the distinction a
    rotation strategy needs, and the one an absolute label misses.
    """
    fwd = pd.to_numeric(pooled[ret_col], errors="coerce")
    median = fwd.groupby(pooled[date_col]).transform("median")
    label = (fwd > median).astype("float64")
    label[fwd.isna()] = np.nan
    return label.rename("label")
