"""
ml/labels.py
Forward return labels for the weekly stock model.

Label definition (Phase 7 default): next-week return positive net of fees.
For each (ticker, bar) the label is 1 if the forward return over `horizon`
trading bars exceeds the round-trip fee cost, else 0. The final `horizon`
bars of each series have no forward window and are labelled NaN (dropped
downstream).
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
    Binary label: 1 if forward return clears the round trip fee, else 0.
    NaN where the forward window runs past the end of the series.
    """
    fwd = forward_return(close, horizon)
    threshold = float(fee_bps) / 10000.0
    label = (fwd > threshold).astype("float64")
    label[fwd.isna()] = np.nan
    return label.rename("label")
