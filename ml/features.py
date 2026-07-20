"""
ml/features.py
Assemble the ML feature matrix from the Phase 2 TA features.

Two feature families:

  per-stock features - the base TA columns the rule scorer already validates
    (rsi, momentum, volume_ratio, trend flags) plus derived returns and
    ATR%. These describe a stock in isolation.

  cross-sectional (XS) features - each per-stock value re-expressed as its
    PERCENTILE RANK among all stocks on the same date (suffix `_xs`). "RSI 55"
    is weak; "RSI in the 30th percentile of the market today" is strong. For a
    strategy that ranks stocks against each other, relative position is the
    signal - so these are computed per date in training and across the current
    universe at inference (point-in-time correct: only uses data known now).

Only columns the rule scorer already validates as present are used as base
features, so the ML layer cannot drift from the data contract the scorer
relies on. Derived features are computed here from the confirmed `close`
and `atr` columns.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ml.labels import (
    DEFAULT_FEE_BPS,
    DEFAULT_HORIZON,
    forward_return,
    make_labels,
    relative_labels,
)

logger = logging.getLogger(__name__)

# Base TA columns the scorer already requires present (the data contract).
BASE_FEATURE_COLS = [
    "rsi",
    "momentum",
    "volume_ratio",
    "above_sma50",
    "above_sma200",
    "golden_cross",
]

# Derived features computed here from confirmed close / atr. ret_60d is the
# classic 3-month momentum factor; multiple horizons let the model weigh
# short vs. medium-term trend.
DERIVED_FEATURE_COLS = ["atr_norm", "ret_5d", "ret_10d", "ret_20d", "ret_60d"]

# Per-stock columns re-expressed as cross-sectional percentile ranks per date.
XS_RANK_SOURCE = ["momentum", "ret_20d", "ret_60d", "rsi", "volume_ratio", "atr_norm"]
XS_FEATURE_COLS = [c + "_xs" for c in XS_RANK_SOURCE]

FEATURE_COLS = BASE_FEATURE_COLS + DERIVED_FEATURE_COLS + XS_FEATURE_COLS

# Columns that must exist on each input frame to build features.
REQUIRED_COLS = set(BASE_FEATURE_COLS) | {"atr", "close"}

# Non-XS features that must be present before a row/ticker is kept.
_CORE_FEATURE_COLS = BASE_FEATURE_COLS + DERIVED_FEATURE_COLS


def _build_frame_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per ticker feature frame (base + derived) aligned to the input index."""
    out = pd.DataFrame(index=df.index)
    for col in BASE_FEATURE_COLS:
        out[col] = pd.to_numeric(df[col], errors="coerce")

    close = pd.to_numeric(df["close"], errors="coerce")
    atr = pd.to_numeric(df["atr"], errors="coerce")

    out["atr_norm"] = np.where(close > 0, atr / close, np.nan)
    out["ret_5d"] = close / close.shift(5) - 1.0
    out["ret_10d"] = close / close.shift(10) - 1.0
    out["ret_20d"] = close / close.shift(20) - 1.0
    out["ret_60d"] = close / close.shift(60) - 1.0
    return out


def _attach_xs_ranks(frame: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Add cross-sectional percentile-rank columns (`*_xs`) in [0, 1].

    Training: rank within each `by` (date) group. Inference: rank across all
    rows (every row is 'as of now'). rank(pct=True) is NaN only where the
    source is NaN, so XS columns share the source's drop mask.
    """
    for src, dst in zip(XS_RANK_SOURCE, XS_FEATURE_COLS):
        s = pd.to_numeric(frame[src], errors="coerce")
        if by is not None:
            frame[dst] = s.groupby(frame[by]).rank(pct=True)
        else:
            frame[dst] = s.rank(pct=True)
    return frame


def _frame_dates(df: pd.DataFrame) -> pd.Series:
    if "date" in df.columns:
        return pd.to_datetime(df["date"], errors="coerce")
    return pd.Series(pd.to_datetime(df.index, errors="coerce"), index=df.index)


def build_training_matrix(
    df_universe: dict[str, pd.DataFrame],
    horizon: int = DEFAULT_HORIZON,
    fee_bps: float = DEFAULT_FEE_BPS,
    label_mode: str = "relative",
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Pooled (ticker, bar) feature matrix with labels.

    label_mode:
      "relative" (default) - 1 if the stock beat the cross-sectional median
        forward return that date (the right target for a rotation strategy).
      "absolute" - 1 if the stock's own forward return cleared the fee.

    Returns
    -------
    (matrix, skipped)
        matrix  : DataFrame with FEATURE_COLS + label, ticker, date.
                  Rows with any NaN feature or NaN label are dropped.
        skipped : ticker -> reason for tickers excluded before feature build.
    """
    frames: list[pd.DataFrame] = []
    skipped: dict[str, str] = {}

    for ticker, df in df_universe.items():
        if df is None or df.empty:
            skipped[ticker] = "empty DataFrame"
            continue
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            skipped[ticker] = f"missing columns: {sorted(missing)}"
            continue

        feats = _build_frame_features(df)
        feats["forward_return"] = forward_return(df["close"], horizon).to_numpy()
        feats["ticker"] = ticker
        feats["date"] = _frame_dates(df).to_numpy()
        frames.append(feats)

    cols = FEATURE_COLS + ["label", "ticker", "date"]
    if not frames:
        return pd.DataFrame(columns=cols), skipped

    full = pd.concat(frames, ignore_index=True)
    full = _attach_xs_ranks(full, by="date")

    if label_mode == "absolute":
        threshold = float(fee_bps) / 10000.0
        lab = (full["forward_return"] > threshold).astype("float64")
        lab[full["forward_return"].isna()] = np.nan
        full["label"] = lab
    else:
        full["label"] = relative_labels(full)

    full = full.dropna(subset=FEATURE_COLS + ["label"]).reset_index(drop=True)
    full["label"] = full["label"].astype(int)
    return full[cols], skipped


def build_inference_matrix(
    df_universe: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Latest bar feature matrix, one row per ticker, indexed by ticker.
    Cross-sectional ranks are taken across the current universe's latest bars
    (all 'as of now'), matching how training ranks within a date.
    """
    rows: dict[str, pd.Series] = {}
    skipped: dict[str, str] = {}

    for ticker, df in df_universe.items():
        if df is None or df.empty:
            skipped[ticker] = "empty DataFrame"
            continue
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            skipped[ticker] = f"missing columns: {sorted(missing)}"
            continue

        feats = _build_frame_features(df)
        last = feats.iloc[-1]
        if last[_CORE_FEATURE_COLS].isna().any():
            skipped[ticker] = "NaN in latest feature row"
            continue
        rows[ticker] = last[_CORE_FEATURE_COLS]

    if not rows:
        return pd.DataFrame(columns=FEATURE_COLS), skipped

    frame = pd.DataFrame.from_dict(rows, orient="index")[_CORE_FEATURE_COLS]
    frame = _attach_xs_ranks(frame, by=None)
    X = frame[FEATURE_COLS]
    X.index.name = "ticker"
    return X, skipped
