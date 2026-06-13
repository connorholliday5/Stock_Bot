"""
ml/features.py
Assemble the ML feature matrix from the Phase 2 TA features.

Only columns the rule scorer already validates as present are used as base
features, so the ML layer cannot drift from the data contract the scorer
relies on. Derived features are computed here from the confirmed `close`
and `atr` columns.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ml.labels import DEFAULT_FEE_BPS, DEFAULT_HORIZON, make_labels

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

# Derived features computed here from confirmed close / atr.
DERIVED_FEATURE_COLS = ["atr_norm", "ret_5d", "ret_10d", "ret_20d"]

FEATURE_COLS = BASE_FEATURE_COLS + DERIVED_FEATURE_COLS

# Columns that must exist on each input frame to build features.
REQUIRED_COLS = set(BASE_FEATURE_COLS) | {"atr", "close"}


def _build_frame_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per ticker feature frame aligned to the input index."""
    out = pd.DataFrame(index=df.index)
    for col in BASE_FEATURE_COLS:
        out[col] = pd.to_numeric(df[col], errors="coerce")

    close = pd.to_numeric(df["close"], errors="coerce")
    atr = pd.to_numeric(df["atr"], errors="coerce")

    out["atr_norm"] = np.where(close > 0, atr / close, np.nan)
    out["ret_5d"] = close / close.shift(5) - 1.0
    out["ret_10d"] = close / close.shift(10) - 1.0
    out["ret_20d"] = close / close.shift(20) - 1.0
    return out


def _frame_dates(df: pd.DataFrame) -> pd.Series:
    if "date" in df.columns:
        return pd.to_datetime(df["date"], errors="coerce")
    return pd.Series(pd.to_datetime(df.index, errors="coerce"), index=df.index)


def build_training_matrix(
    df_universe: dict[str, pd.DataFrame],
    horizon: int = DEFAULT_HORIZON,
    fee_bps: float = DEFAULT_FEE_BPS,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Pooled (ticker, bar) feature matrix with forward return labels.

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
        feats["label"] = make_labels(df["close"], horizon, fee_bps).to_numpy()
        feats["ticker"] = ticker
        feats["date"] = _frame_dates(df).to_numpy()
        frames.append(feats)

    cols = FEATURE_COLS + ["label", "ticker", "date"]
    if not frames:
        return pd.DataFrame(columns=cols), skipped

    full = pd.concat(frames, ignore_index=True)
    full = full.dropna(subset=FEATURE_COLS + ["label"]).reset_index(drop=True)
    full["label"] = full["label"].astype(int)
    return full, skipped


def build_inference_matrix(
    df_universe: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Latest bar feature matrix, one row per ticker, indexed by ticker.
    Matches the scorer convention of using the last row per frame.
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
        if last[FEATURE_COLS].isna().any():
            skipped[ticker] = "NaN in latest feature row"
            continue
        rows[ticker] = last[FEATURE_COLS]

    if not rows:
        return pd.DataFrame(columns=FEATURE_COLS), skipped

    X = pd.DataFrame.from_dict(rows, orient="index")[FEATURE_COLS]
    X.index.name = "ticker"
    return X, skipped
