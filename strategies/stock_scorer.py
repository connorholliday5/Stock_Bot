"""
strategies/stock_scorer.py
Sunday night composite scoring of validated S&P 500 universe.
Returns a ranked DataFrame of top N tickers ready for Monday execution.

Phase 7: optional ML probability blend. When an ml_scorer is supplied with a
positive ml_weight, the model probability is blended into the composite. It
does not replace the rule score and does not gate entries. With no ml_scorer
(or ml_weight 0.0) behaviour is identical to the pre Phase 7 scorer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring weights - must sum to 1.0
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    "rsi_score":        0.20,
    "momentum_score":   0.25,
    "volume_score":     0.20,
    "trend_score":      0.25,
    "volatility_score": 0.10,
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# RSI sweet spot: we want pullback setups - reward mid-range RSI (40-60),
# penalise overbought (>75) and oversold (<30).
RSI_IDEAL_LOW  = 40.0
RSI_IDEAL_HIGH = 60.0
RSI_OB_CUTOFF  = 75.0
RSI_OS_CUTOFF  = 30.0

DEFAULT_TOP_N = 10

# Default blend weight on the ML probability when an ml_scorer is supplied.
# The scheduler passes this explicitly; the function default stays 0.0 so the
# rule scorer is unchanged unless ML is opted in.
DEFAULT_ML_WEIGHT = 0.30


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ScoringResult:
    ranked: pd.DataFrame          # columns: ticker + score components + composite
    excluded: dict[str, str] = field(default_factory=dict)  # ticker -> reason
    score_date: Optional[pd.Timestamp] = None

    def top(self, n: int = DEFAULT_TOP_N) -> pd.DataFrame:
        return self.ranked.head(n)

    def summary(self) -> str:
        lines = [
            f"Scored {len(self.ranked)} tickers  |  excluded {len(self.excluded)}",
            f"Score date: {self.score_date}",
        ]
        if not self.ranked.empty:
            top = self.ranked.head(5)
            lines.append("Top 5:")
            for _, row in top.iterrows():
                lines.append(
                    f"  {row['ticker']:<8}  composite={row['composite']:.4f}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Component scorers - each returns a Series[float] in [0, 1]
# ---------------------------------------------------------------------------

def _score_rsi(df_universe: dict[str, pd.DataFrame]) -> pd.Series:
    """
    RSI score: reward 40-60 range (momentum without overextension).
    Scores decay smoothly outside the ideal band.
    """
    scores: dict[str, float] = {}
    for ticker, df in df_universe.items():
        rsi_vals = df["rsi"].dropna()
        if rsi_vals.empty:
            scores[ticker] = 0.0
            continue
        rsi = float(rsi_vals.iloc[-1])

        if RSI_IDEAL_LOW <= rsi <= RSI_IDEAL_HIGH:
            score = 1.0
        elif rsi < RSI_OS_CUTOFF:
            score = 0.1
        elif rsi > RSI_OB_CUTOFF:
            score = 0.1
        elif rsi < RSI_IDEAL_LOW:
            score = (rsi - RSI_OS_CUTOFF) / (RSI_IDEAL_LOW - RSI_OS_CUTOFF)
        else:
            score = 1.0 - (rsi - RSI_IDEAL_HIGH) / (RSI_OB_CUTOFF - RSI_IDEAL_HIGH)

        scores[ticker] = max(0.0, min(1.0, score))

    return pd.Series(scores, name="rsi_score")


def _score_momentum(df_universe: dict[str, pd.DataFrame]) -> pd.Series:
    """
    Momentum score: cross-sectional rank of trailing momentum values.
    Rank-normalises so extreme outliers don't dominate.
    """
    last_mom: dict[str, float] = {}
    for ticker, df in df_universe.items():
        mom_vals = df["momentum"].dropna()
        last_mom[ticker] = float(mom_vals.iloc[-1]) if not mom_vals.empty else np.nan

    s = pd.Series(last_mom)
    ranked = s.rank(pct=True, na_option="bottom")
    return ranked.rename("momentum_score")


def _score_volume(df_universe: dict[str, pd.DataFrame]) -> pd.Series:
    """
    Volume score: reward rising relative volume (volume_ratio > 1.0).
    Uses a smooth sigmoid-like transform centred at ratio = 1.0.
    """
    scores: dict[str, float] = {}
    for ticker, df in df_universe.items():
        vr_vals = df["volume_ratio"].dropna()
        if vr_vals.empty:
            scores[ticker] = 0.5
            continue
        vr = float(vr_vals.iloc[-1])
        # logistic centred at 1.0, scale 2 -> maps [0.5, 1.5] to roughly [0.27, 0.73]
        score = 1.0 / (1.0 + np.exp(-2.0 * (vr - 1.0)))
        scores[ticker] = max(0.0, min(1.0, score))

    return pd.Series(scores, name="volume_score")


def _score_trend(df_universe: dict[str, pd.DataFrame]) -> pd.Series:
    """
    Trend score: combines above_sma50, above_sma200, and golden_cross flags.
    above_sma50 = 0.4, above_sma200 = 0.4, golden_cross = 0.2.
    """
    scores: dict[str, float] = {}
    for ticker, df in df_universe.items():
        score = 0.0
        last = df.iloc[-1]

        for col, weight in [("above_sma50", 0.4), ("above_sma200", 0.4), ("golden_cross", 0.2)]:
            val = last.get(col, np.nan)
            if pd.notna(val):
                score += float(val) * weight

        scores[ticker] = max(0.0, min(1.0, score))

    return pd.Series(scores, name="trend_score")


def _score_volatility(df_universe: dict[str, pd.DataFrame]) -> pd.Series:
    """
    Volatility score: prefer lower ATR-normalised volatility (tighter risk).
    ATR/close gives a normalised figure; rank inverted so low vol -> high score.
    """
    atr_norm: dict[str, float] = {}
    for ticker, df in df_universe.items():
        atr_vals = df["atr"].dropna()
        close_vals = df["close"].dropna()
        if atr_vals.empty or close_vals.empty:
            atr_norm[ticker] = np.nan
            continue
        atr   = float(atr_vals.iloc[-1])
        close = float(close_vals.iloc[-1])
        atr_norm[ticker] = atr / close if close > 0 else np.nan

    s = pd.Series(atr_norm)
    # Invert rank: low normalised ATR gets high score
    ranked = s.rank(pct=True, na_option="bottom")
    inverted = 1.0 - ranked
    return inverted.rename("volatility_score")


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def score_universe(
    df_universe: dict[str, pd.DataFrame],
    top_n: int = DEFAULT_TOP_N,
    weights: Optional[dict[str, float]] = None,
    ml_scorer: Optional[object] = None,
    ml_weight: float = 0.0,
) -> ScoringResult:
    """
    Score and rank a validated universe dict returned by validate_universe().

    Parameters
    ----------
    df_universe : dict[ticker -> DataFrame]
        Each DataFrame must already have features from add_features().
    top_n : int
        Number of tickers to surface in ScoringResult.top().
    weights : dict, optional
        Override default WEIGHTS. Must contain the same keys and sum to 1.0.
    ml_scorer : object, optional
        Anything exposing predict_proba(valid_universe) -> Series[ticker->prob]
        (see ml.model.MLScorer). When supplied with ml_weight > 0 the model
        probability blends into the composite.
    ml_weight : float
        Blend weight on the ML probability in [0, 1]. 0.0 = pure rule score.

    Returns
    -------
    ScoringResult
        .ranked  - full sorted DataFrame (composite is the ranking column;
                   composite_rule and ml_prob added when ML is active)
        .top(n)  - top-n slice
    """
    w = weights if weights is not None else WEIGHTS

    if not df_universe:
        logger.warning("score_universe called with empty universe")
        return ScoringResult(
            ranked=pd.DataFrame(),
            score_date=pd.Timestamp.now(tz="UTC"),
        )

    excluded: dict[str, str] = {}
    valid_universe: dict[str, pd.DataFrame] = {}
    required_cols = {"rsi", "momentum", "volume_ratio", "above_sma50", "above_sma200", "golden_cross", "atr", "close"}

    for ticker, df in df_universe.items():
        if df is None or df.empty:
            excluded[ticker] = "empty DataFrame"
            continue
        missing = required_cols - set(df.columns)
        if missing:
            excluded[ticker] = f"missing feature columns: {missing}"
            continue
        valid_universe[ticker] = df

    if excluded:
        logger.warning("Excluded %d tickers from scoring: %s", len(excluded), list(excluded.keys()))

    if not valid_universe:
        logger.error("No valid tickers remain after pre-scoring filter")
        return ScoringResult(
            ranked=pd.DataFrame(),
            excluded=excluded,
            score_date=pd.Timestamp.now(tz="UTC"),
        )

    rsi_s   = _score_rsi(valid_universe)
    mom_s   = _score_momentum(valid_universe)
    vol_s   = _score_volume(valid_universe)
    trend_s = _score_trend(valid_universe)
    vola_s  = _score_volatility(valid_universe)

    scores = pd.DataFrame({
        "rsi_score":        rsi_s,
        "momentum_score":   mom_s,
        "volume_score":     vol_s,
        "trend_score":      trend_s,
        "volatility_score": vola_s,
    }).fillna(0.0)

    scores["composite"] = (
        scores["rsi_score"]        * w["rsi_score"]
        + scores["momentum_score"] * w["momentum_score"]
        + scores["volume_score"]   * w["volume_score"]
        + scores["trend_score"]    * w["trend_score"]
        + scores["volatility_score"] * w["volatility_score"]
    )

    # ----- Phase 7 ML blend (optional, opt-in) -----------------------------
    # Blends model probability into the composite. Does not replace the rule
    # score, does not gate entries. Tickers with no ML probability keep their
    # pure rule composite, so a partial-coverage model never penalises them.
    if ml_scorer is not None and ml_weight and float(ml_weight) > 0.0:
        mw = float(min(max(ml_weight, 0.0), 1.0))
        try:
            probs = ml_scorer.predict_proba(valid_universe)
        except Exception as exc:
            logger.warning("ML scoring failed, using rule composite only: %s", exc)
            probs = pd.Series(dtype="float64")

        aligned = pd.to_numeric(probs, errors="coerce").reindex(scores.index)
        scores["composite_rule"] = scores["composite"]
        scores["ml_prob"] = aligned
        blended = (1.0 - mw) * scores["composite_rule"] + mw * aligned
        scores["composite"] = blended.where(aligned.notna(), scores["composite_rule"])
        logger.info(
            "ML blend active: weight=%.2f, covered=%d/%d tickers",
            mw, int(aligned.notna().sum()), len(scores),
        )
    # -----------------------------------------------------------------------

    scores.index.name = "ticker"
    ranked = (
        scores
        .reset_index()
        .rename(columns={"index": "ticker"})
        .sort_values("composite", ascending=False)
        .reset_index(drop=True)
    )

    logger.info(
        "Scoring complete - %d tickers scored, top: %s (%.4f)",
        len(ranked),
        ranked.iloc[0]["ticker"] if not ranked.empty else "N/A",
        ranked.iloc[0]["composite"] if not ranked.empty else 0.0,
    )

    return ScoringResult(
        ranked=ranked,
        excluded=excluded,
        score_date=pd.Timestamp.now(tz="UTC"),
    )


def get_top_tickers(
    df_universe: dict[str, pd.DataFrame],
    top_n: int = DEFAULT_TOP_N,
    ml_scorer: Optional[object] = None,
    ml_weight: float = 0.0,
) -> list[str]:
    """
    Convenience wrapper - returns list of top-n ticker strings.
    Used by scheduler entry point. ML args forwarded to score_universe.
    """
    result = score_universe(df_universe, top_n=top_n, ml_scorer=ml_scorer, ml_weight=ml_weight)
    if result.ranked.empty or "ticker" not in result.ranked.columns:
        return []
    return result.ranked.head(top_n)["ticker"].tolist()
