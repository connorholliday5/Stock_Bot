"""
ml/model.py
XGBoost classifier wrapper for the weekly stock model, plus MLScorer, the
adapter the rule scorer uses to blend an ML probability into the composite.

The model predicts P(forward return positive net of fees). It blends into the
rule composite. It does not replace the rule scorer and does not gate entries.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.features import FEATURE_COLS, build_inference_matrix

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover - environment guard
    XGBClassifier = None
    _HAS_XGB = False


DEFAULT_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "min_child_weight": 5,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_jobs": 2,
    "random_state": 42,
}


def _roc_auc(y_true, scores) -> float:
    """Rank based AUC (Mann Whitney), tie safe. NaN if one class only."""
    y_true = np.asarray(y_true, dtype=float)
    ranks = pd.Series(np.asarray(scores, dtype=float)).rank(method="average").to_numpy()
    n_pos = float((y_true == 1).sum())
    n_neg = float((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[y_true == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@dataclass
class TrainedModel:
    model: object
    feature_cols: list = field(default_factory=lambda: list(FEATURE_COLS))
    params: dict = field(default_factory=dict)
    trained_rows: int = 0

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """P(positive) per row, indexed like X. Empty in -> empty out."""
        if X is None or X.empty:
            return pd.Series(dtype="float64", name="ml_prob")
        Xm = X[self.feature_cols].astype("float64")
        proba = self.model.predict_proba(Xm.to_numpy())[:, 1]
        return pd.Series(proba, index=X.index, name="ml_prob")

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        booster_path = path.with_suffix(".json")
        meta_path = path.with_suffix(".meta.json")
        self.model.save_model(str(booster_path))
        meta = {
            "feature_cols": self.feature_cols,
            "params": self.params,
            "trained_rows": self.trained_rows,
        }
        meta_path.write_text(json.dumps(meta))
        logger.info("Saved model booster=%s meta=%s", booster_path, meta_path)

    @classmethod
    def load(cls, path) -> "TrainedModel":
        if not _HAS_XGB:
            raise RuntimeError("xgboost not installed; cannot load model")
        path = Path(path)
        booster_path = path.with_suffix(".json")
        meta_path = path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        clf = XGBClassifier()
        clf.load_model(str(booster_path))
        return cls(
            model=clf,
            feature_cols=meta.get("feature_cols", list(FEATURE_COLS)),
            params=meta.get("params", {}),
            trained_rows=int(meta.get("trained_rows", 0)),
        )


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: Optional[dict] = None,
) -> TrainedModel:
    """Fit an XGBoost binary classifier on the pooled feature matrix."""
    if not _HAS_XGB:
        raise RuntimeError("xgboost not installed; cannot train model")

    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    y = pd.Series(y).astype(int)
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos > 0:
        p.setdefault("scale_pos_weight", max(neg / pos, 1e-3))

    clf = XGBClassifier(**p)
    clf.fit(X[FEATURE_COLS].astype("float64").to_numpy(), y.to_numpy())
    return TrainedModel(
        model=clf,
        feature_cols=list(FEATURE_COLS),
        params=p,
        trained_rows=int(len(y)),
    )


def evaluate_model(model: TrainedModel, X: pd.DataFrame, y: pd.Series) -> dict:
    """Out of sample metrics: accuracy, auc, logloss, base_rate, n."""
    proba = model.predict_proba(X)
    if proba.empty:
        return {"n": 0, "accuracy": float("nan"), "auc": float("nan"),
                "logloss": float("nan"), "base_rate": float("nan")}

    yv = pd.Series(y).astype(int).to_numpy()
    pv = proba.to_numpy()
    pred = (pv >= 0.5).astype(int)

    acc = float((pred == yv).mean())
    auc = _roc_auc(yv, pv)
    eps = 1e-7
    pr = np.clip(pv, eps, 1.0 - eps)
    logloss = float(-np.mean(yv * np.log(pr) + (1 - yv) * np.log(1 - pr)))
    base_rate = float(yv.mean())
    return {
        "n": int(len(yv)),
        "accuracy": acc,
        "auc": auc,
        "logloss": logloss,
        "base_rate": base_rate,
    }


class MLScorer:
    """
    Wraps a TrainedModel and the feature assembly. Produces a ticker -> prob
    Series for a validated universe dict, for the rule scorer to blend in.
    """

    def __init__(self, model: TrainedModel):
        self.model = model

    @classmethod
    def from_path(cls, path) -> "MLScorer":
        return cls(TrainedModel.load(path))

    def predict_proba(self, df_universe: dict[str, pd.DataFrame]) -> pd.Series:
        X, skipped = build_inference_matrix(df_universe)
        if skipped:
            logger.info("MLScorer skipped %d tickers: %s",
                        len(skipped), list(skipped.keys()))
        if X.empty:
            return pd.Series(dtype="float64", name="ml_prob")
        return self.model.predict_proba(X)
