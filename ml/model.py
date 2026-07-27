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
import os
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

# --- GPU acceleration -------------------------------------------------------
# The retrain now pools ~3 years x ~500 tickers and (with walk-forward) fits
# several models per run. On a CUDA box that is minutes of CPU time or
# seconds of GPU time, which is what makes multi-fold validation practical.
# Detection is a real tiny fit, cached: capability probes lie less than
# version strings, and any failure silently falls back to CPU.
_GPU_STATE: Optional[bool] = None


def gpu_available() -> bool:
    """True when XGBoost can actually train on CUDA here. Cached."""
    global _GPU_STATE
    if _GPU_STATE is not None:
        return _GPU_STATE
    if not _HAS_XGB:
        _GPU_STATE = False
        return False
    if str(os.environ.get("ML_FORCE_CPU", "")).lower() in {"1", "true", "yes"}:
        logger.info("ML_FORCE_CPU set; training on CPU")
        _GPU_STATE = False
        return False
    # XGBoost does NOT raise when CUDA is missing - it emits a C++ level
    # "Device is changed from GPU to CPU" notice and quietly trains on CPU,
    # so a probe-fit always "succeeds" and cannot be used for detection.
    # Ask the driver instead: nvidia-smi exits 0 only with a usable GPU.
    try:
        import shutil
        import subprocess

        if shutil.which("nvidia-smi") is None:
            _GPU_STATE = False
            logger.info("No nvidia-smi on PATH; training on CPU")
            return _GPU_STATE
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        name = (proc.stdout or "").strip().splitlines()
        if proc.returncode == 0 and name:
            _GPU_STATE = True
            logger.info("CUDA device detected ({}); training on GPU", name[0])
        else:
            _GPU_STATE = False
            logger.info("nvidia-smi reported no GPU; training on CPU")
    except Exception as exc:
        _GPU_STATE = False
        logger.info("GPU detection failed (%s); training on CPU", type(exc).__name__)
    return _GPU_STATE


def _apply_device(params: dict) -> dict:
    """Attach device/tree_method unless the caller pinned them."""
    if "device" in params:
        return params
    if gpu_available():
        params["device"] = "cuda"
        params.setdefault("tree_method", "hist")
    return params


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

    p = _apply_device(p)
    clf = XGBClassifier(**p)
    try:
        clf.fit(X[FEATURE_COLS].astype("float32").to_numpy(), y.to_numpy())
    except Exception as exc:
        if p.get("device") != "cuda":
            raise
        # A GPU that probes fine can still fail on a real matrix (OOM, driver
        # mismatch). Never let that kill the weekly retrain - drop to CPU.
        global _GPU_STATE
        _GPU_STATE = False
        logger.warning("GPU training failed (%s); retrying on CPU", exc)
        p.pop("device", None)
        clf = XGBClassifier(**p)
        clf.fit(X[FEATURE_COLS].astype("float32").to_numpy(), y.to_numpy())
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
