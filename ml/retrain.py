"""
ml/retrain.py
Rolling walk forward retrain for the weekly stock model. Wired behind the
scheduler's sunday_ml_retrain hook (Phase 7 replaces the stub there).

DB writes are kept out of this package. The caller passes an optional
performance_writer callable (the scheduler supplies the ModelPerformance
adapter), so this module stays testable with no DB dependency and no schema
assumptions baked in.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd

from ml.features import (
    DEFAULT_FEE_BPS,
    DEFAULT_HORIZON,
    FEATURE_COLS,
    build_training_matrix,
)
from ml.model import TrainedModel, evaluate_model, train_model

logger = logging.getLogger(__name__)


def purged_walk_forward(
    matrix: pd.DataFrame,
    n_folds: int = 4,
    embargo: int = DEFAULT_HORIZON,
    params: Optional[dict] = None,
) -> dict:
    """Expanding-window walk-forward with a purge/embargo gap between each
    train slice and its test slice.

    Why this and not one holdout: weekly-return labels OVERLAP in time, so a
    single split reports one noisy number that can look good by luck. What
    matters in this domain is STABILITY - a model that holds ~0.52-0.55 AUC
    across several out-of-sample windows has a small real edge; one that
    swings 0.42/0.61 has none. Each fold trains on everything before its
    window, minus an embargo so forward-looking labels cannot leak across
    the boundary.

    Returns {"fold_aucs": [...], "auc_mean": x, "auc_std": y, "n_folds": n}.
    """
    out: dict = {"fold_aucs": [], "auc_mean": float("nan"),
                 "auc_std": float("nan"), "n_folds": 0}
    n = len(matrix)
    if n < 200 or n_folds < 2:
        return out

    fold_size = n // (n_folds + 1)          # first block is train-only
    if fold_size <= embargo + 10:
        return out

    aucs: list[float] = []
    for k in range(1, n_folds + 1):
        test_start = fold_size * k
        test_end = fold_size * (k + 1) if k < n_folds else n
        train_end = max(test_start - embargo, 1)
        train = matrix.iloc[:train_end]
        test = matrix.iloc[test_start:test_end]
        if len(test) < 20 or train["label"].nunique() < 2 or test["label"].nunique() < 2:
            continue
        try:
            model = train_model(train[FEATURE_COLS], train["label"], params)
            auc = evaluate_model(model, test[FEATURE_COLS], test["label"]).get("auc")
        except Exception as exc:
            logger.warning("walk-forward fold %d failed: %s", k, exc)
            continue
        if auc is not None and not pd.isna(auc):
            aucs.append(float(auc))

    if aucs:
        arr = pd.Series(aucs)
        out["fold_aucs"] = [round(a, 4) for a in aucs]
        out["auc_mean"] = float(arr.mean())
        out["auc_std"] = float(arr.std(ddof=0))
        out["n_folds"] = len(aucs)
    return out


@dataclass
class RetrainResult:
    ok: bool
    reason: str = "ok"
    model: Optional[TrainedModel] = None
    metrics: dict = field(default_factory=dict)
    model_path: Optional[str] = None
    n_train: int = 0
    n_test: int = 0
    skipped: dict = field(default_factory=dict)


def run_rolling_retrain(
    df_universe: dict[str, pd.DataFrame],
    *,
    horizon: int = DEFAULT_HORIZON,
    fee_bps: float = DEFAULT_FEE_BPS,
    holdout_frac: float = 0.2,
    embargo: Optional[int] = None,
    model_path: Optional[str] = None,
    params: Optional[dict] = None,
    performance_writer: Optional[Callable[[dict], None]] = None,
    label_mode: str = "relative",
    walk_forward_folds: int = 4,
) -> RetrainResult:
    """
    Build the pooled matrix, time order it, train on the early slice and
    evaluate out of sample on the tail. An embargo gap (default = horizon)
    is dropped between train and test so forward looking labels in the train
    set do not leak into the holdout window (a purged walk-forward split).

    label_mode "relative" (default) trains against cross-sectional
    outperformance - the right target for a rotation strategy; "absolute"
    keeps the legacy own-return label.
    """
    matrix, skipped = build_training_matrix(df_universe, horizon, fee_bps,
                                            label_mode=label_mode)

    if matrix.empty:
        logger.warning("retrain aborted: empty training matrix")
        return RetrainResult(ok=False, reason="insufficient_data", skipped=skipped)
    if matrix["label"].nunique() < 2:
        logger.warning("retrain aborted: single class labels")
        return RetrainResult(ok=False, reason="single_class", skipped=skipped)

    matrix = matrix.sort_values("date").reset_index(drop=True)
    emb = horizon if embargo is None else int(embargo)

    n = len(matrix)
    n_holdout = max(int(n * holdout_frac), 1)
    split = n - n_holdout
    train_end = max(split - emb, 1)

    train = matrix.iloc[:train_end]
    test = matrix.iloc[split:]

    if train["label"].nunique() < 2 or len(test) == 0:
        logger.warning("retrain aborted: split left a class empty or no test rows")
        return RetrainResult(ok=False, reason="bad_split", skipped=skipped,
                             n_train=len(train), n_test=len(test))

    model = train_model(train[FEATURE_COLS], train["label"], params)
    metrics = evaluate_model(model, test[FEATURE_COLS], test["label"])

    # Multi-fold stability check. The single holdout AUC above is one draw;
    # these folds say whether it is repeatable. auc_mean drives the quality
    # gate when available, so one lucky window cannot promote a model.
    if walk_forward_folds and walk_forward_folds >= 2:
        wf = purged_walk_forward(matrix, n_folds=walk_forward_folds,
                                 embargo=emb, params=params)
        metrics.update(wf)
        if wf["n_folds"]:
            logger.info("walk-forward: folds=%s mean=%.4f std=%.4f",
                        wf["fold_aucs"], wf["auc_mean"], wf["auc_std"])

    metrics["n_train"] = int(len(train))
    metrics["horizon"] = int(horizon)
    metrics["fee_bps"] = float(fee_bps)
    metrics["label_mode"] = label_mode

    if model_path:
        model.save(model_path)

    if performance_writer is not None:
        try:
            performance_writer(metrics)
        except Exception as exc:  # adapter failure must not kill the job
            logger.error("performance_writer failed: %s", exc)

    logger.info(
        "retrain ok: train=%d test=%d auc=%.4f acc=%.4f base=%.4f",
        len(train), len(test),
        metrics.get("auc", float("nan")),
        metrics.get("accuracy", float("nan")),
        metrics.get("base_rate", float("nan")),
    )
    return RetrainResult(
        ok=True,
        model=model,
        metrics=metrics,
        model_path=model_path,
        n_train=int(len(train)),
        n_test=int(len(test)),
        skipped=skipped,
    )
