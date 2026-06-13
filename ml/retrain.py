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
) -> RetrainResult:
    """
    Build the pooled matrix, time order it, train on the early slice and
    evaluate out of sample on the tail. An embargo gap (default = horizon)
    is dropped between train and test so forward looking labels in the train
    set do not leak into the holdout window.
    """
    matrix, skipped = build_training_matrix(df_universe, horizon, fee_bps)

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
    metrics["n_train"] = int(len(train))
    metrics["horizon"] = int(horizon)
    metrics["fee_bps"] = float(fee_bps)

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
