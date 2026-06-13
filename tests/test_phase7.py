"""
tests/test_phase7.py
Phase 7 ML layer: labels, features, model train/eval/persist, MLScorer,
scorer ML blend, rolling retrain. No network, no DB, fully synthetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.labels import DEFAULT_FEE_BPS, forward_return, make_labels
from ml.features import (
    FEATURE_COLS,
    REQUIRED_COLS,
    build_inference_matrix,
    build_training_matrix,
)
from ml.model import (
    MLScorer,
    TrainedModel,
    evaluate_model,
    train_model,
    _roc_auc,
)
from ml.retrain import run_rolling_retrain
from strategies.stock_scorer import score_universe


# ---------------------------------------------------------------------------
# Synthetic data: a learnable forward-return signal driven by momentum + rsi
# ---------------------------------------------------------------------------

def _make_frame(seed: int, n: int = 320, signal: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n)

    # AR(1) latent driver in [-1, 1]: persistence makes today's features
    # informative about the forward window, so the label is learnable.
    driver = np.zeros(n)
    for t in range(1, n):
        driver[t] = 0.85 * driver[t - 1] + rng.normal(0, 0.35)
    driver = np.clip(driver, -1, 1)
    rets = signal * 0.01 * driver + rng.normal(0, 0.004, n)
    close = 100.0 * np.cumprod(1.0 + rets)

    rsi = np.clip(50 + 25 * driver + rng.normal(0, 5, n), 1, 99)
    momentum = pd.Series(close).pct_change(10).fillna(0.0).to_numpy()
    volume_ratio = np.clip(1.0 + 0.5 * driver + rng.normal(0, 0.2, n), 0.1, 3.0)
    atr = np.abs(close * (0.01 + 0.004 * rng.random(n)))

    sma50 = pd.Series(close).rolling(50, min_periods=1).mean().to_numpy()
    sma200 = pd.Series(close).rolling(200, min_periods=1).mean().to_numpy()

    return pd.DataFrame({
        "date": dates,
        "close": close,
        "rsi": rsi,
        "momentum": momentum,
        "volume_ratio": volume_ratio,
        "atr": atr,
        "above_sma50": (close > sma50).astype(float),
        "above_sma200": (close > sma200).astype(float),
        "golden_cross": (sma50 > sma200).astype(float),
    }).set_index("date")


def _make_universe(n_tickers: int = 12, **kw) -> dict[str, pd.DataFrame]:
    return {f"T{i:02d}": _make_frame(seed=100 + i, **kw) for i in range(n_tickers)}


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def test_forward_return_math():
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    fwd = forward_return(close, horizon=2)
    assert fwd.iloc[0] == pytest.approx(102.0 / 100.0 - 1.0)
    assert pd.isna(fwd.iloc[-1]) and pd.isna(fwd.iloc[-2])


def test_make_labels_fee_threshold_and_tail():
    # +5% then -5% moves around a flat tail
    close = pd.Series([100.0, 105.0, 105.0, 105.0, 99.0, 99.0])
    lab = make_labels(close, horizon=1, fee_bps=10.0)
    assert lab.iloc[0] == 1.0           # +5% clears 0.1% fee
    assert lab.iloc[3] == 0.0           # -5.7% fails
    assert pd.isna(lab.iloc[-1])        # no forward window
    assert set(lab.dropna().unique()).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def test_training_matrix_columns_and_no_nan():
    uni = _make_universe(6)
    matrix, skipped = build_training_matrix(uni, horizon=5, fee_bps=10.0)
    assert not matrix.empty
    assert skipped == {}
    for c in FEATURE_COLS + ["label", "ticker", "date"]:
        assert c in matrix.columns
    assert not matrix[FEATURE_COLS].isna().any().any()
    assert set(matrix["label"].unique()).issubset({0, 1})


def test_training_matrix_skips_missing_columns():
    uni = _make_universe(3)
    bad = uni["T00"].drop(columns=["atr"])
    uni["T00"] = bad
    matrix, skipped = build_training_matrix(uni)
    assert "T00" in skipped
    assert "T00" not in matrix["ticker"].unique()


def test_inference_matrix_one_row_per_ticker():
    uni = _make_universe(8)
    X, skipped = build_inference_matrix(uni)
    assert list(X.columns) == FEATURE_COLS
    assert len(X) == 8
    assert X.index.name == "ticker"
    assert not X.isna().any().any()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def test_auc_helper_perfect_and_degenerate():
    assert _roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    assert np.isnan(_roc_auc([1, 1, 1], [0.5, 0.6, 0.7]))


def test_train_predict_proba_range():
    matrix, _ = build_training_matrix(_make_universe(10), horizon=5)
    model = train_model(matrix[FEATURE_COLS], matrix["label"],
                        params={"n_estimators": 60})
    probs = model.predict_proba(matrix[FEATURE_COLS].head(20))
    assert len(probs) == 20
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    assert model.predict_proba(pd.DataFrame(columns=FEATURE_COLS)).empty


def test_model_learns_signal_out_of_sample():
    matrix, _ = build_training_matrix(_make_universe(14, signal=1.5), horizon=5)
    matrix = matrix.sort_values("date").reset_index(drop=True)
    cut = int(len(matrix) * 0.8)
    train, test = matrix.iloc[:cut], matrix.iloc[cut:]
    model = train_model(train[FEATURE_COLS], train["label"],
                        params={"n_estimators": 120})
    metrics = evaluate_model(model, test[FEATURE_COLS], test["label"])
    assert metrics["n"] == len(test)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["auc"] > 0.55     # beats coin flip on a real signal


def test_save_load_roundtrip(tmp_path):
    matrix, _ = build_training_matrix(_make_universe(8), horizon=5)
    model = train_model(matrix[FEATURE_COLS], matrix["label"],
                        params={"n_estimators": 50})
    X = matrix[FEATURE_COLS].head(15)
    before = model.predict_proba(X)

    path = tmp_path / "model" / "weekly"
    model.save(path)
    loaded = TrainedModel.load(path)
    after = loaded.predict_proba(X)

    assert loaded.feature_cols == FEATURE_COLS
    np.testing.assert_allclose(before.to_numpy(), after.to_numpy(), rtol=1e-5)


def test_mlscorer_returns_ticker_series():
    uni = _make_universe(9)
    matrix, _ = build_training_matrix(uni, horizon=5)
    model = train_model(matrix[FEATURE_COLS], matrix["label"],
                        params={"n_estimators": 50})
    scorer = MLScorer(model)
    probs = scorer.predict_proba(uni)
    assert len(probs) == 9
    assert set(probs.index) == set(uni.keys())
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


# ---------------------------------------------------------------------------
# Scorer ML blend
# ---------------------------------------------------------------------------

def test_scorer_unchanged_without_ml():
    uni = _make_universe(10)
    base = score_universe(uni).ranked
    # default args: no ml -> no ml columns, composite is pure rule score
    assert "ml_prob" not in base.columns
    assert "composite_rule" not in base.columns
    assert base["composite"].between(0.0, 1.0).all()


def test_scorer_blend_changes_composite_and_adds_columns():
    uni = _make_universe(12)
    matrix, _ = build_training_matrix(uni, horizon=5)
    model = train_model(matrix[FEATURE_COLS], matrix["label"],
                        params={"n_estimators": 60})
    scorer = MLScorer(model)

    rule = score_universe(uni).ranked.set_index("ticker")["composite"]
    blended_res = score_universe(uni, ml_scorer=scorer, ml_weight=0.4).ranked
    blended = blended_res.set_index("ticker")

    assert "ml_prob" in blended.columns
    assert "composite_rule" in blended.columns
    # rule component preserved
    np.testing.assert_allclose(
        blended["composite_rule"].sort_index().to_numpy(),
        rule.sort_index().to_numpy(), rtol=1e-9,
    )
    # blend actually moved at least one composite
    diff = (blended["composite"].sort_index() - rule.sort_index()).abs()
    assert diff.max() > 1e-6
    assert blended["composite"].between(0.0, 1.0).all()


def test_scorer_blend_zero_weight_is_noop():
    uni = _make_universe(6)
    matrix, _ = build_training_matrix(uni, horizon=5)
    model = train_model(matrix[FEATURE_COLS], matrix["label"],
                        params={"n_estimators": 40})
    scorer = MLScorer(model)
    a = score_universe(uni).ranked.set_index("ticker")["composite"]
    b = score_universe(uni, ml_scorer=scorer, ml_weight=0.0).ranked.set_index("ticker")["composite"]
    np.testing.assert_allclose(a.sort_index().to_numpy(), b.sort_index().to_numpy(), rtol=1e-12)


def test_scorer_blend_survives_failing_ml():
    class Boom:
        def predict_proba(self, _):
            raise RuntimeError("model exploded")

    uni = _make_universe(5)
    rule = score_universe(uni).ranked.set_index("ticker")["composite"]
    out = score_universe(uni, ml_scorer=Boom(), ml_weight=0.5).ranked.set_index("ticker")
    # falls back to rule composite for every ticker (ml_prob all NaN)
    np.testing.assert_allclose(
        out["composite"].sort_index().to_numpy(),
        rule.sort_index().to_numpy(), rtol=1e-9,
    )


# ---------------------------------------------------------------------------
# Rolling retrain
# ---------------------------------------------------------------------------

def test_retrain_ok_writes_performance_and_model(tmp_path):
    captured = {}

    def writer(metrics):
        captured.update(metrics)

    path = tmp_path / "models" / "weekly"
    res = run_rolling_retrain(
        _make_universe(14, signal=1.5),
        horizon=5,
        model_path=str(path),
        params={"n_estimators": 80},
        performance_writer=writer,
    )
    assert res.ok
    assert res.n_train > 0 and res.n_test > 0
    assert "auc" in res.metrics and "n_train" in res.metrics
    assert captured.get("auc") == res.metrics["auc"]
    assert (path.with_suffix(".json")).exists()
    assert (path.with_suffix(".meta.json")).exists()


def test_retrain_writer_failure_does_not_kill_job():
    def bad_writer(_):
        raise RuntimeError("db down")

    res = run_rolling_retrain(
        _make_universe(10),
        horizon=5,
        params={"n_estimators": 40},
        performance_writer=bad_writer,
    )
    assert res.ok  # adapter failure swallowed


def test_retrain_insufficient_data():
    res = run_rolling_retrain({}, horizon=5)
    assert not res.ok
    assert res.reason == "insufficient_data"


def test_retrain_single_class_aborts():
    # force constant close -> all labels 0 (no move clears the fee)
    uni = _make_universe(4)
    for t, df in uni.items():
        df = df.copy()
        df["close"] = 100.0
        df["atr"] = 1.0
        uni[t] = df
    res = run_rolling_retrain(uni, horizon=5)
    assert not res.ok
    assert res.reason in {"single_class", "bad_split"}
