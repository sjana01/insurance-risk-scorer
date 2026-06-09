"""Tests for src/metrics.py"""

import numpy as np
import pytest

from src.metrics import normalized_gini, gini_xgb_eval


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_perfect_predictor():
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    gini = normalized_gini(y_true, y_pred)
    assert abs(gini - 1.0) < 1e-9


def test_perfect_inverse_predictor():
    """Perfectly wrong predictor should give Gini = -1."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.9, 0.8, 0.2, 0.1])
    gini = normalized_gini(y_true, y_pred)
    assert abs(gini - (-1.0)) < 1e-9


def test_random_predictor_near_zero():
    """Random scores should give Gini near 0 (probabilistically)."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=10_000)
    y_pred = rng.random(size=10_000)
    gini = normalized_gini(y_true, y_pred)
    assert abs(gini) < 0.05, f"Random predictor Gini should be near 0, got {gini:.4f}"


def test_known_value():
    """AUC = 0.75 → Gini = 0.5."""
    # Construct y_true / y_pred so AUC = 0.75 exactly
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    # AUC for this = (1 + 1 + 0 + 1) / 4 ... let's just check the formula
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_pred)
    expected_gini = 2 * auc - 1
    gini = normalized_gini(y_true, y_pred)
    assert abs(gini - expected_gini) < 1e-9


def test_range_is_minus_one_to_one():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=500)
    y_pred = rng.random(500)
    gini = normalized_gini(y_true, y_pred)
    assert -1.0 <= gini <= 1.0


def test_gini_xgb_eval_returns_tuple():
    """gini_xgb_eval must return (name: str, value: float) tuple."""

    class FakeDMatrix:
        def get_label(self):
            return np.array([0, 0, 1, 1], dtype=float)

    y_pred = np.array([0.1, 0.2, 0.8, 0.9])
    result = gini_xgb_eval(y_pred, FakeDMatrix())
    assert isinstance(result, tuple)
    assert len(result) == 2
    name, value = result
    assert name == "gini"
    assert isinstance(value, float)
    # Negated because XGBoost minimises — perfect predictor → -1.0
    assert value == pytest.approx(-1.0, abs=1e-9)


def test_gini_xgb_eval_negated():
    """XGBoost eval metric is negated so lower = better."""

    class FakeDMatrix:
        def get_label(self):
            return np.array([0, 1], dtype=float)

    # score > 0.5 for positive class → positive gini → negated value < 0
    _, value = gini_xgb_eval(np.array([0.3, 0.7]), FakeDMatrix())
    assert value < 0
