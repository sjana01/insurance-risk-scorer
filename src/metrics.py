"""
Evaluation metrics for the Porto Seguro competition.

Primary metric: Normalized Gini Coefficient
  Gini = 2 * AUC - 1
  Range: [-1, 1]; random classifier ≈ 0, perfect classifier = 1.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def normalized_gini(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Normalized Gini coefficient (= 2 * ROC-AUC - 1)."""
    return 2.0 * roc_auc_score(y_true, y_pred) - 1.0


def gini_xgb_eval(y_pred: np.ndarray, dtrain) -> tuple:
    """XGBoost custom eval metric. Lower is better convention — we negate."""
    y_true = dtrain.get_label()
    gini = normalized_gini(y_true, y_pred)
    return "gini", -gini


def print_cv_summary(fold_scores: list) -> None:
    scores = np.array(fold_scores)
    print(f"CV Normalized Gini: {scores.mean():.4f} ± {scores.std():.4f}")
    for i, s in enumerate(scores, 1):
        print(f"  Fold {i}: {s:.4f}")
