"""Evaluation metrics for imbalanced fraud classification (training/evaluation only)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def fbeta_optimal_threshold(y_true: np.ndarray, proba: np.ndarray, beta: float = 2.0) -> float:
    """Threshold maximizing F-beta (beta=2 weighs recall 2x: a missed fraud costs more
    than a false alarm)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        fbeta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    return float(thresholds[int(np.nanargmax(fbeta))])


def alert_rate_threshold(proba: np.ndarray, alert_rate: float) -> float:
    """Threshold flagging (approximately) the top ``alert_rate`` share of the traffic."""
    return float(np.quantile(proba, 1.0 - alert_rate))


def classification_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    pred = proba >= threshold
    tp = int(np.sum(pred & (y_true == 1)))
    fp = int(np.sum(pred & (y_true == 0)))
    fn = int(np.sum(~pred & (y_true == 1)))
    tn = int(np.sum(~pred & (y_true == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    fpr, tpr, _ = roc_curve(y_true, proba)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-12, 1 - 1e-12))),
        "brier": float(brier_score_loss(y_true, proba)),
        "recall_at_1pct_fpr": float(np.interp(0.01, fpr, tpr)),
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "alert_rate": float(pred.mean()),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "n_samples": len(y_true),
        "positive_rate": float(y_true.mean()),
    }
