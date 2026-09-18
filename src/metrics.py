"""
metrics.py

From-scratch evaluation metrics -- no sklearn.metrics used.

Includes:
- confusion matrix / precision / recall / F1 at a given threshold
- ROC curve + AUC (trapezoidal rule)
- Precision-Recall curve + Average Precision
- A simple cost-weighted decision curve, to make the FN-vs-FP
  clinical tradeoff concrete rather than abstract.
"""

from __future__ import annotations
import numpy as np


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    c = confusion_counts(y_true, y_pred)
    precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    specificity = c["tn"] / (c["tn"] + c["fp"]) if (c["tn"] + c["fp"]) > 0 else 0.0
    return {
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        **c,
    }


def roc_curve(y_true: np.ndarray, y_scores: np.ndarray):
    """
    Compute ROC curve points (fpr, tpr) by sweeping every unique score
    as a threshold. Returns (fpr_array, tpr_array, thresholds_array).
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    thresholds = np.unique(y_scores)[::-1]
    thresholds = np.concatenate(([np.inf], thresholds, [-np.inf]))

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    tpr_list = []
    fpr_list = []
    for t in thresholds:
        pred = (y_scores >= t).astype(int)
        tp = np.sum((y_true == 1) & (pred == 1))
        fp = np.sum((y_true == 0) & (pred == 1))
        tpr_list.append(tp / P if P > 0 else 0.0)
        fpr_list.append(fp / N if N > 0 else 0.0)

    return np.array(fpr_list), np.array(tpr_list), thresholds


def auc_trapezoidal(x: np.ndarray, y: np.ndarray) -> float:
    """Area under curve via the trapezoidal rule. x must be sorted ascending."""
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    return float(np.trapezoid(y_sorted, x_sorted))


def roc_auc_score(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return auc_trapezoidal(fpr, tpr)


def precision_recall_curve(y_true: np.ndarray, y_scores: np.ndarray):
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    thresholds = np.unique(y_scores)[::-1]

    precisions, recalls = [], []
    P = np.sum(y_true == 1)
    for t in thresholds:
        pred = (y_scores >= t).astype(int)
        tp = np.sum((y_true == 1) & (pred == 1))
        fp = np.sum((y_true == 0) & (pred == 1))
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
        recalls.append(tp / P if P > 0 else 0.0)
    return np.array(precisions), np.array(recalls), thresholds


def average_precision(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)
    order = np.argsort(recalls)
    r, p = recalls[order], precisions[order]
    return float(np.trapezoid(p, r))


def decision_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    fn_cost: float,
    fp_cost: float,
    thresholds: np.ndarray | None = None,
) -> dict:
    """
    Cost-weighted decision curve: for each threshold, compute total
    "clinical cost" = fn_cost * (# false negatives) + fp_cost * (# false positives).

    This operationalizes the FN-vs-FP discussion: instead of asserting
    FNs are "worse", we assign each an explicit relative cost and find the
    threshold that minimizes total expected cost -- which is the standard
    way clinical decision-support tools justify a chosen operating threshold.

    Returns the threshold with minimum total cost, plus the full sweep.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    costs = []
    for t in thresholds:
        pred = (y_scores >= t).astype(int)
        c = confusion_counts(y_true, pred)
        total_cost = fn_cost * c["fn"] + fp_cost * c["fp"]
        costs.append(total_cost)

    costs = np.array(costs)
    best_idx = int(np.argmin(costs))
    return {
        "thresholds": thresholds,
        "costs": costs,
        "best_threshold": float(thresholds[best_idx]),
        "best_cost": float(costs[best_idx]),
        "cost_at_0.5": float(costs[np.argmin(np.abs(thresholds - 0.5))]),
    }
