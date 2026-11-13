"""Evaluation metrics for fraud-risk scores.

Fraud is rare (<1% of firm-years), so accuracy is meaningless. We report:
  AUC       - ranking quality over all thresholds (0.5 = random)
  AUPRC     - area under precision-recall; random baseline = fraud prevalence
  Top-k%    - precision / recall / NDCG when an analyst investigates only the
              top k% highest-risk firms (Bao et al. use k = 1%)
  F1 / FPR  - at a fixed decision threshold tau (for the LLM red-flag call)
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)


def ndcg_at_k(y: np.ndarray, s: np.ndarray, k: int) -> float:
    order = np.argsort(-s)[:k]
    gains = y[order] / np.log2(np.arange(2, k + 2))
    ideal = np.sort(y)[::-1][:k] / np.log2(np.arange(2, k + 2))
    return float(gains.sum() / ideal.sum()) if ideal.sum() > 0 else 0.0


def ranking_metrics(y, s, top_frac: float = 0.01) -> dict:
    y, s = np.asarray(y, int), np.asarray(s, float)
    k = max(1, int(round(top_frac * len(y))))
    flagged = np.zeros_like(y)
    flagged[np.argsort(-s)[:k]] = 1
    return {
        "n": len(y),
        "n_fraud": int(y.sum()),
        "prevalence": float(y.mean()),
        "AUC": float(roc_auc_score(y, s)),
        "AUPRC": float(average_precision_score(y, s)),
        f"precision@top{top_frac:.0%}": float(precision_score(y, flagged, zero_division=0)),
        f"recall@top{top_frac:.0%}": float(recall_score(y, flagged, zero_division=0)),
        f"NDCG@top{top_frac:.0%}": ndcg_at_k(y, s, k),
    }


def threshold_metrics(y, s, tau: float) -> dict:
    y, s = np.asarray(y, int), np.asarray(s, float)
    pred = (s >= tau).astype(int)
    neg = y == 0
    return {
        "tau": tau,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "FPR": float(pred[neg].mean()) if neg.any() else float("nan"),
        "score_sep": float(s[y == 1].mean() - s[neg].mean())
        if y.any() and neg.any() else float("nan"),
    }
