from __future__ import annotations

from typing import Dict

import numpy as np


def binary_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    y = y_true.reshape(-1).astype(int)
    p = np.clip(prob.reshape(-1), 1e-7, 1.0 - 1e-7)
    pred = (p >= 0.5).astype(int)
    acc = float((pred == y).mean())
    bce = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    out = {"accuracy": acc, "bce": bce}
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        if len(np.unique(y)) > 1:
            out["auroc"] = float(roc_auc_score(y, p))
            out["auprc"] = float(average_precision_score(y, p))
        else:
            out["auroc"] = float("nan")
            out["auprc"] = float("nan")
    except Exception:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def ridge_r2(z_hat: np.ndarray, z_true: np.ndarray, alpha: float = 1e-3) -> float:
    X = np.asarray(z_hat, dtype=np.float64)
    Y = np.asarray(z_true, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError(f"Expected two 2D arrays with the same sample count, got {X.shape} and {Y.shape}")
    if Y.shape[0] < 2:
        return float("nan")
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    XtX = X.T @ X + alpha * np.eye(X.shape[1])
    W = np.linalg.solve(XtX, X.T @ Y)
    Yp = X @ W
    ss_res = ((Y - Yp) ** 2).sum()
    ss_tot = ((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum() + 1e-12
    return float(1.0 - ss_res / ss_tot)


def mean_corrcoef_matching(z_hat: np.ndarray, z_true: np.ndarray) -> float:
    X = np.asarray(z_hat, dtype=np.float64)
    Y = np.asarray(z_true, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0]:
        raise ValueError(f"Expected two 2D arrays with the same sample count, got {X.shape} and {Y.shape}")
    if X.shape[0] < 2:
        return float("nan")
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    Y = (Y - Y.mean(axis=0, keepdims=True)) / (Y.std(axis=0, keepdims=True) + 1e-8)
    C = np.clip(np.abs((X.T @ Y) / X.shape[0]), 0.0, 1.0)
    try:
        from scipy.optimize import linear_sum_assignment
        row, col = linear_sum_assignment(-C)
        return float(C[row, col].mean())
    except Exception:
        # Greedy fallback.
        used_r, used_c, vals = set(), set(), []
        for _ in range(min(C.shape)):
            best = None
            for i in range(C.shape[0]):
                if i in used_r:
                    continue
                for j in range(C.shape[1]):
                    if j in used_c:
                        continue
                    if best is None or C[i, j] > best[0]:
                        best = (C[i, j], i, j)
            if best is None:
                break
            vals.append(best[0])
            used_r.add(best[1]); used_c.add(best[2])
        return float(np.mean(vals)) if vals else float("nan")


def footprint_metrics(gate: np.ndarray, true_fp: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (gate >= threshold).astype(int)
    true = (true_fp >= 0.5).astype(int)
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    acc = (tp + tn) / max(1, tp + fp + fn + tn)
    return {
        "gate_precision": float(precision),
        "gate_recall": float(recall),
        "gate_f1": float(f1),
        "gate_accuracy": float(acc),
        "gate_active": float(pred.sum()),
    }


def directed_graph_metrics(weight: np.ndarray, true_graph: np.ndarray, threshold: float = 0.05) -> Dict[str, float]:
    pred = (np.abs(weight) >= threshold).astype(int)
    true = (np.abs(true_graph) > 1e-8).astype(int)
    offdiag = 1 - np.eye(true.shape[0], dtype=int)
    pred = pred * offdiag
    true = true * offdiag
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    acc = (tp + tn) / max(1, tp + fp + fn + tn)
    true_edge_mask = true.astype(bool)
    if true_edge_mask.any():
        sign_acc = float((np.sign(weight[true_edge_mask]) == np.sign(true_graph[true_edge_mask])).mean())
    else:
        sign_acc = float("nan")
    return {
        "graph_precision": float(precision),
        "graph_recall": float(recall),
        "graph_f1": float(f1),
        "graph_accuracy": float(acc),
        "graph_active": float(pred.sum()),
        "graph_sign_accuracy": sign_acc,
    }


def linear_probe_r2(x: np.ndarray, target: np.ndarray) -> float:
    return ridge_r2(x, target)
