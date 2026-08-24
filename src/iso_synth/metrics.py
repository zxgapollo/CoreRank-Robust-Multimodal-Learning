from __future__ import annotations

from typing import Dict

import numpy as np


def binary_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    y = y_true.reshape(-1).astype(int)
    p = np.clip(prob.reshape(-1), 1e-7, 1.0 - 1e-7)
    pred = (p >= 0.5).astype(int)
    nll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    out = {
        "acc": float((pred == y).mean()),
        "nll": nll,
        "bce": nll,
    }
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        if len(np.unique(y)) > 1:
            out["auc"] = float(roc_auc_score(y, p))
            out["auprc"] = float(average_precision_score(y, p))
        else:
            out["auc"] = float("nan")
            out["auprc"] = float("nan")
    except Exception:
        out["auc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def ridge_r2(z_hat: np.ndarray, z_true: np.ndarray, alpha: float = 1e-3) -> float:
    X = np.asarray(z_hat, dtype=np.float64)
    Y = np.asarray(z_true, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0] or X.shape[0] < 2:
        return float("nan")
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    reg = alpha * np.eye(X.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    try:
        W = np.linalg.solve(X.T @ X + reg, X.T @ Y)
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(X.T @ X + reg) @ X.T @ Y
    pred = X @ W
    ss_res = float(((Y - pred) ** 2).sum())
    ss_tot = float(((Y - Y.mean(axis=0, keepdims=True)) ** 2).sum()) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def mean_corrcoef_matching(z_hat: np.ndarray, z_true: np.ndarray) -> float:
    X = np.asarray(z_hat, dtype=np.float64)
    Y = np.asarray(z_true, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0] or X.shape[0] < 2:
        return float("nan")
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    Y = (Y - Y.mean(axis=0, keepdims=True)) / (Y.std(axis=0, keepdims=True) + 1e-8)
    C = np.abs((X.T @ Y) / X.shape[0])
    try:
        from scipy.optimize import linear_sum_assignment

        row, col = linear_sum_assignment(-C)
        return float(C[row, col].mean())
    except Exception:
        vals = []
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        for _ in range(min(C.shape)):
            best = None
            for i in range(C.shape[0]):
                if i in used_rows:
                    continue
                for j in range(C.shape[1]):
                    if j in used_cols:
                        continue
                    if best is None or C[i, j] > best[0]:
                        best = (float(C[i, j]), i, j)
            if best is None:
                break
            vals.append(best[0])
            used_rows.add(best[1])
            used_cols.add(best[2])
        return float(np.mean(vals)) if vals else float("nan")


def linear_cka(z_hat: np.ndarray, z_true: np.ndarray) -> float:
    X = np.asarray(z_hat, dtype=np.float64)
    Y = np.asarray(z_true, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[0] != Y.shape[0] or X.shape[0] < 2:
        return float("nan")
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    norm_x = np.linalg.norm(X.T @ X, ord="fro")
    norm_y = np.linalg.norm(Y.T @ Y, ord="fro")
    return float(hsic / (norm_x * norm_y + 1e-12))


def state_recovery_metrics(z_hat: np.ndarray | None, z_true: np.ndarray) -> Dict[str, float]:
    if z_hat is None:
        return {"state_r2": float("nan"), "state_mcc": float("nan"), "state_cka": float("nan")}
    return {
        "state_r2": ridge_r2(z_hat, z_true),
        "state_mcc": mean_corrcoef_matching(z_hat, z_true),
        "state_cka": linear_cka(z_hat, z_true),
    }
