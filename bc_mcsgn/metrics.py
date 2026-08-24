from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def binary_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    y = y_true.reshape(-1).astype(int)
    p = np.clip(prob.reshape(-1), 1e-7, 1.0 - 1e-7)
    pred = (p >= 0.5).astype(int)
    out = {
        "acc": float((pred == y).mean()),
        "nll": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
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


def ridge_predict(x: np.ndarray, y: np.ndarray, alpha: float = 1e-3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    reg = alpha * np.eye(xb.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    try:
        w = np.linalg.solve(xb.T @ xb + reg, xb.T @ y)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(xb.T @ xb + reg) @ xb.T @ y
    return xb @ w


def ridge_r2(z_hat: np.ndarray | None, z_true: np.ndarray, alpha: float = 1e-3) -> float:
    if z_hat is None:
        return float("nan")
    x = np.asarray(z_hat, dtype=np.float64)
    y = np.asarray(z_true, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        return float("nan")
    pred = ridge_predict(x, y, alpha=alpha)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean(axis=0, keepdims=True)) ** 2).sum()) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def mean_corrcoef_matching(z_hat: np.ndarray | None, z_true: np.ndarray) -> float:
    if z_hat is None:
        return float("nan")
    x = np.asarray(z_hat, dtype=np.float64)
    y = np.asarray(z_true, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        return float("nan")
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)
    y = (y - y.mean(axis=0, keepdims=True)) / (y.std(axis=0, keepdims=True) + 1e-8)
    corr = np.abs((x.T @ y) / x.shape[0])
    try:
        from scipy.optimize import linear_sum_assignment

        row, col = linear_sum_assignment(-corr)
        return float(corr[row, col].mean())
    except Exception:
        vals = []
        used_r: set[int] = set()
        used_c: set[int] = set()
        for _ in range(min(corr.shape)):
            best: Tuple[float, int, int] | None = None
            for i in range(corr.shape[0]):
                if i in used_r:
                    continue
                for j in range(corr.shape[1]):
                    if j in used_c:
                        continue
                    if best is None or corr[i, j] > best[0]:
                        best = (float(corr[i, j]), i, j)
            if best is None:
                break
            vals.append(best[0])
            used_r.add(best[1])
            used_c.add(best[2])
        return float(np.mean(vals)) if vals else float("nan")


def state_recovery_metrics(z_hat: np.ndarray | None, z_true: np.ndarray, prefix: str = "state") -> Dict[str, float]:
    return {
        f"{prefix}_r2": ridge_r2(z_hat, z_true),
        f"{prefix}_mcc": mean_corrcoef_matching(z_hat, z_true),
    }


def shortcut_sensitivity(prob: np.ndarray, true_s: np.ndarray) -> float:
    """Approximate E Var(f(X)|S) by residual variance after a ridge fit on true S."""
    p = np.asarray(prob, dtype=np.float64).reshape(-1, 1)
    s = np.asarray(true_s, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] != p.shape[0] or s.shape[0] < 2:
        return float("nan")
    pred_from_s = ridge_predict(s, p)
    return float(np.var(p - pred_from_s))


def _binary_f1(pred: np.ndarray, true: np.ndarray) -> float:
    pred = pred.astype(bool)
    true = true.astype(bool)
    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 1.0


def graph_binary(a: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    out = np.abs(np.asarray(a)) >= threshold
    np.fill_diagonal(out, False)
    return out


def transitive_closure(adj: np.ndarray) -> np.ndarray:
    reach = adj.astype(bool).copy()
    n = reach.shape[0]
    for k in range(n):
        reach = np.logical_or(reach, np.logical_and(reach[:, k : k + 1], reach[k : k + 1, :]))
    np.fill_diagonal(reach, False)
    return reach


def graph_recovery_metrics(a_hat: np.ndarray, a_true: np.ndarray, threshold: float = 0.15) -> Dict[str, float]:
    pred = graph_binary(a_hat, threshold=threshold)
    true = graph_binary(a_true, threshold=1e-8)
    pred_skel = np.logical_or(pred, pred.T)
    true_skel = np.logical_or(true, true.T)
    shd = float(np.logical_xor(pred, true).sum())
    return {
        "edge_f1": _binary_f1(pred, true),
        "skeleton_f1": _binary_f1(pred_skel, true_skel),
        "ancestor_f1": _binary_f1(transitive_closure(pred), transitive_closure(true)),
        "shd": shd,
    }


def mask_recovery_metrics(mask_hat: np.ndarray, mask_true: np.ndarray, threshold: float = 0.5, prefix: str = "mask") -> Dict[str, float]:
    pred = np.asarray(mask_hat) >= threshold
    true = np.asarray(mask_true) > 0
    return {f"{prefix}_f1": _binary_f1(pred, true)}
