from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .data import ISODataConfig, ISOParams


def modality_subsets(n_modalities: int) -> List[Tuple[int, ...]]:
    subsets: List[Tuple[int, ...]] = []
    for k in range(1, n_modalities + 1):
        subsets.extend(itertools.combinations(range(n_modalities), k))
    return subsets


def subset_name(subset: Sequence[int]) -> str:
    return "".join(str(i) for i in subset)


def nuisance_adjusted_information(
    params: ISOParams,
    cfg: ISODataConfig,
    subset: Iterable[int],
    damping: float = 1e-3,
) -> np.ndarray:
    """Compute oracle nuisance-adjusted Fisher information for S*.

    For the linear Gaussian part of the generator,

        X_m = A_m S* + B_m U_m + eps_m,

    the effective state information after adjusting out U_m is the Schur
    complement of the joint Fisher block.
    """

    r = cfg.s_dim
    K = np.zeros((r, r), dtype=np.float64)
    for mm in subset:
        A = params.A[mm].astype(np.float64)
        B = params.B[mm].astype(np.float64)
        inv_sigma2 = 1.0 / float(params.noise_stds[mm] ** 2)
        Izz = inv_sigma2 * (A.T @ A)
        Izu = inv_sigma2 * (A.T @ B)
        Iuu = inv_sigma2 * (B.T @ B)
        if B.size == 0:
            K += Izz
        else:
            K += Izz - Izu @ np.linalg.pinv(Iuu + damping * np.eye(Iuu.shape[0])) @ Izu.T
    return 0.5 * (K + K.T)


def ambiguity_proxy(
    params: ISOParams,
    cfg: ISODataConfig,
    subset: Iterable[int],
    prior_precision: float = 1.0,
) -> float:
    """Return beta^T Cov(S* | X_M) beta / beta^T beta under a Gaussian proxy."""

    K = nuisance_adjusted_information(params, cfg, subset)
    precision = prior_precision * np.eye(cfg.s_dim) + K
    cov = np.linalg.pinv(precision)
    beta = params.beta_y.astype(np.float64)
    denom = float(beta @ beta) + 1e-12
    return float(beta @ cov @ beta / denom)


def observability_score(params: ISOParams, cfg: ISODataConfig, subset: Iterable[int]) -> float:
    """Label-relevant observability score lambda_Y(M) in [0, 1]."""

    amb = ambiguity_proxy(params, cfg, subset)
    return float(np.clip(1.0 - amb, 0.0, 1.0))


def diagnostics_for_subset(params: ISOParams, cfg: ISODataConfig, subset: Sequence[int]) -> Dict[str, float | str | int]:
    K = nuisance_adjusted_information(params, cfg, subset)
    eig = np.linalg.eigvalsh(K).clip(min=0.0)
    trace = float(eig.sum())
    if trace <= 1e-12:
        effective_rank = 0.0
        min_eig_norm = 0.0
    else:
        probs = eig / trace
        effective_rank = float(np.exp(-(probs * np.log(probs + 1e-12)).sum()))
        min_eig_norm = float((cfg.s_dim * eig / trace).min())
    return {
        "subset": subset_name(subset),
        "subset_size": len(subset),
        "lambda_y": observability_score(params, cfg, subset),
        "ambiguity_proxy": ambiguity_proxy(params, cfg, subset),
        "oracle_effective_rank": effective_rank,
        "oracle_min_eig_norm": min_eig_norm,
        "oracle_trace": trace,
    }


def diagnostics_table(params: ISOParams, cfg: ISODataConfig) -> List[Dict[str, float | str | int]]:
    return [diagnostics_for_subset(params, cfg, subset) for subset in modality_subsets(cfg.n_modalities)]
