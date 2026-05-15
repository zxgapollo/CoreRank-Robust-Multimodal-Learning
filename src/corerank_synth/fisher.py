from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch.func import jacrev, vmap

from .models import CoreRankVAE


def _symmetrize(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.transpose(-1, -2))


def _batched_eye(batch: int, dim: int, ref: torch.Tensor) -> torch.Tensor:
    return torch.eye(dim, device=ref.device, dtype=ref.dtype).unsqueeze(0).expand(batch, dim, dim)


def modality_core_fisher(
    model: CoreRankVAE,
    modality: int,
    z: torch.Tensor,
    u: torch.Tensor,
    noise_std: float,
    damping: float = 1e-3,
) -> torch.Tensor:
    """Compute decoder-induced nuisance-adjusted Fisher for one modality.

    Returns [B, r, r]. The decoder mean is differentiated wrt the global z and u.
    The learned footprint gate is included inside the decoder input, so the Jacobian wrt z
    already includes the gate effect.
    """
    dec = model.decoders[modality]
    gate = model.gates()[modality]
    sigma2_inv = 1.0 / (noise_std ** 2)
    r = model.cfg.z_dim
    q = model.cfg.u_dim

    def mean_wrt_z(z_single: torch.Tensor, u_single: torch.Tensor) -> torch.Tensor:
        return dec(gate * z_single, u_single)

    # Jz/Ju shapes: [B, x_dim, z_dim/u_dim]
    Jz = vmap(jacrev(mean_wrt_z, argnums=0))(z, u)
    Ju = vmap(jacrev(mean_wrt_z, argnums=1))(z, u)

    Izz = sigma2_inv * torch.matmul(Jz.transpose(-1, -2), Jz)
    Izu = sigma2_inv * torch.matmul(Jz.transpose(-1, -2), Ju)
    Iuu = sigma2_inv * torch.matmul(Ju.transpose(-1, -2), Ju)
    eye_u = _batched_eye(Iuu.shape[0], q, z)
    # Schur complement: Izz - Izu (Iuu + damping I)^-1 Iuz.
    sol = torch.linalg.solve(Iuu + damping * eye_u, Izu.transpose(-1, -2))
    Icore = Izz - torch.matmul(Izu, sol)
    eye_z = _batched_eye(Icore.shape[0], r, z)
    Icore = _symmetrize(Icore) + 1e-9 * eye_z
    return Icore


def batch_core_information(
    model: CoreRankVAE,
    z: torch.Tensor,
    us: List[torch.Tensor],
    obs_mask: torch.Tensor,
    noise_std: float,
    damping: float = 1e-3,
    max_fisher_batch: Optional[int] = None,
) -> torch.Tensor:
    """Compute K_O(x)=sum_m I_m^core for each sample.

    If max_fisher_batch is set, compute the Fisher on a random subset of the batch to reduce cost.
    The caller should align the returned samples with rank computation only; this is mainly for training.
    """
    if max_fisher_batch is not None and z.shape[0] > max_fisher_batch:
        idx = torch.randperm(z.shape[0], device=z.device)[:max_fisher_batch]
        z = z[idx]
        obs_mask = obs_mask[idx]
        us = [um[idx] for um in us]
    b, r = z.shape
    K = torch.zeros(b, r, r, device=z.device, dtype=z.dtype)
    for m in range(model.cfg.n_modalities):
        mask = obs_mask[:, m].view(-1, 1, 1)
        if torch.count_nonzero(mask) == 0:
            continue
        Icore = modality_core_fisher(model, m, z, us[m], noise_std=noise_std, damping=damping)
        K = K + mask * Icore
    return _symmetrize(K)


def rank_score_from_K(K: torch.Tensor, eps: float = 1e-3, jitter: float = 1e-6) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Return mean normalized-logdet rank score and diagnostics.

    K: [B, r, r].
    """
    if K.ndim != 3 or K.shape[-1] != K.shape[-2]:
        raise ValueError(f"K must have shape [B, r, r], got {tuple(K.shape)}")
    b, r, _ = K.shape

    K = _symmetrize(K)
    eig_raw = torch.linalg.eigvalsh(K)
    eig = eig_raw.clamp_min(0.0)
    trace = eig.sum(dim=-1)
    safe_trace = trace.clamp_min(jitter)
    has_information = trace > jitter

    # Normalize by trace without adding isotropic mass before normalization.
    # Otherwise K=0 would become Kbar=I and falsely satisfy the rank constraint.
    eig_bar = (r * eig) / safe_trace.unsqueeze(-1)
    logdet = torch.log(eig_bar + eps).sum(dim=-1)
    zero_info_logdet = torch.full_like(logdet, r * math.log(eps))
    logdet = torch.where(has_information, logdet, zero_info_logdet)

    probs = eig / safe_trace.unsqueeze(-1)
    entropy_rank = torch.exp(-(probs * (probs + jitter).log()).sum(dim=-1))
    eff_rank = torch.where(has_information, entropy_rank, torch.zeros_like(entropy_rank))
    min_eig = torch.where(has_information, eig_bar.min(dim=-1).values, torch.zeros_like(trace))
    return logdet.mean(), {
        "logdet_per_sample": logdet.detach(),
        "effective_rank": eff_rank.detach(),
        "min_eig": min_eig.detach(),
        "trace": trace.detach(),
        "eigvals": eig_bar.detach(),
        "raw_eigvals": eig_raw.detach(),
    }
