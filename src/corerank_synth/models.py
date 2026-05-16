from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    n_modalities: int = 3
    x_dim: int = 16
    z_dim: int = 6
    u_dim: int = 3
    hidden_dim: int = 64
    encoder_layers: int = 2
    decoder_layers: int = 2
    gate_temperature: float = 0.67
    init_gate_logit: float = 0.0
    decoder_noise_std: float = 0.35
    context_dim: int = 2
    structural_classifier: bool = True


def make_mlp(in_dim: int, out_dim: int, hidden_dim: int, layers: int, final_activation: Optional[str] = None) -> nn.Sequential:
    mods: List[nn.Module] = []
    last = in_dim
    for _ in range(max(0, layers - 1)):
        mods.append(nn.Linear(last, hidden_dim))
        mods.append(nn.SiLU())
        last = hidden_dim
    mods.append(nn.Linear(last, out_dim))
    if final_activation == "tanh":
        mods.append(nn.Tanh())
    elif final_activation == "sigmoid":
        mods.append(nn.Sigmoid())
    return nn.Sequential(*mods)


class ModalityEncoder(nn.Module):
    def __init__(self, x_dim: int, z_dim: int, u_dim: int, hidden_dim: int, layers: int):
        super().__init__()
        self.net = make_mlp(x_dim, 2 * z_dim + 2 * u_dim, hidden_dim, layers)
        self.z_dim = z_dim
        self.u_dim = u_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(x)
        z_mu, z_logvar, u_mu, u_logvar = torch.split(h, [self.z_dim, self.z_dim, self.u_dim, self.u_dim], dim=-1)
        z_logvar = z_logvar.clamp(-6.0, 4.0)
        u_logvar = u_logvar.clamp(-6.0, 4.0)
        return z_mu, z_logvar, u_mu, u_logvar


class ModalityDecoder(nn.Module):
    def __init__(self, z_dim: int, u_dim: int, x_dim: int, hidden_dim: int, layers: int):
        super().__init__()
        self.net = make_mlp(z_dim + u_dim, x_dim, hidden_dim, layers)

    def forward(self, z_gated: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_gated, u], dim=-1))


class SoftFootprintGates(nn.Module):
    def __init__(self, n_modalities: int, z_dim: int, init_logit: float = 0.0, temperature: float = 0.67):
        super().__init__()
        self.logits = nn.Parameter(torch.full((n_modalities, z_dim), init_logit))
        self.temperature = temperature

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.logits / max(self.temperature, 1e-6))

    def expected_l0(self) -> torch.Tensor:
        return self.forward().sum()


class CoreStructuralSEM(nn.Module):
    """Learn a directed graph among core coordinates.

    The graph is used as a soft linear SEM, z_j <- sum_k A[j, k] z_k + e_j.
    Acyclicity is encouraged in the training objective; the module itself only
    enforces a zero diagonal.
    """

    def __init__(self, z_dim: int, context_dim: int, init_scale: float = 0.02):
        super().__init__()
        self.z_dim = z_dim
        self.context_dim = context_dim
        self.weight = nn.Parameter(init_scale * torch.randn(z_dim, z_dim))
        self.context_weight = nn.Parameter(torch.zeros(z_dim, context_dim))
        self.register_buffer("offdiag_mask", torch.ones(z_dim, z_dim) - torch.eye(z_dim))
        self.register_buffer("identity", torch.eye(z_dim))

    def adjacency(self) -> torch.Tensor:
        return self.weight * self.offdiag_mask

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.adjacency().T

    def context_effect(self, context: torch.Tensor) -> torch.Tensor:
        return context @ self.context_weight.T

    def innovation(self, z: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return z - self.predict(z) - self.context_effect(context)

    def structural_matrix(self) -> torch.Tensor:
        return self.identity - self.adjacency()

    def log_abs_det(self) -> torch.Tensor:
        _, logabsdet = torch.linalg.slogdet(self.structural_matrix())
        return logabsdet

    def innovation_jacobian(self) -> torch.Tensor:
        return torch.linalg.inv(self.structural_matrix())

    def transform_information_to_innovation(self, K_z: torch.Tensor) -> torch.Tensor:
        jac = self.innovation_jacobian()
        return torch.einsum("ij,bjk,kl->bil", jac.T, K_z, jac)

    def acyclicity(self) -> torch.Tensor:
        graph = self.adjacency()
        return torch.trace(torch.matrix_exp(graph * graph)) - graph.shape[0]

    def l1(self) -> torch.Tensor:
        return self.adjacency().abs().mean()


class CoreRankVAE(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleList([
            ModalityEncoder(cfg.x_dim, cfg.z_dim, cfg.u_dim, cfg.hidden_dim, cfg.encoder_layers)
            for _ in range(cfg.n_modalities)
        ])
        self.decoders = nn.ModuleList([
            ModalityDecoder(cfg.z_dim, cfg.u_dim, cfg.x_dim, cfg.hidden_dim, cfg.decoder_layers)
            for _ in range(cfg.n_modalities)
        ])
        self.gates = SoftFootprintGates(cfg.n_modalities, cfg.z_dim, cfg.init_gate_logit, cfg.gate_temperature)
        self.core_graph = CoreStructuralSEM(cfg.z_dim, cfg.context_dim)
        self.classifier = make_mlp(cfg.z_dim, 1, cfg.hidden_dim, 2)

    def encode(self, xs: List[torch.Tensor], obs_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Return q(z|x_O) mean/logvar and per-modality q(u_m|x_m) params.

        obs_mask: [B, M] with 1 for observed modality.
        """
        device = xs[0].device
        batch = xs[0].shape[0]
        prior_prec = torch.ones(batch, self.cfg.z_dim, device=device)
        prec_sum = prior_prec.clone()
        eta_sum = torch.zeros(batch, self.cfg.z_dim, device=device)
        u_mus: List[torch.Tensor] = []
        u_logvars: List[torch.Tensor] = []
        for m, enc in enumerate(self.encoders):
            z_mu_m, z_lv_m, u_mu_m, u_lv_m = enc(xs[m])
            mask = obs_mask[:, m:m+1]
            z_prec_m = torch.exp(-z_lv_m) * mask
            eta_sum = eta_sum + z_mu_m * z_prec_m
            prec_sum = prec_sum + z_prec_m
            u_mus.append(u_mu_m)
            u_logvars.append(u_lv_m)
        z_var = 1.0 / prec_sum.clamp_min(1e-7)
        z_mu = z_var * eta_sum
        z_logvar = torch.log(z_var.clamp_min(1e-7))
        return z_mu, z_logvar, u_mus, u_logvars

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor, us: List[torch.Tensor]) -> List[torch.Tensor]:
        g = self.gates()
        recons = []
        for m, dec in enumerate(self.decoders):
            recons.append(dec(g[m].unsqueeze(0) * z, us[m]))
        return recons

    def _default_context(self, z: torch.Tensor) -> torch.Tensor:
        return torch.zeros(z.shape[0], self.cfg.context_dim, device=z.device, dtype=z.dtype)

    def core_features(self, z: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.cfg.structural_classifier:
            return z
        context = self._default_context(z) if context is None else context
        return self.core_graph.innovation(z, context)

    def forward(self, xs: List[torch.Tensor], obs_mask: torch.Tensor, context: Optional[torch.Tensor] = None, sample: bool = True) -> dict:
        z_mu, z_logvar, u_mus, u_logvars = self.encode(xs, obs_mask)
        context = self._default_context(z_mu) if context is None else context
        z = self.reparameterize(z_mu, z_logvar, sample=sample)
        us = [self.reparameterize(mu, lv, sample=sample) for mu, lv in zip(u_mus, u_logvars)]
        recons = self.decode(z, us)
        structural_pred = self.core_graph.predict(z_mu)
        structural_context = self.core_graph.context_effect(context)
        innovation_mu = self.core_graph.innovation(z_mu, context)
        innovation = self.core_graph.innovation(z, context)
        logits = self.classifier(innovation if self.cfg.structural_classifier else z)
        return {
            "z": z,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "u": us,
            "u_mu": u_mus,
            "u_logvar": u_logvars,
            "recon": recons,
            "logits": logits,
            "structural_pred": structural_pred,
            "structural_context": structural_context,
            "innovation": innovation,
            "innovation_mu": innovation_mu,
        }


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1)


def kl_sem_normal(mu: torch.Tensor, logvar: torch.Tensor, sem: CoreStructuralSEM, context: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || p_A(z|context)) for a linear SEM innovation prior.

    p_A is defined by e = (I - A)z - Gamma c, e ~ N(0, I).
    """
    graph_matrix = sem.structural_matrix()
    innovation_mu = sem.innovation(mu, context)
    var = torch.exp(logvar)
    var_term = (graph_matrix.pow(2).sum(dim=0).unsqueeze(0) * var).sum(dim=-1)
    mean_term = innovation_mu.pow(2).sum(dim=-1)
    entropy_term = logvar.sum(dim=-1)
    # For an acyclic zero-diagonal SEM, det(I - A) = 1. The acyclicity
    # constraint makes the log-Jacobian term constant, so the KL only needs
    # the Gaussian innovation energy and posterior entropy.
    return 0.5 * (mean_term + var_term - mu.shape[-1] - entropy_term)


class EarlyFusionClassifier(nn.Module):
    def __init__(self, n_modalities: int, x_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.n_modalities = n_modalities
        self.x_dim = x_dim
        self.net = make_mlp(n_modalities * x_dim, 1, hidden_dim, 3)

    def forward(self, xs: List[torch.Tensor], obs_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if obs_mask is None:
            xcat = torch.cat(xs, dim=-1)
        else:
            masked = [xs[m] * obs_mask[:, m:m+1] for m in range(self.n_modalities)]
            xcat = torch.cat(masked, dim=-1)
        return self.net(xcat)
