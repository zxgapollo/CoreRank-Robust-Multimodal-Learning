from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


@dataclass
class ModelConfig:
    n_modalities: int = 4
    x_dim: int = 10
    k: int = 6
    u_dim: int = 2
    hidden_dim: int = 64
    layers: int = 2
    logvar_min: float = -6.0
    logvar_max: float = 3.0
    fixed_graph: bool = False
    fixed_masks: bool = False


def make_mlp(in_dim: int, out_dim: int, hidden_dim: int, layers: int = 2) -> nn.Sequential:
    mods: List[nn.Module] = []
    last = in_dim
    for _ in range(max(0, layers - 1)):
        mods.append(nn.Linear(last, hidden_dim))
        mods.append(nn.SiLU())
        last = hidden_dim
    mods.append(nn.Linear(last, out_dim))
    return nn.Sequential(*mods)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
    if not sample:
        return mu
    return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1)


def diag_gaussian_kl(mu_a: torch.Tensor, logvar_a: torch.Tensor, mu_b: torch.Tensor, logvar_b: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(
        logvar_b - logvar_a + (torch.exp(logvar_a) + (mu_a - mu_b).pow(2)) / torch.exp(logvar_b).clamp_min(1e-8) - 1.0,
        dim=-1,
    )


class WarmupNet(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleList([make_mlp(cfg.x_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.layers) for _ in range(cfg.n_modalities)])
        self.proto_head = make_mlp(cfg.hidden_dim, 2 * cfg.k, cfg.hidden_dim, cfg.layers)
        self.classifier = make_mlp(cfg.k, 1, cfg.hidden_dim, cfg.layers)

    def fuse(self, xs: List[torch.Tensor], obs_mask: torch.Tensor) -> torch.Tensor:
        hs = torch.stack([enc(x) for enc, x in zip(self.encoders, xs)], dim=1)
        weights = obs_mask.unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (hs * weights).sum(dim=1) / denom

    def forward(self, xs: List[torch.Tensor], obs_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.fuse(xs, obs_mask)
        mu, logvar = torch.split(self.proto_head(h), self.cfg.k, dim=-1)
        logvar = logvar.clamp(self.cfg.logvar_min, self.cfg.logvar_max)
        logits = self.classifier(mu)
        return {"logits": logits, "proto_mu": mu, "proto_logvar": logvar}

    def representation(self, xs: List[torch.Tensor], obs_mask: torch.Tensor) -> torch.Tensor:
        return self.forward(xs, obs_mask)["proto_mu"]


class ConcatMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.net = make_mlp(cfg.n_modalities * cfg.x_dim, 1, cfg.hidden_dim, cfg.layers + 1)

    def _concat(self, xs: List[torch.Tensor], obs_mask: torch.Tensor) -> torch.Tensor:
        return torch.cat([xs[m] * obs_mask[:, m : m + 1] for m in range(self.cfg.n_modalities)], dim=-1)

    def forward(self, xs: List[torch.Tensor], obs_mask: torch.Tensor) -> torch.Tensor:
        return self.net(self._concat(xs, obs_mask))


class LateFusionMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleList([make_mlp(cfg.x_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.layers) for _ in range(cfg.n_modalities)])
        self.head = make_mlp(cfg.hidden_dim, 1, cfg.hidden_dim, cfg.layers)

    def fuse(self, xs: List[torch.Tensor], obs_mask: torch.Tensor) -> torch.Tensor:
        hs = torch.stack([enc(x) for enc, x in zip(self.encoders, xs)], dim=1)
        weights = obs_mask.unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (hs * weights).sum(dim=1) / denom

    def forward(self, xs: List[torch.Tensor], obs_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.fuse(xs, obs_mask))


class BCMCSGN(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        true_graph: Optional[torch.Tensor] = None,
        true_state_mask: Optional[torch.Tensor] = None,
        true_modality_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.modality_encoders = nn.ModuleList([make_mlp(cfg.x_dim, cfg.hidden_dim, cfg.hidden_dim, cfg.layers) for _ in range(cfg.n_modalities)])
        self.z_encoder = make_mlp(cfg.hidden_dim + cfg.k, 2 * cfg.k, cfg.hidden_dim, cfg.layers)
        self.u_encoders = nn.ModuleList([make_mlp(cfg.x_dim + cfg.k, 2 * cfg.u_dim, cfg.hidden_dim, cfg.layers) for _ in range(cfg.n_modalities)])
        self.decoders = nn.ModuleList(
            [make_mlp(2 * cfg.k + cfg.u_dim, cfg.x_dim, cfg.hidden_dim, cfg.layers) for _ in range(cfg.n_modalities)]
        )
        self.bias_net = make_mlp(cfg.n_modalities * cfg.u_dim, cfg.k, cfg.hidden_dim, cfg.layers)
        self.classifier = make_mlp(cfg.k, 1, cfg.hidden_dim, cfg.layers)
        lower = torch.tril(torch.ones(cfg.k, cfg.k), diagonal=-1)
        self.register_buffer("lower_mask", lower)

        if cfg.fixed_graph:
            if true_graph is None:
                raise ValueError("fixed_graph=True requires true_graph.")
            self.register_buffer("fixed_graph_value", true_graph.float())
        else:
            self.graph_raw = nn.Parameter(0.03 * torch.randn(cfg.k, cfg.k))

        if cfg.fixed_masks:
            if true_state_mask is None or true_modality_mask is None:
                raise ValueError("fixed_masks=True requires true_state_mask and true_modality_mask.")
            self.register_buffer("fixed_state_mask_value", true_state_mask.float())
            self.register_buffer("fixed_modality_mask_value", true_modality_mask.float())
        else:
            self.state_mask_logits = nn.Parameter(torch.zeros(cfg.k))
            self.modality_mask_logits = nn.Parameter(torch.zeros(cfg.n_modalities, cfg.k))

    def adjacency(self) -> torch.Tensor:
        if self.cfg.fixed_graph:
            return self.fixed_graph_value
        return self.graph_raw * self.lower_mask

    def state_mask(self) -> torch.Tensor:
        if self.cfg.fixed_masks:
            return self.fixed_state_mask_value
        return torch.sigmoid(self.state_mask_logits)

    def modality_mask(self) -> torch.Tensor:
        if self.cfg.fixed_masks:
            return self.fixed_modality_mask_value
        return torch.sigmoid(self.modality_mask_logits)

    def fuse_modalities(self, xs: List[torch.Tensor], obs_mask: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        hs = [enc(x) for enc, x in zip(self.modality_encoders, xs)]
        h_stack = torch.stack(hs, dim=1)
        weights = obs_mask.unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (h_stack * weights).sum(dim=1) / denom, hs

    def encode(
        self,
        xs: List[torch.Tensor],
        obs_mask: torch.Tensor,
        proto: torch.Tensor,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        fused, _ = self.fuse_modalities(xs, obs_mask)
        z_mu, z_logvar = torch.split(self.z_encoder(torch.cat([fused, proto], dim=-1)), self.cfg.k, dim=-1)
        z_logvar = z_logvar.clamp(self.cfg.logvar_min, self.cfg.logvar_max)
        u_mu: List[torch.Tensor] = []
        u_logvar: List[torch.Tensor] = []
        for enc, x in zip(self.u_encoders, xs):
            mu, logvar = torch.split(enc(torch.cat([x, proto], dim=-1)), self.cfg.u_dim, dim=-1)
            u_mu.append(mu)
            u_logvar.append(logvar.clamp(self.cfg.logvar_min, self.cfg.logvar_max))
        return {"z_mu": z_mu, "z_logvar": z_logvar, "u_mu": u_mu, "u_logvar": u_logvar}

    def state_from_z(self, z: torch.Tensor) -> torch.Tensor:
        a = self.adjacency()
        graph_features = z + z @ a.T
        return graph_features * self.state_mask()[None, :]

    def forward(
        self,
        xs: List[torch.Tensor],
        obs_mask: torch.Tensor,
        proto: torch.Tensor,
        sample: bool = True,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        enc = self.encode(xs, obs_mask, proto)
        z_mu = enc["z_mu"]  # type: ignore[assignment]
        z_logvar = enc["z_logvar"]  # type: ignore[assignment]
        u_mu = enc["u_mu"]  # type: ignore[assignment]
        u_logvar = enc["u_logvar"]  # type: ignore[assignment]
        z = reparameterize(z_mu, z_logvar, sample=sample)
        us = [reparameterize(mu, logvar, sample=sample) for mu, logvar in zip(u_mu, u_logvar)]
        s = self.state_from_z(z)
        delta_hat = self.bias_net(torch.cat(us, dim=-1))
        proto_recon = s + delta_hat
        logits = self.classifier(s)
        masks = self.modality_mask()
        recons = [
            dec(torch.cat([z * masks[m][None, :], s, us[m]], dim=-1))
            for m, dec in enumerate(self.decoders)
        ]
        return {
            "logits": logits,
            "z": z,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "u": us,
            "u_mu": u_mu,
            "u_logvar": u_logvar,
            "s": s,
            "delta_hat": delta_hat,
            "proto_recon": proto_recon,
            "recon": recons,
        }

    def graph_l1(self) -> torch.Tensor:
        return self.adjacency().abs().mean()

    def mask_l1(self) -> torch.Tensor:
        return self.state_mask().mean() + self.modality_mask().mean()

    def dag_penalty(self) -> torch.Tensor:
        a = self.adjacency()
        return torch.trace(torch.matrix_exp(a * a)) - a.shape[0]
