from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn


@dataclass
class ModelConfig:
    n_modalities: int = 3
    x_dim: int = 16
    s_dim: int = 6
    hidden_dim: int = 64
    encoder_layers: int = 2
    decoder_layers: int = 2
    logvar_min: float = -6.0
    logvar_max: float = 4.0
    robust_fusion: bool = True
    agreement_scale: float = 1.0


def make_mlp(in_dim: int, out_dim: int, hidden_dim: int, layers: int) -> nn.Sequential:
    mods: List[nn.Module] = []
    last = in_dim
    for _ in range(max(0, layers - 1)):
        mods.append(nn.Linear(last, hidden_dim))
        mods.append(nn.SiLU())
        last = hidden_dim
    mods.append(nn.Linear(last, out_dim))
    return nn.Sequential(*mods)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, layers: int = 3):
        super().__init__()
        feature_layers: List[nn.Module] = []
        last = in_dim
        for _ in range(max(1, layers - 1)):
            feature_layers.append(nn.Linear(last, hidden_dim))
            feature_layers.append(nn.SiLU())
            last = hidden_dim
        self.features = nn.Sequential(*feature_layers)
        self.head = nn.Linear(last, 1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class UnimodalMLP(nn.Module):
    def __init__(self, modality: int, x_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.modality = modality
        self.net = MLPClassifier(x_dim, hidden_dim=hidden_dim)

    def forward(self, xs: List[torch.Tensor], obs_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.net(xs[self.modality])

    def representation(self, xs: List[torch.Tensor], obs_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.net.forward_features(xs[self.modality])


class ConcatMLP(nn.Module):
    def __init__(self, n_modalities: int, x_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.n_modalities = n_modalities
        self.x_dim = x_dim
        self.net = MLPClassifier(n_modalities * x_dim, hidden_dim=hidden_dim)

    def _concat(self, xs: List[torch.Tensor], obs_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if obs_mask is None:
            return torch.cat(xs, dim=-1)
        masked = [xs[m] * obs_mask[:, m : m + 1] for m in range(self.n_modalities)]
        return torch.cat(masked, dim=-1)

    def forward(self, xs: List[torch.Tensor], obs_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.net(self._concat(xs, obs_mask))

    def representation(self, xs: List[torch.Tensor], obs_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.net.forward_features(self._concat(xs, obs_mask))


class LateFusionMLP(nn.Module):
    def __init__(self, n_modalities: int, x_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.n_modalities = n_modalities
        self.encoders = nn.ModuleList([make_mlp(x_dim, hidden_dim, hidden_dim, 2) for _ in range(n_modalities)])
        self.head = make_mlp(hidden_dim, 1, hidden_dim, 2)

    def fused_features(self, xs: List[torch.Tensor], obs_mask: Optional[torch.Tensor]) -> torch.Tensor:
        hs = torch.stack([enc(xs[m]) for m, enc in enumerate(self.encoders)], dim=1)
        if obs_mask is None:
            return hs.mean(dim=1)
        weights = obs_mask.unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (hs * weights).sum(dim=1) / denom

    def forward(self, xs: List[torch.Tensor], obs_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.head(self.fused_features(xs, obs_mask))

    def representation(self, xs: List[torch.Tensor], obs_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.fused_features(xs, obs_mask)


class OracleStateMLP(nn.Module):
    def __init__(self, s_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = MLPClassifier(s_dim, hidden_dim=hidden_dim)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)

    def representation(self, s: torch.Tensor) -> torch.Tensor:
        return s


class GaussianExpertEncoder(nn.Module):
    def __init__(self, x_dim: int, s_dim: int, hidden_dim: int, layers: int, logvar_min: float, logvar_max: float):
        super().__init__()
        self.net = make_mlp(x_dim, 2 * s_dim, hidden_dim, layers)
        self.s_dim = s_dim
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = torch.split(self.net(x), self.s_dim, dim=-1)
        return mu, logvar.clamp(self.logvar_min, self.logvar_max)


class ISOPoE(nn.Module):
    """Product-of-Gaussian-experts model over the intrinsic state S*."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleList(
            [
                GaussianExpertEncoder(cfg.x_dim, cfg.s_dim, cfg.hidden_dim, cfg.encoder_layers, cfg.logvar_min, cfg.logvar_max)
                for _ in range(cfg.n_modalities)
            ]
        )
        self.decoders = nn.ModuleList(
            [make_mlp(cfg.s_dim, cfg.x_dim, cfg.hidden_dim, cfg.decoder_layers) for _ in range(cfg.n_modalities)]
        )
        self.classifier = make_mlp(cfg.s_dim, 1, cfg.hidden_dim, 2)

    def expert_params(self, xs: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        mus: List[torch.Tensor] = []
        logvars: List[torch.Tensor] = []
        for enc, x in zip(self.encoders, xs):
            mu, logvar = enc(x)
            mus.append(mu)
            logvars.append(logvar)
        return mus, logvars

    def fuse(self, mus: List[torch.Tensor], logvars: List[torch.Tensor], obs_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = mus[0].shape[0]
        device = mus[0].device
        precision = torch.ones(batch, self.cfg.s_dim, device=device)
        eta = torch.zeros(batch, self.cfg.s_dim, device=device)
        raw_precs = [torch.exp(-logvar) * obs_mask[:, m : m + 1] for m, logvar in enumerate(logvars)]
        if self.cfg.robust_fusion and len(mus) > 1:
            mu_stack = torch.stack(mus, dim=1)
            prec_stack = torch.stack(raw_precs, dim=1)
            total_prec = prec_stack.sum(dim=1).clamp_min(1e-7)
            total_eta = (prec_stack * mu_stack).sum(dim=1)
            adjusted_precs: List[torch.Tensor] = []
            for m, raw_prec in enumerate(raw_precs):
                other_prec = (total_prec - raw_prec).clamp_min(1e-7)
                other_mu = (total_eta - raw_prec * mus[m]) / other_prec
                has_other = (obs_mask.sum(dim=1, keepdim=True) - obs_mask[:, m : m + 1]) > 0.5
                disagreement = (mus[m] - other_mu).pow(2).mean(dim=-1, keepdim=True)
                agreement_weight = torch.exp(-disagreement / max(self.cfg.agreement_scale, 1e-6))
                agreement_weight = torch.where(has_other, agreement_weight, torch.ones_like(agreement_weight))
                adjusted_precs.append(raw_prec * agreement_weight)
            raw_precs = adjusted_precs
        for m, (mu, logvar) in enumerate(zip(mus, logvars)):
            prec_m = raw_precs[m]
            precision = precision + prec_m
            eta = eta + prec_m * mu
        var = 1.0 / precision.clamp_min(1e-7)
        mu = var * eta
        return mu, torch.log(var.clamp_min(1e-7))

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
        if not sample:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, xs: List[torch.Tensor], obs_mask: torch.Tensor, sample: bool = True) -> dict:
        expert_mu, expert_logvar = self.expert_params(xs)
        s_mu, s_logvar = self.fuse(expert_mu, expert_logvar, obs_mask)
        s = self.reparameterize(s_mu, s_logvar, sample=sample)
        recons = [decoder(s) for decoder in self.decoders]
        logits = self.classifier(s_mu)
        return {
            "logits": logits,
            "s": s,
            "s_mu": s_mu,
            "s_logvar": s_logvar,
            "expert_mu": expert_mu,
            "expert_logvar": expert_logvar,
            "recon": recons,
        }

    def representation(self, xs: List[torch.Tensor], obs_mask: torch.Tensor) -> torch.Tensor:
        mus, logvars = self.expert_params(xs)
        s_mu, _ = self.fuse(mus, logvars, obs_mask)
        return s_mu


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1)
