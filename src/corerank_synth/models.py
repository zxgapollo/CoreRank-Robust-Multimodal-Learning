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

    def forward(self, xs: List[torch.Tensor], obs_mask: torch.Tensor, sample: bool = True) -> dict:
        z_mu, z_logvar, u_mus, u_logvars = self.encode(xs, obs_mask)
        z = self.reparameterize(z_mu, z_logvar, sample=sample)
        us = [self.reparameterize(mu, lv, sample=sample) for mu, lv in zip(u_mus, u_logvars)]
        recons = self.decode(z, us)
        logits = self.classifier(z)
        return {
            "z": z,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "u": us,
            "u_mu": u_mus,
            "u_logvar": u_logvars,
            "recon": recons,
            "logits": logits,
        }


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1)


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
