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
    fixed_structure: bool = False
    init_incidence_probability: float = 0.50
    init_task_probability: float = 0.50
    incidence_init_noise: float = 0.25
    task_init_noise: float = 0.15
    poe_prior_precision: float = 0.10
    gate_threshold: float = 0.50


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
    """Discriminative prototype retained only as a baseline."""

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
        return {"logits": self.classifier(mu), "proto_mu": mu, "proto_logvar": logvar}

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


class MultimodalTransformer(nn.Module):
    """Lightweight fully discriminative multimodal Transformer baseline.

    Each modality is embedded as one token. A learned CLS token aggregates
    observed modalities while ``obs_mask`` is used only as a padding mask.
    There is no latent-factor bottleneck, structural incidence, certificate,
    reconstruction path, or causal objective in this baseline.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(cfg.x_dim, cfg.hidden_dim),
                    nn.LayerNorm(cfg.hidden_dim),
                    nn.GELU(),
                )
                for _ in range(cfg.n_modalities)
            ]
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
        self.modality_embedding = nn.Parameter(torch.randn(1, cfg.n_modalities, cfg.hidden_dim) * 0.02)
        n_heads = min(4, cfg.hidden_dim)
        while cfg.hidden_dim % n_heads != 0:
            n_heads -= 1
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=n_heads,
            dim_feedforward=2 * cfg.hidden_dim,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, cfg.layers))
        self.head = nn.Sequential(nn.LayerNorm(cfg.hidden_dim), nn.Linear(cfg.hidden_dim, 1))

    def forward(self, xs: List[torch.Tensor], obs_mask: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([encoder(x) for encoder, x in zip(self.token_encoders, xs)], dim=1)
        tokens = tokens + self.modality_embedding
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        sequence = torch.cat([cls, tokens], dim=1)
        cls_visible = torch.zeros((obs_mask.shape[0], 1), dtype=torch.bool, device=obs_mask.device)
        padding_mask = torch.cat([cls_visible, obs_mask < 0.5], dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=padding_mask)
        return self.head(encoded[:, 0])


class SFMNet(nn.Module):
    """Selective Factorization Model aligned with the identifiability theorem.

    The decoder for modality m receives only ``B[m] * Z`` and its private
    residual ``U_m``. In particular, there is no full-state or classifier
    bypass into a decoder. The same incidence matrix B is used by the
    uncertainty-aware product-of-experts encoder and by the witness
    certificate.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        true_task_mask: Optional[torch.Tensor] = None,
        true_incidence: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.content_encoders = nn.ModuleList(
            [make_mlp(cfg.x_dim, 2 * cfg.k, cfg.hidden_dim, cfg.layers) for _ in range(cfg.n_modalities)]
        )
        self.private_encoders = nn.ModuleList(
            [make_mlp(cfg.x_dim, 2 * cfg.u_dim, cfg.hidden_dim, cfg.layers) for _ in range(cfg.n_modalities)]
        )
        # The k content inputs are gated before the decoder. No global S enters.
        self.decoders = nn.ModuleList(
            [make_mlp(cfg.k + cfg.u_dim, cfg.x_dim, cfg.hidden_dim, cfg.layers) for _ in range(cfg.n_modalities)]
        )
        self.selected_classifier = make_mlp(cfg.k, 1, cfg.hidden_dim, cfg.layers)
        self.full_classifier = make_mlp(cfg.k, 1, cfg.hidden_dim, cfg.layers)

        if cfg.fixed_structure:
            if true_task_mask is None or true_incidence is None:
                raise ValueError("fixed_structure=True requires true_task_mask and true_incidence")
            self.register_buffer("fixed_task_mask_value", true_task_mask.float())
            self.register_buffer("fixed_incidence_value", true_incidence.float())
        else:
            p = min(max(cfg.init_incidence_probability, 1e-4), 1.0 - 1e-4)
            init = torch.logit(torch.tensor(p))
            self.incidence_logits = nn.Parameter(init + cfg.incidence_init_noise * torch.randn(cfg.n_modalities, cfg.k))
            task_p = min(max(cfg.init_task_probability, 1e-4), 1.0 - 1e-4)
            task_init = float(torch.logit(torch.tensor(task_p)))
            self.task_mask_logits = nn.Parameter(task_init + cfg.task_init_noise * torch.randn(cfg.k))

    def incidence(self, hard: bool = False) -> torch.Tensor:
        gate = self.fixed_incidence_value if self.cfg.fixed_structure else torch.sigmoid(self.incidence_logits)
        return (gate >= self.cfg.gate_threshold).float() if hard else gate

    def task_mask(self, hard: bool = False) -> torch.Tensor:
        gate = self.fixed_task_mask_value if self.cfg.fixed_structure else torch.sigmoid(self.task_mask_logits)
        return (gate >= self.cfg.gate_threshold).float() if hard else gate

    def incidence_gate(self) -> torch.Tensor:
        """Binary support in the forward pass with a sigmoid surrogate gradient."""
        if self.cfg.fixed_structure:
            return self.fixed_incidence_value
        soft = self.incidence()
        hard = (soft >= self.cfg.gate_threshold).float()
        return hard.detach() - soft.detach() + soft

    def task_gate(self) -> torch.Tensor:
        if self.cfg.fixed_structure:
            return self.fixed_task_mask_value
        soft = self.task_mask()
        hard = (soft >= self.cfg.gate_threshold).float()
        return hard.detach() - soft.detach() + soft

    def encode(self, xs: List[torch.Tensor], obs_mask: torch.Tensor) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        expert_mu: List[torch.Tensor] = []
        expert_logvar: List[torch.Tensor] = []
        u_mu: List[torch.Tensor] = []
        u_logvar: List[torch.Tensor] = []
        for content_enc, private_enc, x in zip(self.content_encoders, self.private_encoders, xs):
            mu_m, logvar_m = torch.split(content_enc(x), self.cfg.k, dim=-1)
            expert_mu.append(mu_m)
            expert_logvar.append(logvar_m.clamp(self.cfg.logvar_min, self.cfg.logvar_max))
            mu_u, logvar_u = torch.split(private_enc(x), self.cfg.u_dim, dim=-1)
            u_mu.append(mu_u)
            u_logvar.append(logvar_u.clamp(self.cfg.logvar_min, self.cfg.logvar_max))

        mu_stack = torch.stack(expert_mu, dim=1)
        logvar_stack = torch.stack(expert_logvar, dim=1)
        evidence = obs_mask[:, :, None] * self.incidence_gate()[None, :, :]
        expert_precision = evidence * torch.exp(-logvar_stack)
        precision = self.cfg.poe_prior_precision + expert_precision.sum(dim=1)
        z_mu = (expert_precision * mu_stack).sum(dim=1) / precision.clamp_min(1e-8)
        z_logvar = -torch.log(precision.clamp_min(1e-8))
        z_logvar = z_logvar.clamp(self.cfg.logvar_min, self.cfg.logvar_max)
        return {
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "expert_mu": expert_mu,
            "expert_logvar": expert_logvar,
            "u_mu": u_mu,
            "u_logvar": u_logvar,
        }

    def decode(self, z: torch.Tensor, us: List[torch.Tensor]) -> List[torch.Tensor]:
        gates = self.incidence_gate()
        return [
            decoder(torch.cat([z * gates[m][None, :], us[m]], dim=-1))
            for m, decoder in enumerate(self.decoders)
        ]

    def state_from_z(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.task_gate()[None, :]

    def forward(
        self,
        xs: List[torch.Tensor],
        obs_mask: torch.Tensor,
        sample: bool = True,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        enc = self.encode(xs, obs_mask)
        z_mu = enc["z_mu"]
        z_logvar = enc["z_logvar"]
        u_mu = enc["u_mu"]
        u_logvar = enc["u_logvar"]
        assert isinstance(z_mu, torch.Tensor) and isinstance(z_logvar, torch.Tensor)
        assert isinstance(u_mu, list) and isinstance(u_logvar, list)
        z = reparameterize(z_mu, z_logvar, sample=sample)
        us = [reparameterize(mu, logvar, sample=sample) for mu, logvar in zip(u_mu, u_logvar)]
        s = self.state_from_z(z)
        certificate, factor_certificate = self.structure_certificate(obs_mask, hard=True)
        return {
            "logits": self.selected_classifier(s),
            "full_logits": self.full_classifier(z),
            "z": z,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "u": us,
            "u_mu": u_mu,
            "u_logvar": u_logvar,
            "s": s,
            "recon": self.decode(z, us),
            "incidence": self.incidence(),
            "task_mask": self.task_mask(),
            "certificate": certificate,
            "factor_certificate": factor_certificate,
        }

    def witness_scores(self, discrete_forward: bool = False) -> torch.Tensor:
        """W[j,k]: some modality contains j and excludes k.

        During optimization, ``discrete_forward=True`` evaluates the actual
        binary support while retaining a straight-through gradient. This keeps
        a collection of 0.5 gates from masquerading as a valid witness.
        """
        b = self.incidence_gate() if discrete_forward else self.incidence()
        pair = b[:, :, None] * (1.0 - b[:, None, :])
        scores = 1.0 - torch.prod(1.0 - pair, dim=0)
        eye = torch.eye(self.cfg.k, dtype=torch.bool, device=scores.device)
        return scores.masked_fill(eye, 1.0)

    def witness_loss(self, margin: float = 0.80) -> torch.Tensor:
        scores = self.witness_scores(discrete_forward=True)
        relevance = self.task_gate()[:, None]
        offdiag = ~torch.eye(self.cfg.k, dtype=torch.bool, device=scores.device)
        hinge = torch.relu(margin - scores).pow(2) * offdiag
        return (hinge * relevance).sum() / (relevance.sum() * max(1, self.cfg.k - 1)).clamp_min(1e-8)

    def structure_certificate(self, obs_mask: torch.Tensor, hard: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return sample certificate and per-factor certificate.

        For a selected factor j, certification requires an observed modality
        that includes j but excludes every competitor k, one competitor at a
        time. This is the finite support witness condition in executable form.
        """
        squeeze = obs_mask.ndim == 1
        obs = obs_mask[None, :] if squeeze else obs_mask
        b = self.incidence(hard=hard)
        r = self.task_mask(hard=hard)
        pair = b[:, :, None] * (1.0 - b[:, None, :])
        available = obs[:, :, None, None] * pair[None, :, :, :]
        pair_cert = available.amax(dim=1)
        offdiag = ~torch.eye(self.cfg.k, dtype=torch.bool, device=pair_cert.device)
        factor_cert = torch.where(offdiag[None, :, :], pair_cert, torch.ones_like(pair_cert)).amin(dim=-1)
        selected = r > 0.5
        if bool(selected.any()):
            certificate = factor_cert[:, selected].amin(dim=-1)
        else:
            certificate = torch.zeros(obs.shape[0], device=obs.device, dtype=obs.dtype)
        if squeeze:
            return certificate[0], factor_cert[0]
        return certificate, factor_cert

    def decoder_edge_strength(self) -> torch.Tensor:
        """Normalized first-layer sensitivity for each declared Z -> X_m edge."""
        rows: List[torch.Tensor] = []
        for decoder in self.decoders:
            first_linear = next(module for module in decoder if isinstance(module, nn.Linear))
            strength = torch.linalg.vector_norm(first_linear.weight[:, : self.cfg.k], dim=0)
            rows.append(strength / strength.amax().clamp_min(1e-8))
        return torch.stack(rows, dim=0)

    def faithfulness_loss(self, margin: float = 0.20) -> torch.Tensor:
        b = self.incidence()
        penalty = b * torch.relu(margin - self.decoder_edge_strength()).pow(2)
        return penalty.sum() / b.sum().clamp_min(1e-8)

    def gate_regularizer(self) -> torch.Tensor:
        b = self.incidence()
        r = self.task_mask()
        binary = (b * (1.0 - b)).mean() + (r * (1.0 - r)).mean()
        nonempty = torch.relu(1.0 - r.sum()).pow(2)
        return binary + nonempty

    def residual_intervention_loss(
        self,
        z_reference: torch.Tensor,
        us: List[torch.Tensor],
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Change only private residuals, regenerate, and re-encode selected Z."""
        if z_reference.shape[0] < 2:
            return z_reference.new_zeros(())
        permuted = [torch.roll(u, shifts=1, dims=0) for u in us]
        counterfactual_x = self.decode(z_reference, permuted)
        reencoded = self.encode(counterfactual_x, obs_mask)["z_mu"]
        assert isinstance(reencoded, torch.Tensor)
        r = self.task_gate()[None, :]
        return ((reencoded - z_reference.detach()).pow(2) * r).sum() / (r.sum() * z_reference.shape[0]).clamp_min(1e-8)


# Backward import compatibility. The implementation is now SFM-Net and has no DAG.
BCMCSGN = SFMNet
