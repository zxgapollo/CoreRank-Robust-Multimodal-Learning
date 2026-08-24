from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from torch import nn
import torch.nn.functional as F


def mlp(in_dim: int, out_dim: int, hidden: int, dropout: float = 0.1) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class MRIEncoder(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        widths = [1, 16, 32, 64, 96, 128]
        blocks: List[nn.Module] = []
        for in_channels, out_channels in zip(widths[:-1], widths[1:]):
            blocks.extend(
                [
                    nn.Conv3d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(min(8, out_channels), out_channels),
                    nn.GELU(),
                ]
            )
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(widths[-1], hidden), nn.LayerNorm(hidden), nn.GELU())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(self.pool(self.features(image)))


class ModalityStem(nn.Module):
    def __init__(self, group_dims: Sequence[int], hidden: int):
        super().__init__()
        self.mri = MRIEncoder(hidden)
        self.tabular = nn.ModuleList([mlp(dim, hidden, hidden) for dim in group_dims])

    def forward(self, image: torch.Tensor, groups: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return [self.mri(image)] + [encoder(value) for encoder, value in zip(self.tabular, groups)]


class SPMNet(nn.Module):
    """Selective product-of-experts multimodal network for real ADNI data.

    The default configuration is intentionally identical to the original ADNI
    implementation.  Explicit switches expose one-factor-at-a-time ablations
    without relying on zero-valued penalties to approximate a removed module.
    """

    def __init__(
        self,
        group_dims: Sequence[int],
        hidden: int = 128,
        latent: int = 32,
        private: int = 8,
        classes: int = 3,
        incidence_mode: str = "learned",
        task_mode: str = "learned",
        fusion: str = "poe",
        use_private: bool = True,
        direct_bypass: bool = False,
    ):
        super().__init__()
        if incidence_mode not in {"learned", "all"}:
            raise ValueError(f"Unknown incidence mode: {incidence_mode}")
        if task_mode not in {"learned", "all"}:
            raise ValueError(f"Unknown task mode: {task_mode}")
        if fusion not in {"poe", "mean"}:
            raise ValueError(f"Unknown fusion rule: {fusion}")
        if private < 1:
            raise ValueError("private must be positive")
        self.modality_count = 1 + len(group_dims)
        self.latent = latent
        self.private_dim = private
        self.incidence_mode = incidence_mode
        self.task_mode = task_mode
        self.fusion = fusion
        self.use_private = use_private
        self.direct_bypass = direct_bypass
        self.stem = ModalityStem(group_dims, hidden)
        self.content = nn.ModuleList([mlp(hidden, 2 * latent, hidden) for _ in range(self.modality_count)])
        self.private = nn.ModuleList(
            [mlp(hidden, 2 * private, hidden) for _ in range(self.modality_count)] if use_private else []
        )
        decoder_input = latent + (private if use_private else 0)
        self.decoders = nn.ModuleList([mlp(decoder_input, hidden, hidden) for _ in range(self.modality_count)])
        self.classifier = mlp(latent, classes, hidden)
        self.incidence_logits = nn.Parameter(
            torch.full((self.modality_count, latent), 0.85),
            requires_grad=incidence_mode == "learned",
        )
        self.task_logits = nn.Parameter(torch.zeros(latent), requires_grad=task_mode == "learned")
        if direct_bypass:
            self.bypass_projection = mlp(hidden, latent, hidden)
        self.prior_precision = 0.10

    def incidence(self) -> torch.Tensor:
        if self.incidence_mode == "all":
            return torch.ones_like(self.incidence_logits)
        return torch.sigmoid(self.incidence_logits)

    def task_mask(self) -> torch.Tensor:
        if self.task_mode == "all":
            return torch.ones_like(self.task_logits)
        return torch.sigmoid(self.task_logits)

    def _st_gate(self, soft: torch.Tensor) -> torch.Tensor:
        hard = (soft >= 0.5).to(soft.dtype)
        return hard.detach() - soft.detach() + soft

    def _fuse(self, mus: torch.Tensor, logvars: torch.Tensor, availability: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        incidence = self._st_gate(self.incidence())
        evidence = availability[:, :, None] * incidence[None, :, :]
        if self.fusion == "poe":
            expert_precision = evidence * torch.exp(-logvars)
            precision = self.prior_precision + expert_precision.sum(dim=1)
            z_mu = (expert_precision * mus).sum(dim=1) / precision.clamp_min(1e-8)
            z_logvar = (-torch.log(precision.clamp_min(1e-8))).clamp(-6.0, 3.0)
        else:
            count = evidence.sum(dim=1)
            safe_count = count.clamp_min(1.0)
            z_mu = (evidence * mus).sum(dim=1) / safe_count
            # Independent equal-weight Gaussian experts: variance of their mean.
            variance = (evidence * torch.exp(logvars)).sum(dim=1) / safe_count.pow(2)
            empty = count < 0.5
            z_mu = torch.where(empty, torch.zeros_like(z_mu), z_mu)
            prior_variance = torch.full_like(variance, 1.0 / self.prior_precision)
            variance = torch.where(empty, prior_variance, variance)
            z_logvar = torch.log(variance.clamp_min(1e-8)).clamp(-6.0, 3.0)
        return z_mu, z_logvar

    def forward_from_embeddings(
        self,
        embeddings: Sequence[torch.Tensor],
        availability: torch.Tensor,
        sample: bool = True,
        modality_dropout: float = 0.0,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        """Run the latent model from precomputed stems.

        This split is useful for missing-modality evaluation: expensive MRI and
        tabular stems are computed once, then the same subject batch is evaluated
        under multiple availability masks.
        """
        if len(embeddings) != self.modality_count:
            raise ValueError(f"Expected {self.modality_count} embeddings, found {len(embeddings)}")

        content_params = [head(value) for head, value in zip(self.content, embeddings)]
        mus = torch.stack([value.chunk(2, dim=-1)[0] for value in content_params], dim=1)
        logvars = torch.stack([value.chunk(2, dim=-1)[1].clamp(-6.0, 3.0) for value in content_params], dim=1)
        if self.use_private:
            private_params = [head(value) for head, value in zip(self.private, embeddings)]
            u_mus = [value.chunk(2, dim=-1)[0] for value in private_params]
            u_logvars = [value.chunk(2, dim=-1)[1].clamp(-6.0, 3.0) for value in private_params]
        else:
            u_mus = []
            u_logvars = []

        effective = availability
        if self.training and modality_dropout > 0:
            keep = (torch.rand_like(availability) >= modality_dropout).to(availability.dtype)
            keep[:, 0] = 1.0
            effective = availability * keep
        z_mu, z_logvar = self._fuse(mus, logvars, effective)
        z = z_mu + torch.randn_like(z_mu) * torch.exp(0.5 * z_logvar) if sample else z_mu
        task = self._st_gate(self.task_mask())
        selected = z * task[None, :]
        if self.direct_bypass:
            stacked_embeddings = torch.stack(list(embeddings), dim=1)
            observed_count = effective.sum(dim=1, keepdim=True)
            pooled = (stacked_embeddings * effective[:, :, None]).sum(dim=1) / observed_count.clamp_min(1.0)
            bypass = self.bypass_projection(pooled) * (observed_count > 0).to(pooled.dtype)
            selected = selected + bypass
        logits = self.classifier(selected)
        incidence = self._st_gate(self.incidence())
        reconstructions: List[torch.Tensor] = []
        for index, decoder in enumerate(self.decoders):
            decoder_inputs = [z * incidence[index][None, :]]
            if self.use_private:
                u = u_mus[index]
                if sample:
                    u = u + torch.randn_like(u) * torch.exp(0.5 * u_logvars[index])
                decoder_inputs.append(u)
            reconstructions.append(decoder(torch.cat(decoder_inputs, dim=-1)))
        return {
            "logits": logits,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "embeddings": embeddings,
            "reconstructions": reconstructions,
            "effective_availability": effective,
        }

    def forward(
        self,
        image: torch.Tensor,
        groups: Sequence[torch.Tensor],
        availability: torch.Tensor,
        sample: bool = True,
        modality_dropout: float = 0.0,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        embeddings = self.stem(image, groups)
        return self.forward_from_embeddings(
            embeddings,
            availability,
            sample=sample,
            modality_dropout=modality_dropout,
        )

    def regularization(self) -> Dict[str, torch.Tensor]:
        incidence = self.incidence()
        task = self.task_mask()
        witness = (F.relu(2.0 - incidence.sum(dim=0)).pow(2) * task).mean()
        task_floor = F.relu(4.0 - task.sum()).pow(2) / 16.0
        return {
            "incidence_sparsity": incidence.mean(),
            "task_sparsity": task.mean(),
            "sparsity": incidence.mean() + task.mean(),
            "witness": witness,
            "task_floor": task_floor,
        }


class MultimodalTransformer(nn.Module):
    def __init__(self, group_dims: Sequence[int], hidden: int = 128, layers: int = 3, classes: int = 3):
        super().__init__()
        self.modality_count = 1 + len(group_dims)
        self.stem = ModalityStem(group_dims, hidden)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.modality_embedding = nn.Parameter(torch.randn(1, self.modality_count, hidden) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=4 * hidden,
            dropout=0.10,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, classes))

    def forward(self, image: torch.Tensor, groups: Sequence[torch.Tensor], availability: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack(self.stem(image, groups), dim=1) + self.modality_embedding
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        sequence = torch.cat([cls, tokens], dim=1)
        visible_cls = torch.zeros((availability.shape[0], 1), dtype=torch.bool, device=availability.device)
        padding_mask = torch.cat([visible_cls, availability < 0.5], dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=padding_mask)
        return self.head(encoded[:, 0])
