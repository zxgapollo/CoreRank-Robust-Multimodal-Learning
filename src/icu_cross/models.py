from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .metre_tcn import TemporalBlock


def mlp(in_dim: int, out_dim: int, hidden: int, dropout: float = 0.10) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


class TemporalEncoder(nn.Module):
    """Matched per-modality encoder for 48-hour clinical sequences."""

    def __init__(self, in_dim: int, hidden: int, dropout: float = 0.10):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pool_score = nn.Linear(hidden, 1)
        self.output = nn.LayerNorm(hidden)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        embedded = self.input(values)
        encoded = embedded + self.temporal(embedded.transpose(1, 2)).transpose(1, 2)
        weights = torch.softmax(self.pool_score(encoded).squeeze(-1), dim=1)
        return self.output((encoded * weights[:, :, None]).sum(dim=1))


class SharedMETREModalityStems(nn.Module):
    """Exact METER-style TCN core shared across isolated temporal modalities.

    Each temporal modality keeps its original position in METER's 200-channel
    input and all other channel groups are set to zero.  The four masked views
    are encoded with one shared TCN, so the fusion models receive separate
    modality embeddings without replicating four independent METER backbones.
    """

    def __init__(
        self,
        modality_dims: Sequence[int],
        hidden: int,
        temporal_modalities: Sequence[bool],
        channels: Sequence[int] = (256, 256, 256, 256),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        temporal = tuple(temporal_modalities)
        if len(temporal) != len(modality_dims):
            raise ValueError("temporal_modalities must align with modality_dims")
        if not channels:
            raise ValueError("METER encoder requires at least one temporal block")
        self.temporal_flags = temporal
        self.total_temporal_dim = sum(
            dim for dim, is_temporal in zip(modality_dims, temporal) if is_temporal
        )
        if self.total_temporal_dim != 200:
            raise ValueError(
                "The shared METER encoder requires exactly 200 temporal channels, "
                f"found {self.total_temporal_dim}"
            )

        blocks: list[nn.Module] = []
        block_inputs = self.total_temporal_dim
        for level, block_outputs in enumerate(channels):
            blocks.append(
                TemporalBlock(
                    block_inputs,
                    int(block_outputs),
                    kernel_size,
                    2 ** level,
                    dropout,
                )
            )
            block_inputs = int(block_outputs)
        self.temporal = nn.Sequential(*blocks)
        self.temporal_output = (
            nn.Identity()
            if block_inputs == hidden
            else nn.Sequential(
                nn.Linear(block_inputs, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        )
        self.static_encoders = nn.ModuleDict({
            str(index): mlp(dim, hidden, hidden)
            for index, (dim, is_temporal) in enumerate(zip(modality_dims, temporal))
            if not is_temporal
        })

        offset = 0
        self.temporal_slices: dict[int, tuple[int, int]] = {}
        for index, (dim, is_temporal) in enumerate(zip(modality_dims, temporal)):
            if is_temporal:
                self.temporal_slices[index] = (offset, offset + dim)
                offset += dim

    def forward(self, modalities: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        embeddings: list[torch.Tensor] = []
        for index, values in enumerate(modalities):
            if not self.temporal_flags[index]:
                embeddings.append(self.static_encoders[str(index)](values))
                continue
            start, end = self.temporal_slices[index]
            channel_first = values.transpose(1, 2)
            isolated = F.pad(
                channel_first,
                (0, 0, start, self.total_temporal_dim - end),
            )
            encoded = self.temporal(isolated)[:, :, -1]
            embeddings.append(self.temporal_output(encoded))
        return embeddings


class ModalityStems(nn.Module):
    def __init__(
        self,
        modality_dims: Sequence[int],
        hidden: int,
        temporal_modalities: Sequence[bool] | None = None,
        encoder_kind: str = "matched",
        metre_channels: Sequence[int] = (256, 256, 256, 256),
    ):
        super().__init__()
        temporal = tuple(temporal_modalities or (False,) * len(modality_dims))
        if len(temporal) != len(modality_dims):
            raise ValueError("temporal_modalities must align with modality_dims")
        self.encoder_kind = encoder_kind
        if encoder_kind == "matched":
            self.encoders = nn.ModuleList([
                TemporalEncoder(dim, hidden) if is_temporal else mlp(dim, hidden, hidden)
                for dim, is_temporal in zip(modality_dims, temporal)
            ])
            self.metre = None
        elif encoder_kind == "metre_shared":
            self.encoders = nn.ModuleList()
            self.metre = SharedMETREModalityStems(
                modality_dims,
                hidden,
                temporal,
                channels=metre_channels,
            )
        else:
            raise ValueError(f"Unknown encoder_kind: {encoder_kind}")

    def forward(self, modalities: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if self.metre is not None:
            return self.metre(modalities)
        return [encoder(values) for encoder, values in zip(self.encoders, modalities)]


class SPMNet(nn.Module):
    """Selective product-of-experts network over five aligned ICU modalities."""

    def __init__(
        self,
        modality_dims: Sequence[int],
        hidden: int = 128,
        latent: int = 32,
        private: int = 8,
        temporal_modalities: Sequence[bool] | None = None,
        encoder_kind: str = "matched",
        metre_channels: Sequence[int] = (256, 256, 256, 256),
    ):
        super().__init__()
        self.modality_count = len(modality_dims)
        self.latent = latent
        self.stems = ModalityStems(
            modality_dims,
            hidden,
            temporal_modalities,
            encoder_kind=encoder_kind,
            metre_channels=metre_channels,
        )
        self.content = nn.ModuleList([mlp(hidden, 2 * latent, hidden) for _ in modality_dims])
        self.private = nn.ModuleList([mlp(hidden, 2 * private, hidden) for _ in modality_dims])
        self.decoders = nn.ModuleList([mlp(latent + private, hidden, hidden) for _ in modality_dims])
        self.classifier = mlp(latent, 1, hidden)
        self.incidence_logits = nn.Parameter(torch.full((self.modality_count, latent), 0.85))
        self.task_logits = nn.Parameter(torch.zeros(latent))
        self.prior_precision = 0.10

    def incidence(self) -> torch.Tensor:
        return torch.sigmoid(self.incidence_logits)

    def task_mask(self) -> torch.Tensor:
        return torch.sigmoid(self.task_logits)

    @staticmethod
    def _st_gate(soft: torch.Tensor) -> torch.Tensor:
        hard = (soft >= 0.5).to(soft.dtype)
        return hard.detach() - soft.detach() + soft

    @staticmethod
    def _drop_modalities(availability: torch.Tensor, probability: float) -> torch.Tensor:
        if probability <= 0:
            return availability
        keep = (torch.rand_like(availability) >= probability).to(availability.dtype)
        effective = availability * keep
        empty = effective.sum(dim=1) < 0.5
        if empty.any():
            first_available = availability[empty].argmax(dim=1)
            effective[empty, first_available] = 1.0
        return effective

    def _fuse(
        self,
        mus: torch.Tensor,
        logvars: torch.Tensor,
        availability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        incidence = self._st_gate(self.incidence())
        evidence = availability[:, :, None] * incidence[None, :, :]
        expert_precision = evidence * torch.exp(-logvars)
        precision = self.prior_precision + expert_precision.sum(dim=1)
        z_mu = (expert_precision * mus).sum(dim=1) / precision.clamp_min(1e-8)
        z_logvar = (-torch.log(precision.clamp_min(1e-8))).clamp(-6.0, 3.0)
        return z_mu, z_logvar

    def forward(
        self,
        modalities: Sequence[torch.Tensor],
        availability: torch.Tensor,
        sample: bool = True,
        modality_dropout: float = 0.0,
    ) -> Dict[str, object]:
        embeddings = self.stems(modalities)
        content_params = [head(value) for head, value in zip(self.content, embeddings)]
        private_params = [head(value) for head, value in zip(self.private, embeddings)]
        mus = torch.stack([value.chunk(2, dim=-1)[0] for value in content_params], dim=1)
        logvars = torch.stack([value.chunk(2, dim=-1)[1].clamp(-6.0, 3.0) for value in content_params], dim=1)
        u_mus = [value.chunk(2, dim=-1)[0] for value in private_params]
        u_logvars = [value.chunk(2, dim=-1)[1].clamp(-6.0, 3.0) for value in private_params]
        effective = self._drop_modalities(availability, modality_dropout) if self.training else availability
        z_mu, z_logvar = self._fuse(mus, logvars, effective)
        z = z_mu + torch.randn_like(z_mu) * torch.exp(0.5 * z_logvar) if sample else z_mu
        task = self._st_gate(self.task_mask())
        logits = self.classifier(z * task[None, :]).squeeze(-1)
        incidence = self._st_gate(self.incidence())
        reconstructions: List[torch.Tensor] = []
        for index, decoder in enumerate(self.decoders):
            u = u_mus[index]
            if sample:
                u = u + torch.randn_like(u) * torch.exp(0.5 * u_logvars[index])
            reconstructions.append(decoder(torch.cat([z * incidence[index][None, :], u], dim=-1)))
        return {
            "logits": logits,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "embeddings": embeddings,
            "reconstructions": reconstructions,
            "effective_availability": effective,
        }

    def regularization(self) -> Dict[str, torch.Tensor]:
        incidence = self.incidence()
        task = self.task_mask()
        witness = (F.relu(2.0 - incidence.sum(dim=0)).pow(2) * task).mean()
        task_floor = F.relu(4.0 - task.sum()).pow(2) / 16.0
        return {
            "sparsity": incidence.mean() + task.mean(),
            "witness": witness,
            "task_floor": task_floor,
        }


class MultimodalTransformer(nn.Module):
    """Matched discriminative baseline with one token per ICU modality."""

    def __init__(
        self,
        modality_dims: Sequence[int],
        hidden: int = 128,
        layers: int = 3,
        temporal_modalities: Sequence[bool] | None = None,
        encoder_kind: str = "matched",
        metre_channels: Sequence[int] = (256, 256, 256, 256),
    ):
        super().__init__()
        self.modality_count = len(modality_dims)
        self.stems = ModalityStems(
            modality_dims,
            hidden,
            temporal_modalities,
            encoder_kind=encoder_kind,
            metre_channels=metre_channels,
        )
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
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, modalities: Sequence[torch.Tensor], availability: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack(self.stems(modalities), dim=1) + self.modality_embedding
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        sequence = torch.cat([cls, tokens], dim=1)
        cls_visible = torch.zeros((availability.shape[0], 1), dtype=torch.bool, device=availability.device)
        padding_mask = torch.cat([cls_visible, availability < 0.5], dim=1)
        encoded = self.encoder(sequence, src_key_padding_mask=padding_mask)
        return self.head(encoded[:, 0]).squeeze(-1)
