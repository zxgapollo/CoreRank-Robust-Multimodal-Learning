from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """METRE public-code TCN residual block (Bai et al. implementation)."""

    def __init__(
        self,
        inputs: int,
        outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.network = nn.Sequential(
            weight_norm(nn.Conv1d(inputs, outputs, kernel_size, padding=padding, dilation=dilation)),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Conv1d(outputs, outputs, kernel_size, padding=padding, dilation=dilation)),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(inputs, outputs, 1) if inputs != outputs else None
        self.relu = nn.ReLU()
        self._initialize()

    def _initialize(self) -> None:
        for module in self.network:
            if isinstance(module, nn.Conv1d):
                module.weight.data.normal_(0.0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0.0, 0.01)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = self.network(values)
        residual = values if self.downsample is None else self.downsample(values)
        return self.relu(output + residual)


class METRETemporalConv(nn.Module):
    """Official METRE mortality TCN architecture with per-hour logits."""

    def __init__(
        self,
        inputs: int = 200,
        channels: Sequence[int] = (256, 256, 256, 256),
        kernel_size: int = 3,
        dropout: float = 0.2,
        classes: int = 2,
    ):
        super().__init__()
        blocks = []
        for level, outputs in enumerate(channels):
            block_inputs = inputs if level == 0 else channels[level - 1]
            blocks.append(
                TemporalBlock(block_inputs, outputs, kernel_size, 2 ** level, dropout)
            )
        self.temporal = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, classes),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.temporal(values).transpose(1, 2).contiguous()
        return self.classifier(encoded)

