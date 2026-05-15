from __future__ import annotations

import torch

from corerank_synth.train import _make_obs_mask


def test_obs_mask_keeps_at_least_one_modality() -> None:
    mask = _make_obs_mask(batch_size=128, n_modalities=3, device="cpu", drop_prob=1.0)

    assert torch.all(mask.sum(dim=1) == 1.0)
