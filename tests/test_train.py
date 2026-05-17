from __future__ import annotations

import torch

from corerank_synth.models import CoreStructuralSEM
from corerank_synth.train import _make_obs_mask


def test_obs_mask_keeps_at_least_one_modality() -> None:
    mask = _make_obs_mask(batch_size=128, n_modalities=3, device="cpu", drop_prob=1.0)

    assert torch.all(mask.sum(dim=1) == 1.0)


def test_core_structural_graph_masks_diagonal_and_penalizes_cycles() -> None:
    graph = CoreStructuralSEM(z_dim=3, init_scale=0.0)
    with torch.no_grad():
        graph.weight[1, 0] = 0.4
        graph.weight[2, 1] = -0.3

    adj = graph.adjacency()

    assert torch.allclose(torch.diag(adj), torch.zeros(3))
    assert abs(graph.acyclicity().item()) < 1e-6

    with torch.no_grad():
        graph.weight[0, 2] = 0.2

    assert graph.acyclicity().item() > 0.0


def test_core_structural_sem_returns_innovation() -> None:
    graph = CoreStructuralSEM(z_dim=2, init_scale=0.0)
    with torch.no_grad():
        graph.weight[1, 0] = 0.5

    z = torch.tensor([[3.0, 4.0]])
    innovation = graph.innovation(z)

    assert torch.allclose(innovation, torch.tensor([[3.0, 2.5]]))
