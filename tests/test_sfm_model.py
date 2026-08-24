from __future__ import annotations

import numpy as np
import torch
from torch import nn

from bc_mcsgn.data import SCMConfig, audit_core_structure, make_scm_dataset, true_modality_mask, true_state_mask
from bc_mcsgn.models import ModelConfig, MultimodalTransformer, SFMNet
from bc_mcsgn.train import TrainConfig, _stage_schedule


def oracle_model(x_dim: int = 8) -> SFMNet:
    cfg = ModelConfig(n_modalities=4, x_dim=x_dim, k=6, u_dim=2, hidden_dim=16, fixed_structure=True)
    return SFMNet(
        cfg,
        true_task_mask=torch.tensor(true_state_mask()),
        true_incidence=torch.tensor(true_modality_mask()),
    )


def test_synthetic_scm_passes_shared_core_structure_audit() -> None:
    cfg = SCMConfig(seed=2, n_train=64, n_val=32, n_test=32, x_dim=8)
    _, params = make_scm_dataset(cfg)
    audit = audit_core_structure(params, cfg)

    assert audit["passed"] is True
    assert audit["max_nonedge_core_effect"] == 0.0
    assert audit["max_irrelevant_label_effect"] == 0.0
    assert audit["min_active_edge_core_effect"] > 0.0
    assert audit["full_modality_certificate"] == 1.0
    assert audit["witness_missing_certificate"] == 0.0
    assert len(set(audit["environment_fingerprints"].values())) == 1


def test_decoder_has_no_global_state_or_graph_bypass() -> None:
    model = oracle_model()
    assert not hasattr(model, "graph_raw")
    for decoder in model.decoders:
        first = next(module for module in decoder if isinstance(module, nn.Linear))
        assert first.in_features == model.cfg.k + model.cfg.u_dim


def test_decoder_is_exactly_invariant_to_excluded_factor() -> None:
    model = oracle_model()
    z = torch.randn(5, 6)
    us = [torch.randn(5, 2) for _ in range(4)]
    baseline = model.decode(z, us)[0]
    changed = z.clone()
    changed[:, 0] += 100.0  # Z1 is excluded from Gamma_1={Z3,Z5}.
    perturbed = model.decode(changed, us)[0]
    torch.testing.assert_close(baseline, perturbed, rtol=0.0, atol=0.0)


def test_oracle_certificate_detects_missing_witness_modalities() -> None:
    model = oracle_model()
    full, factor_full = model.structure_certificate(torch.ones(4), hard=True)
    missing, factor_missing = model.structure_certificate(torch.tensor([1.0, 1.0, 0.0, 0.0]), hard=True)

    assert float(full) == 1.0
    assert torch.all(factor_full[torch.tensor([2, 4, 5])] == 1.0)
    assert float(missing) == 0.0
    assert torch.any(factor_missing[torch.tensor([2, 4, 5])] == 0.0)


def test_sfm_forward_and_residual_intervention_are_finite() -> None:
    model = oracle_model()
    xs = [torch.randn(7, 8) for _ in range(4)]
    obs = torch.ones(7, 4)
    out = model(xs, obs, sample=False)

    assert out["logits"].shape == (7, 1)
    assert out["z_mu"].shape == (7, 6)
    assert out["certificate"].shape == (7,)
    loss = model.residual_intervention_loss(out["z_mu"], out["u_mu"], obs)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_lightweight_transformer_handles_missing_modalities() -> None:
    cfg = ModelConfig(n_modalities=4, x_dim=8, hidden_dim=16, layers=1)
    model = MultimodalTransformer(cfg)
    xs = [torch.randn(6, 8) for _ in range(4)]
    obs = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    logits = model(xs, obs)

    assert logits.shape == (6, 1)
    assert torch.isfinite(logits).all()


def test_witness_objective_reaches_learned_incidence_and_task_gates() -> None:
    cfg = ModelConfig(n_modalities=4, x_dim=8, k=6, u_dim=2, hidden_dim=16)
    model = SFMNet(cfg)
    loss = model.witness_loss() + model.gate_regularizer()
    loss.backward()

    assert model.incidence_logits.grad is not None
    assert model.task_mask_logits.grad is not None
    assert torch.isfinite(model.incidence_logits.grad).all()
    assert torch.isfinite(model.task_mask_logits.grad).all()


def test_gauge_protected_schedule_preserves_epoch_budget() -> None:
    schedule = _stage_schedule(TrainConfig(correction_epochs=7))
    assert [name for name, _ in schedule] == ["structure", "task", "joint"]
    assert sum(epochs for _, epochs in schedule) == 7
    assert all(epochs > 0 for _, epochs in schedule)
