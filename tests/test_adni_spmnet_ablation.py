from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from adni_real.models import SPMNet
from adni_real.run_experiment import multiclass_metrics, resolve_spmnet_configuration, spmnet_loss


def configuration(ablation: str = "full") -> SimpleNamespace:
    return SimpleNamespace(
        ablation=ablation,
        reconstruction_weight=0.20,
        kl_weight=0.002,
        sparsity_weight=0.005,
        witness_weight=0.10,
        task_floor_weight=0.05,
        modality_dropout=0.15,
    )


def batch(batch_size: int = 3) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    image = torch.randn(batch_size, 1, 32, 32, 32)
    groups = [torch.randn(batch_size, 5), torch.randn(batch_size, 7)]
    availability = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    return image, groups, availability


def test_default_model_keeps_original_parameter_layout() -> None:
    model = SPMNet([5, 7], hidden=16, latent=4, private=2)

    assert not hasattr(model, "bypass_projection")
    assert model.incidence_logits.requires_grad
    assert model.task_logits.requires_grad
    for decoder in model.decoders:
        first = next(module for module in decoder if isinstance(module, nn.Linear))
        assert first.in_features == 6


def test_fixed_gate_ablation_uses_true_constants() -> None:
    model = SPMNet([5, 7], hidden=16, latent=4, private=2, incidence_mode="all", task_mode="all")

    assert not model.incidence_logits.requires_grad
    assert not model.task_logits.requires_grad
    torch.testing.assert_close(model.incidence(), torch.ones(3, 4))
    torch.testing.assert_close(model.task_mask(), torch.ones(4))


def test_no_private_removes_private_heads_and_decoder_inputs() -> None:
    model = SPMNet([5, 7], hidden=16, latent=4, private=2, use_private=False)
    image, groups, availability = batch()
    output = model(image, groups, availability, sample=False)

    assert len(model.private) == 0
    assert output["logits"].shape == (3, 3)
    for decoder in model.decoders:
        first = next(module for module in decoder if isinstance(module, nn.Linear))
        assert first.in_features == 4


def test_mean_fusion_is_invariant_to_expert_log_variance_in_the_mean() -> None:
    model = SPMNet([5], hidden=16, latent=2, private=2, fusion="mean", incidence_mode="all")
    mus = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]])
    availability = torch.ones(1, 2)
    low_variance = torch.full_like(mus, -4.0)
    mixed_variance = torch.tensor([[[-4.0, 2.0], [2.0, -4.0]]])

    mean_low, _ = model._fuse(mus, low_variance, availability)
    mean_mixed, _ = model._fuse(mus, mixed_variance, availability)

    torch.testing.assert_close(mean_low, torch.tensor([[3.0, 5.0]]))
    torch.testing.assert_close(mean_low, mean_mixed)


def test_direct_bypass_forward_is_finite() -> None:
    model = SPMNet([5, 7], hidden=16, latent=4, private=2, direct_bypass=True)
    image, groups, availability = batch()
    output = model(image, groups, availability, sample=False)

    assert output["logits"].shape == (3, 3)
    assert torch.isfinite(output["logits"]).all()


def test_ablation_resolution_removes_only_intended_terms() -> None:
    full = resolve_spmnet_configuration(configuration("full"))
    no_task = resolve_spmnet_configuration(configuration("no_task_mask"))
    no_incidence = resolve_spmnet_configuration(configuration("no_incidence"))

    assert full["modality_dropout"] == 0.15
    assert no_task["task_mode"] == "all"
    assert no_task["task_sparsity_weight"] == 0.0
    assert no_task["incidence_sparsity_weight"] == full["incidence_sparsity_weight"]
    assert no_incidence["incidence_mode"] == "all"
    assert no_incidence["incidence_sparsity_weight"] == 0.0
    assert no_incidence["witness_weight"] == 0.0
    assert no_incidence["task_sparsity_weight"] == full["task_sparsity_weight"]


def test_full_loss_matches_original_formula() -> None:
    model = SPMNet([5, 7], hidden=16, latent=4, private=2)
    image, groups, availability = batch()
    output = model(image, groups, availability, sample=False)
    labels = torch.tensor([0, 1, 2])
    class_weights = torch.ones(3)
    weights = resolve_spmnet_configuration(configuration("full"))

    loss, parts = spmnet_loss(model, output, labels, class_weights, weights)
    regularization = model.regularization()
    original = (
        torch.nn.functional.cross_entropy(output["logits"], labels, weight=class_weights)
        + 0.20 * torch.tensor(parts["reconstruction"])
        + 0.002 * torch.tensor(parts["kl"])
        + 0.005 * regularization["sparsity"]
        + 0.10 * regularization["witness"]
        + 0.05 * regularization["task_floor"]
    )

    torch.testing.assert_close(loss, original, rtol=1e-5, atol=1e-6)


def test_multiclass_metrics_include_proper_scoring_rules() -> None:
    labels = np.asarray([0, 1, 2])
    probabilities = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.1, 0.2, 0.7]])

    metrics = multiclass_metrics(labels, probabilities)

    assert 0.0 < metrics["nll"] < 1.0
    assert 0.0 < metrics["brier_multiclass"] < 1.0

