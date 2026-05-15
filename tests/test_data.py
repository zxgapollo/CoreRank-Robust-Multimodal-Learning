from __future__ import annotations

import numpy as np
import pytest

from corerank_synth.data import SyntheticConfig, make_core_graph, make_footprint, make_synthetic_data


def test_default_complementary_footprint_matches_design() -> None:
    cfg = SyntheticConfig(scenario="complementary", z_dim=6, n_modalities=3)

    fp = make_footprint(cfg)

    expected = np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 1, 0, 0, 1, 1],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(fp, expected)


def test_redundant_footprint_duplicates_first_modality() -> None:
    cfg = SyntheticConfig(scenario="redundant", z_dim=6, n_modalities=3)

    fp = make_footprint(cfg)

    np.testing.assert_array_equal(fp[0], fp[2])


def test_core_graph_is_acyclic_and_nonzero() -> None:
    cfg = SyntheticConfig(scenario="complementary", z_dim=6, n_modalities=3, core_graph_strength=0.4)

    graph = make_core_graph(cfg)

    assert graph.shape == (cfg.z_dim, cfg.z_dim)
    assert np.count_nonzero(graph) > 0
    assert np.allclose(np.triu(graph), 0.0)


def test_biased_scenario_biases_only_configured_modality() -> None:
    cfg = SyntheticConfig(
        scenario="biased",
        seed=11,
        n_train=256,
        n_val=64,
        n_test=64,
        bias_strength=3.0,
        biased_modality=1,
        standardize=False,
    )

    train, _, _, params = make_synthetic_data(cfg)
    y = train.y.numpy().reshape(-1)
    y_sign = 2.0 * y - 1.0
    corr_to_label = [
        abs(np.corrcoef(train.x[m].numpy() @ params.bias_vec[m], y_sign)[0, 1])
        for m in range(cfg.n_modalities)
    ]

    assert corr_to_label[1] > max(corr_to_label[0], corr_to_label[2])


def test_domain_scenario_shifts_only_configured_modality() -> None:
    cfg = SyntheticConfig(
        scenario="domain",
        seed=23,
        n_train=256,
        n_val=64,
        n_test=256,
        domain_shift_strength=3.0,
        domain_shifted_modality=2,
        standardize=False,
    )

    train, _, test, params = make_synthetic_data(cfg)
    shifts = []
    for m in range(cfg.n_modalities):
        direction = params.domain_vec[m]
        train_projection = train.x[m].numpy() @ direction
        test_projection = test.x[m].numpy() @ direction
        shifts.append(abs(float(test_projection.mean() - train_projection.mean())))

    assert shifts[2] > max(shifts[0], shifts[1]) + 1.0


def test_config_rejects_invalid_bias_modality() -> None:
    with pytest.raises(ValueError):
        SyntheticConfig(n_modalities=2, biased_modality=2)


def test_config_rejects_invalid_domain_modality() -> None:
    with pytest.raises(ValueError):
        SyntheticConfig(n_modalities=2, domain_shifted_modality=2)
