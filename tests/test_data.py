from __future__ import annotations

import numpy as np
import pytest

from corerank_synth.data import SyntheticConfig, make_footprint, make_synthetic_data


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


def test_config_rejects_invalid_bias_modality() -> None:
    with pytest.raises(ValueError):
        SyntheticConfig(n_modalities=2, biased_modality=2)
