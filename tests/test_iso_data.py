from __future__ import annotations

import numpy as np

from iso_synth.data import ISODataConfig, SCENARIOS, make_footprint, make_iso_data
from iso_synth.diagnostics import ambiguity_proxy, observability_score


def test_complementary_footprint_covers_state() -> None:
    cfg = ISODataConfig(scenario="complementary", s_dim=6, n_modalities=3)
    fp = make_footprint(cfg)

    assert fp.shape == (3, 6)
    assert np.all(fp.sum(axis=0) > 0)
    assert fp[0].sum() < fp.sum()


def test_redundant_last_modality_duplicates_first() -> None:
    cfg = ISODataConfig(scenario="redundant", s_dim=6, n_modalities=3)
    fp = make_footprint(cfg)

    np.testing.assert_array_equal(fp[0], fp[2])


def test_nuisance_only_last_modality_has_no_state_signal() -> None:
    cfg = ISODataConfig(scenario="nuisance_only", s_dim=6, n_modalities=3)
    fp = make_footprint(cfg)

    assert fp[2].sum() == 0.0
    assert fp[:2].sum() > 0.0


def test_mediated_context_has_context_and_downstream_modalities() -> None:
    cfg = ISODataConfig(scenario="mediated_context", s_dim=6, n_modalities=3)
    fp = make_footprint(cfg)

    assert fp[0, 0] == 1.0
    assert fp[0, 1:].sum() == 0.0
    assert fp[1].sum() > 0.0
    assert fp[2].sum() > fp[0].sum()


def test_default_train_test_are_observation_layer_ood() -> None:
    for scenario in SCENARIOS:
        cfg = ISODataConfig(
            scenario=scenario,
            seed=11,
            n_train=768,
            n_val=128,
            n_test=768,
            ood_residual_shift=0.65,
        )
        train, _, _, test, _ = make_iso_data(cfg)
        source_mean = float(train.u[0][:, 0].mean())
        target_mean = float(test.u[0][:, 0].mean())

        assert target_mean - source_mean > 0.35


def test_observability_monotone_and_ambiguity_decreases() -> None:
    cfg = ISODataConfig(scenario="complementary", seed=3, n_train=32, n_val=16, n_test=16)
    _, _, _, _, params = make_iso_data(cfg)

    lam_0 = observability_score(params, cfg, (0,))
    lam_01 = observability_score(params, cfg, (0, 1))
    lam_all = observability_score(params, cfg, (0, 1, 2))
    amb_0 = ambiguity_proxy(params, cfg, (0,))
    amb_all = ambiguity_proxy(params, cfg, (0, 1, 2))

    assert lam_01 >= lam_0 - 1e-8
    assert lam_all >= lam_01 - 1e-8
    assert amb_all <= amb_0 + 1e-8


def test_shortcut_correlation_flips_between_id_and_ood() -> None:
    cfg = ISODataConfig(
        scenario="shortcut",
        seed=5,
        n_train=512,
        n_val=128,
        n_test=512,
        train_shortcut_corr=0.8,
        test_shortcut_corr=-0.6,
    )
    train, _, _, test, _ = make_iso_data(cfg)

    y_train = 2.0 * train.y.numpy().reshape(-1) - 1.0
    y_test = 2.0 * test.y.numpy().reshape(-1) - 1.0
    corr_train = np.corrcoef(train.q.numpy().reshape(-1), y_train)[0, 1]
    corr_test = np.corrcoef(test.q.numpy().reshape(-1), y_test)[0, 1]

    assert corr_train > 0.55
    assert corr_test < -0.35


def test_noisy_modality_has_larger_noise_std() -> None:
    cfg = ISODataConfig(scenario="noisy_modality", noisy_modality=2, noisy_noise_std=2.0)
    _, _, _, _, params = make_iso_data(cfg)

    assert params.noise_stds[2] == 2.0
    assert params.noise_stds[0] < params.noise_stds[2]
