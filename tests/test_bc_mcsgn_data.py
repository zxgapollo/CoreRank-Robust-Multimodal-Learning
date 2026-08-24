from __future__ import annotations

import numpy as np

from bc_mcsgn.data import (
    SCMConfig,
    audit_environment_shifts,
    load_fixed_dataset,
    make_scm_dataset,
    save_fixed_dataset,
    split_missing_label_correlations,
    split_true_certificate_rate,
    split_u_label_correlations,
    true_graph,
    true_modality_mask,
    true_state_mask,
)


def small_cfg(seed: int = 3) -> SCMConfig:
    return SCMConfig(seed=seed, n_train=512, n_val=128, n_test=256, x_dim=8, u_dim=2)


def test_true_scm_metadata_matches_design_md() -> None:
    graph = true_graph()
    state_mask = true_state_mask()
    modality_mask = true_modality_mask()

    assert graph.shape == (6, 6)
    assert graph[2, 0] != 0
    assert graph[2, 1] != 0
    assert graph[4, 2] != 0
    assert graph[4, 3] != 0
    assert graph[5, 4] != 0
    np.testing.assert_array_equal(np.where(state_mask > 0)[0], np.array([2, 4, 5]))
    np.testing.assert_array_equal(np.where(modality_mask[0] > 0)[0], np.array([2, 4]))
    np.testing.assert_array_equal(np.where(modality_mask[1] > 0)[0], np.array([1, 4, 5]))
    np.testing.assert_array_equal(np.where(modality_mask[2] > 0)[0], np.array([0, 3]))
    np.testing.assert_array_equal(np.where(modality_mask[3] > 0)[0], np.array([2, 5]))


def test_fixed_dataset_contains_id_and_ood_splits() -> None:
    cfg = small_cfg()
    splits, params = make_scm_dataset(cfg)

    assert set(splits) == {
        "train",
        "val",
        "test_id",
        "test_concept_shift",
        "test_domain_shift",
        "test_missing_information",
    }
    assert splits["train"].x[0].shape == (cfg.n_train, cfg.x_dim)
    assert splits["test_id"].obs_mask.shape == (cfg.n_test, cfg.n_modalities)
    assert splits["test_domain_shift"].s_tilde.shape == (cfg.n_test, cfg.k)
    np.testing.assert_array_equal(params.state_mask, true_state_mask())
    np.testing.assert_array_equal(params.modality_mask, true_modality_mask())


def test_concept_shift_removes_all_private_label_shortcuts() -> None:
    cfg = SCMConfig(seed=8, n_train=2048, n_val=256, n_test=2048, x_dim=8, u_dim=2, alpha=1.8)
    splits, _ = make_scm_dataset(cfg)

    train_corr = np.array(split_u_label_correlations(splits["train"]))
    concept_corr = np.array(split_u_label_correlations(splits["test_concept_shift"]))

    assert np.nanmean(train_corr) > 0.35
    assert np.nanmax(np.abs(concept_corr)) < 0.08


def test_domain_shift_is_heavy_scaled_and_keeps_shortcut_source() -> None:
    cfg = SCMConfig(seed=10, n_train=2048, n_val=128, n_test=4096, x_dim=8, u_dim=2, domain_noise_scale=2.5)
    splits, _ = make_scm_dataset(cfg)

    train_vars = np.array([u.numpy().var() for u in splits["train"].u])
    domain_vars = np.array([u.numpy().var() for u in splits["test_domain_shift"].u])
    domain_corr = np.array(split_u_label_correlations(splits["test_domain_shift"]))

    assert np.mean(domain_vars / train_vars) > 1.8
    assert np.nanmean(domain_corr) > 0.15


def test_missing_information_mixes_certified_and_witness_breaking_patterns() -> None:
    cfg = SCMConfig(
        seed=12,
        n_train=2048,
        n_val=256,
        n_test=4096,
        x_dim=8,
        u_dim=2,
        missing_certified_fraction=0.50,
    )
    splits, params = make_scm_dataset(cfg)

    missing = splits["test_missing_information"]
    patterns = np.unique(missing.obs_mask.numpy(), axis=0)
    missing_corr = np.array(split_missing_label_correlations(missing))
    certificate_rate = split_true_certificate_rate(missing, params)

    assert {tuple(row.tolist()) for row in patterns} == {(1.0, 1.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)}
    assert np.nanmax(np.abs(missing_corr)) < 0.08
    assert abs(certificate_rate - cfg.missing_certified_fraction) < 0.04


def test_latent_label_distribution_is_stable_across_id_and_ood() -> None:
    cfg = SCMConfig(seed=19, n_train=4096, n_val=512, n_test=4096, x_dim=8, u_dim=2)
    splits, _ = make_scm_dataset(cfg)

    train_y_rate = float(splits["train"].y.mean())
    for name in ["test_id", "test_concept_shift", "test_domain_shift", "test_missing_information"]:
        assert abs(float(splits[name].y.mean()) - train_y_rate) < 0.04
        np.testing.assert_allclose(splits[name].s.numpy(), splits[name].z.numpy() * true_state_mask()[None, :])


def test_three_shift_audit_passes_without_changing_core() -> None:
    cfg = SCMConfig(seed=21, n_train=2048, n_val=256, n_test=2048, x_dim=8, u_dim=2, alpha=1.8)
    splits, params = make_scm_dataset(cfg)

    audit = audit_environment_shifts(splits, params, cfg)
    assert audit["passed"]
    np.testing.assert_array_equal(params.modality_mask, true_modality_mask())


def test_save_and_load_fixed_dataset_roundtrip(tmp_path) -> None:
    cfg = small_cfg(seed=17)
    path = tmp_path / "bc_fixed.npz"
    splits, params = save_fixed_dataset(path, cfg)
    loaded, loaded_params, loaded_cfg = load_fixed_dataset(path)

    assert loaded_cfg.seed == cfg.seed
    np.testing.assert_allclose(splits["train"].x[0].numpy(), loaded["train"].x[0].numpy())
    np.testing.assert_allclose(
        splits["test_domain_shift"].s_tilde.numpy(),
        loaded["test_domain_shift"].s_tilde.numpy(),
    )
    np.testing.assert_allclose(params.graph, loaded_params.graph)
