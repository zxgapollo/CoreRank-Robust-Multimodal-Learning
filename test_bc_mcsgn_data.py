from __future__ import annotations

import numpy as np

from bc_mcsgn.data import (
    SCMConfig,
    load_fixed_dataset,
    make_scm_dataset,
    save_fixed_dataset,
    split_missing_label_correlations,
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

    assert set(splits) == {"train", "val", "test_id", "test_ood_a", "test_ood_b", "test_ood_c", "test_ood_d"}
    assert splits["train"].x[0].shape == (cfg.n_train, cfg.x_dim)
    assert splits["test_id"].obs_mask.shape == (cfg.n_test, cfg.n_modalities)
    assert splits["test_ood_b"].s_tilde.shape == (cfg.n_test, cfg.k)
    np.testing.assert_array_equal(params.state_mask, true_state_mask())
    np.testing.assert_array_equal(params.modality_mask, true_modality_mask())


def test_ood_residual_regimes_have_expected_label_correlation_pattern() -> None:
    cfg = SCMConfig(seed=8, n_train=2048, n_val=256, n_test=2048, x_dim=8, u_dim=2, alpha=1.8)
    splits, _ = make_scm_dataset(cfg)

    train_corr = np.nanmean(split_u_label_correlations(splits["train"]))
    removed_corr = abs(np.nanmean(split_u_label_correlations(splits["test_ood_a"])))
    reversed_corr = np.nanmean(split_u_label_correlations(splits["test_ood_b"]))

    assert train_corr > 0.35
    assert removed_corr < 0.15
    assert reversed_corr < -0.35


def test_ood_c_amplifies_residual_variance() -> None:
    cfg = SCMConfig(seed=10, n_train=1024, n_val=128, n_test=1024, x_dim=8, u_dim=2, ood_var_scale=3.0)
    splits, _ = make_scm_dataset(cfg)

    train_var = np.mean([u.numpy().var() for u in splits["train"].u])
    ood_var = np.mean([u.numpy().var() for u in splits["test_ood_c"].u])

    assert ood_var > train_var * 2.0


def test_ood_d_reverses_missingness_label_correlation() -> None:
    cfg = SCMConfig(seed=12, n_train=2048, n_val=256, n_test=2048, x_dim=8, u_dim=2)
    splits, _ = make_scm_dataset(cfg)

    train_corr = np.nanmean(split_missing_label_correlations(splits["train"]))
    ood_corr = np.nanmean(split_missing_label_correlations(splits["test_ood_d"]))

    assert train_corr > 0.15
    assert ood_corr < -0.15


def test_save_and_load_fixed_dataset_roundtrip(tmp_path) -> None:
    cfg = small_cfg(seed=17)
    path = tmp_path / "bc_fixed.npz"
    splits, params = save_fixed_dataset(path, cfg)
    loaded, loaded_params, loaded_cfg = load_fixed_dataset(path)

    assert loaded_cfg.seed == cfg.seed
    np.testing.assert_allclose(splits["train"].x[0].numpy(), loaded["train"].x[0].numpy())
    np.testing.assert_allclose(splits["test_ood_b"].s_tilde.numpy(), loaded["test_ood_b"].s_tilde.numpy())
    np.testing.assert_allclose(params.graph, loaded_params.graph)
