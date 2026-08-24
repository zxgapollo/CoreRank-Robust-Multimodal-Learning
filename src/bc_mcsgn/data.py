from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import torch


SPLIT_NAMES = (
    "train",
    "val",
    "test_id",
    "test_concept_shift",
    "test_domain_shift",
    "test_missing_information",
)
OOD_SPLITS = ("test_concept_shift", "test_domain_shift", "test_missing_information")


@dataclass
class SCMConfig:
    seed: int = 0
    n_train: int = 20_000
    n_val: int = 5_000
    n_test: int = 5_000
    k: int = 6
    n_modalities: int = 4
    x_dim: int = 10
    u_dim: int = 2
    hidden_gen_dim: int = 24
    noise_std: float = 0.15
    proto_noise_std: float = 0.10
    alpha: float = 1.5
    concept_shortcut_scale: float = 0.0
    domain_noise_scale: float = 2.5
    domain_tail_df: float = 3.5
    domain_style_angle_degrees: float = 120.0
    domain_action_scale: float = 1.25
    missing_base: float = 0.95
    missing_gap: float = 0.0
    missing_certified_fraction: float = 0.50
    additive_u_strength: float = 0.70
    multiplicative_u_strength: float = 0.30
    delta_strength: float = 0.85
    standardize_x: bool = True

    def __post_init__(self) -> None:
        if self.k != 6:
            raise ValueError("The first BC-MCSGN SCM milestone fixes k=6 to match the design md.")
        if self.n_modalities != 4:
            raise ValueError("The first BC-MCSGN SCM milestone fixes m=4 modalities.")
        for name in ("n_train", "n_val", "n_test", "x_dim", "u_dim", "hidden_gen_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("noise_std", "proto_noise_std", "alpha", "domain_noise_scale", "domain_action_scale"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        if not -1.0 <= self.concept_shortcut_scale <= 1.0:
            raise ValueError("concept_shortcut_scale must be in [-1, 1]")
        if self.domain_tail_df <= 2.0:
            raise ValueError("domain_tail_df must exceed 2 so residual variance is finite")
        if not 0.0 <= self.missing_base <= 1.0:
            raise ValueError("missing_base must be in [0, 1]")
        if not 0.0 <= self.missing_gap <= 1.0:
            raise ValueError("missing_gap must be in [0, 1]")
        if not 0.0 <= self.missing_certified_fraction <= 1.0:
            raise ValueError("missing_certified_fraction must be in [0, 1]")

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SCMParams:
    graph: np.ndarray
    state_mask: np.ndarray
    modality_mask: np.ndarray
    label_weights: np.ndarray
    alpha_vec: np.ndarray
    u_label_dirs: List[np.ndarray]
    gen_w1: List[np.ndarray]
    gen_b1: List[np.ndarray]
    gen_w2: List[np.ndarray]
    gen_b2: List[np.ndarray]
    u_add: List[np.ndarray]
    u_mul: List[np.ndarray]
    delta_w1: np.ndarray
    delta_b1: np.ndarray
    delta_w2: np.ndarray
    delta_b2: np.ndarray


@dataclass
class BCSplit:
    x: List[torch.Tensor]
    x_intervened: List[torch.Tensor]
    y: torch.Tensor
    z: torch.Tensor
    s: torch.Tensor
    u: List[torch.Tensor]
    delta: torch.Tensor
    s_tilde: torch.Tensor
    obs_mask: torch.Tensor


class BCDataset(torch.utils.data.Dataset):
    def __init__(self, split: BCSplit):
        self.split = split
        self.n = split.y.shape[0]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        return {
            "x": [xm[idx] for xm in self.split.x],
            "x_intervened": [xm[idx] for xm in self.split.x_intervened],
            "y": self.split.y[idx],
            "z": self.split.z[idx],
            "s": self.split.s[idx],
            "u": [um[idx] for um in self.split.u],
            "delta": self.split.delta[idx],
            "s_tilde": self.split.s_tilde[idx],
            "obs_mask": self.split.obs_mask[idx],
        }


def collate_bc_batch(items: List[Dict[str, torch.Tensor | List[torch.Tensor]]]) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
    m = len(items[0]["x"])  # type: ignore[arg-type]
    return {
        "x": [torch.stack([it["x"][j] for it in items], dim=0).float() for j in range(m)],  # type: ignore[index]
        "x_intervened": [torch.stack([it["x_intervened"][j] for it in items], dim=0).float() for j in range(m)],  # type: ignore[index]
        "y": torch.stack([it["y"] for it in items], dim=0).float(),  # type: ignore[arg-type]
        "z": torch.stack([it["z"] for it in items], dim=0).float(),  # type: ignore[arg-type]
        "s": torch.stack([it["s"] for it in items], dim=0).float(),  # type: ignore[arg-type]
        "u": [torch.stack([it["u"][j] for it in items], dim=0).float() for j in range(m)],  # type: ignore[index]
        "delta": torch.stack([it["delta"] for it in items], dim=0).float(),  # type: ignore[arg-type]
        "s_tilde": torch.stack([it["s_tilde"] for it in items], dim=0).float(),  # type: ignore[arg-type]
        "obs_mask": torch.stack([it["obs_mask"] for it in items], dim=0).float(),  # type: ignore[arg-type]
    }


def true_graph() -> np.ndarray:
    graph = np.zeros((6, 6), dtype=np.float32)
    for child, parent, weight in [
        (2, 0, 1.2),
        (2, 1, -0.8),
        (4, 2, 1.0),
        (4, 3, 0.5),
        (5, 4, 1.0),
    ]:
        graph[child, parent] = weight
    return graph


def true_state_mask() -> np.ndarray:
    mask = np.zeros(6, dtype=np.float32)
    mask[[2, 4, 5]] = 1.0
    return mask


def true_label_weights() -> np.ndarray:
    weights = np.zeros(6, dtype=np.float32)
    weights[[2, 4, 5]] = np.array([1.5, 1.2, -0.8], dtype=np.float32)
    return weights


def true_modality_mask() -> np.ndarray:
    mask = np.zeros((4, 6), dtype=np.float32)
    # One-indexed design: Gamma_1={3,5}, Gamma_2={2,5,6}, Gamma_3={1,4}, Gamma_4={3,6}.
    mask[0, [2, 4]] = 1.0
    mask[1, [1, 4, 5]] = 1.0
    mask[2, [0, 3]] = 1.0
    mask[3, [2, 5]] = 1.0
    return mask


def _unit_rows(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def _make_params(cfg: SCMConfig) -> SCMParams:
    rng = np.random.default_rng(cfg.seed + 991)
    modality_mask = true_modality_mask()
    gen_w1: List[np.ndarray] = []
    gen_b1: List[np.ndarray] = []
    gen_w2: List[np.ndarray] = []
    gen_b2: List[np.ndarray] = []
    u_add: List[np.ndarray] = []
    u_mul: List[np.ndarray] = []
    u_dirs: List[np.ndarray] = []

    for m in range(cfg.n_modalities):
        w1 = rng.normal(0, 0.8 / np.sqrt(cfg.k), size=(cfg.k, cfg.hidden_gen_dim)).astype(np.float32)
        w1 *= modality_mask[m, :, None]
        w2 = rng.normal(0, 0.9 / np.sqrt(cfg.hidden_gen_dim), size=(cfg.hidden_gen_dim, cfg.x_dim)).astype(np.float32)
        gen_w1.append(w1)
        gen_b1.append(rng.normal(0, 0.15, size=(cfg.hidden_gen_dim,)).astype(np.float32))
        gen_w2.append(w2)
        gen_b2.append(rng.normal(0, 0.08, size=(cfg.x_dim,)).astype(np.float32))
        u_add.append(rng.normal(0, 0.8 / np.sqrt(cfg.u_dim), size=(cfg.u_dim, cfg.x_dim)).astype(np.float32))
        u_mul.append(rng.normal(0, 0.7 / np.sqrt(cfg.u_dim), size=(cfg.u_dim, cfg.x_dim)).astype(np.float32))
        # Keep the shortcut direction positively oriented so diagnostic
        # correlations have a stable sign across seeds and modalities.
        raw_dir = np.abs(rng.normal(0, 1, size=(1, cfg.u_dim)).astype(np.float32)) + 0.1
        u_dirs.append(_unit_rows(raw_dir)[0])

    delta_hidden = max(16, cfg.hidden_gen_dim)
    delta_w1 = rng.normal(0, 0.7 / np.sqrt(cfg.n_modalities * cfg.u_dim), size=(cfg.n_modalities * cfg.u_dim, delta_hidden)).astype(np.float32)
    delta_b1 = rng.normal(0, 0.08, size=(delta_hidden,)).astype(np.float32)
    delta_w2 = rng.normal(0, 0.8 / np.sqrt(delta_hidden), size=(delta_hidden, cfg.k)).astype(np.float32)
    delta_b2 = rng.normal(0, 0.05, size=(cfg.k,)).astype(np.float32)
    alpha_vec = np.array([1.00, 0.80, 1.15, 0.95], dtype=np.float32) * cfg.alpha

    return SCMParams(
        graph=true_graph(),
        state_mask=true_state_mask(),
        modality_mask=modality_mask,
        label_weights=true_label_weights(),
        alpha_vec=alpha_vec,
        u_label_dirs=u_dirs,
        gen_w1=gen_w1,
        gen_b1=gen_b1,
        gen_w2=gen_w2,
        gen_b2=gen_b2,
        u_add=u_add,
        u_mul=u_mul,
        delta_w1=delta_w1,
        delta_b1=delta_b1,
        delta_w2=delta_w2,
        delta_b2=delta_b2,
    )


def _sample_latents(n: int, rng: np.random.Generator) -> np.ndarray:
    eps = rng.normal(0, 1, size=(n, 6)).astype(np.float32)
    z = np.zeros((n, 6), dtype=np.float32)
    z[:, 0] = eps[:, 0]
    z[:, 1] = eps[:, 1]
    z[:, 3] = eps[:, 3]
    z[:, 2] = np.tanh(1.2 * z[:, 0] - 0.8 * z[:, 1]) + 0.2 * eps[:, 2]
    z[:, 4] = np.sin(z[:, 2]) + 0.5 * z[:, 3] + 0.2 * eps[:, 4]
    z[:, 5] = np.tanh(z[:, 4]) + 0.2 * eps[:, 5]
    return z.astype(np.float32)


def _sample_labels(z: np.ndarray, label_weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    logits = z @ label_weights
    p = 1.0 / (1.0 + np.exp(-logits))
    return rng.binomial(1, p).astype(np.float32)


def _split_to_regime(split_name: str) -> str:
    if split_name in {"train", "val", "test_id"}:
        return "id"
    return split_name.replace("test_", "")


def _orthogonal_nuisance_dir(label_dir: np.ndarray) -> np.ndarray:
    """A deterministic nuisance direction that is not the label shortcut direction."""
    direction = np.roll(label_dir.astype(np.float32), 1)
    direction = direction - float(direction @ label_dir) * label_dir
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.ones_like(label_dir, dtype=np.float32)
        direction = direction - float(direction @ label_dir) * label_dir
        norm = float(np.linalg.norm(direction))
    return (direction / (norm + 1e-8)).astype(np.float32)


def _sample_private_residual(
    regime: str,
    y_sign: np.ndarray,
    alpha: float,
    label_dir: np.ndarray,
    cfg: SCMConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample U while keeping the invariant latent/label SCM untouched."""
    coef = cfg.concept_shortcut_scale * alpha if regime == "concept_shift" else alpha
    if regime == "domain_shift":
        variance_normalizer = np.sqrt(cfg.domain_tail_df / (cfg.domain_tail_df - 2.0))
        eta = rng.standard_t(cfg.domain_tail_df, size=(y_sign.shape[0], cfg.u_dim)).astype(np.float32)
        eta = cfg.domain_noise_scale * eta / variance_normalizer
    else:
        eta = rng.normal(0, 1, size=(y_sign.shape[0], cfg.u_dim)).astype(np.float32)
    return (eta + coef * y_sign[:, None] * label_dir[None, :]).astype(np.float32)


def _domain_style_matrix(cfg: SCMConfig) -> np.ndarray:
    """Fixed environment-specific rotation of private residual coordinates."""
    matrix = np.eye(cfg.u_dim, dtype=np.float32)
    if cfg.u_dim == 1:
        matrix[0, 0] = -1.0
        return matrix
    angle = np.deg2rad(cfg.domain_style_angle_degrees)
    c, s = np.cos(angle), np.sin(angle)
    matrix[:2, :2] = np.array([[c, -s], [s, c]], dtype=np.float32)
    return matrix


def _residual_action(u: np.ndarray, regime: str, cfg: SCMConfig) -> np.ndarray:
    if regime != "domain_shift":
        return u
    return (cfg.domain_action_scale * (u @ _domain_style_matrix(cfg))).astype(np.float32)


def _sample_observation_mask(
    regime: str,
    y: np.ndarray,
    cfg: SCMConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample ordinary or witness-breaking modality availability.

    The missing-information environment mixes fully observed, structurally
    certified samples with the fixed [1,1,0,0] pattern that loses a finite
    support witness. Selection is independent of Y, so missingness itself is
    not a label shortcut.
    """
    n = y.shape[0]
    if regime == "missing_information":
        certified = rng.uniform(size=n) < cfg.missing_certified_fraction
        full = np.ones((n, cfg.n_modalities), dtype=np.float32)
        witness_breaking = np.tile(np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32), (n, 1))
        return np.where(certified[:, None], full, witness_breaking).astype(np.float32)

    y_flat = y.reshape(-1)
    offsets = np.array([0.02, -0.02, 0.00, -0.04], dtype=np.float32)
    columns: List[np.ndarray] = []
    for modality in range(cfg.n_modalities):
        p = cfg.missing_base + offsets[modality] + cfg.missing_gap * y_flat
        p = np.clip(p, 0.05, 0.99)
        columns.append(rng.binomial(1, p).astype(np.float32)[:, None])
    return np.concatenate(columns, axis=1)


def _fixed_mlp(z: np.ndarray, params: SCMParams, modality: int) -> np.ndarray:
    h = np.tanh(z @ params.gen_w1[modality] + params.gen_b1[modality])
    return np.tanh(h @ params.gen_w2[modality] + params.gen_b2[modality]).astype(np.float32)


def _delta_from_u(us: List[np.ndarray], cfg: SCMConfig, params: SCMParams) -> np.ndarray:
    u_flat = np.concatenate(us, axis=1)
    h = np.tanh(u_flat @ params.delta_w1 + params.delta_b1)
    delta = h @ params.delta_w2 + params.delta_b2
    return (cfg.delta_strength * delta).astype(np.float32)


def _generate_split(n: int, split_name: str, cfg: SCMConfig, params: SCMParams, rng: np.random.Generator) -> BCSplit:
    regime = _split_to_regime(split_name)
    z = _sample_latents(n, rng)
    y = _sample_labels(z, params.label_weights, rng)
    y_sign = (2.0 * y - 1.0).astype(np.float32)
    s = (z * params.state_mask[None, :]).astype(np.float32)

    xs: List[np.ndarray] = []
    xs_intervened: List[np.ndarray] = []
    us: List[np.ndarray] = []
    obs_mask = _sample_observation_mask(regime, y, cfg, rng)
    for m in range(cfg.n_modalities):
        u = _sample_private_residual(
            regime,
            y_sign,
            float(params.alpha_vec[m]),
            params.u_label_dirs[m],
            cfg,
            rng,
        )
        u_action = _residual_action(u, regime, cfg)
        x_bar = _fixed_mlp(z, params, m)
        additive = u_action @ params.u_add[m]
        multiplicative = u_action @ params.u_mul[m]
        noise = rng.normal(0, cfg.noise_std, size=(n, cfg.x_dim)).astype(np.float32)
        x = x_bar + cfg.additive_u_strength * additive + cfg.multiplicative_u_strength * multiplicative * x_bar + noise
        # Paired residual intervention: the same Z and invariant core generator,
        # with a fresh label-independent private residual. It supplies the
        # extra information required for raw content/private separation; it is
        # never used to alter the label or the core structure.
        u_intervened = rng.normal(0, 1, size=(n, cfg.u_dim)).astype(np.float32)
        u_intervened_action = _residual_action(u_intervened, regime, cfg)
        additive_intervened = u_intervened_action @ params.u_add[m]
        multiplicative_intervened = u_intervened_action @ params.u_mul[m]
        noise_intervened = rng.normal(0, cfg.noise_std, size=(n, cfg.x_dim)).astype(np.float32)
        x_intervened = (
            x_bar
            + cfg.additive_u_strength * additive_intervened
            + cfg.multiplicative_u_strength * multiplicative_intervened * x_bar
            + noise_intervened
        )
        xs.append(x.astype(np.float32))
        xs_intervened.append(x_intervened.astype(np.float32))
        us.append(u.astype(np.float32))

    delta = _delta_from_u(us, cfg, params)
    s_tilde = s + delta + rng.normal(0, cfg.proto_noise_std, size=s.shape).astype(np.float32)
    return BCSplit(
        x=[torch.tensor(x) for x in xs],
        x_intervened=[torch.tensor(x) for x in xs_intervened],
        y=torch.tensor(y[:, None]),
        z=torch.tensor(z),
        s=torch.tensor(s),
        u=[torch.tensor(u) for u in us],
        delta=torch.tensor(delta),
        s_tilde=torch.tensor(s_tilde.astype(np.float32)),
        obs_mask=torch.tensor(obs_mask),
    )


def _standardize_x(splits: Mapping[str, BCSplit]) -> None:
    train = splits["train"]
    for m in range(len(train.x)):
        mean = train.x[m].mean(dim=0, keepdim=True)
        std = train.x[m].std(dim=0, keepdim=True).clamp_min(1e-5)
        for split in splits.values():
            split.x[m] = (split.x[m] - mean) / std
            split.x[m] = split.x[m] * split.obs_mask[:, m : m + 1]
            split.x_intervened[m] = (split.x_intervened[m] - mean) / std
            split.x_intervened[m] = split.x_intervened[m] * split.obs_mask[:, m : m + 1]


def make_scm_dataset(cfg: SCMConfig) -> Tuple[Dict[str, BCSplit], SCMParams]:
    params = _make_params(cfg)
    rng = np.random.default_rng(cfg.seed + 2027)
    sizes = {
        "train": cfg.n_train,
        "val": cfg.n_val,
        "test_id": cfg.n_test,
        "test_concept_shift": cfg.n_test,
        "test_domain_shift": cfg.n_test,
        "test_missing_information": cfg.n_test,
    }
    splits = {name: _generate_split(n, name, cfg, params, rng) for name, n in sizes.items()}
    if cfg.standardize_x:
        _standardize_x(splits)
    return splits, params


def _params_to_arrays(params: SCMParams) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {
        "param_graph": params.graph,
        "param_state_mask": params.state_mask,
        "param_modality_mask": params.modality_mask,
        "param_label_weights": params.label_weights,
        "param_alpha_vec": params.alpha_vec,
        "param_delta_w1": params.delta_w1,
        "param_delta_b1": params.delta_b1,
        "param_delta_w2": params.delta_w2,
        "param_delta_b2": params.delta_b2,
    }
    for m, arr in enumerate(params.u_label_dirs):
        arrays[f"param_u_label_dir_{m}"] = arr
    for prefix, seq in [
        ("gen_w1", params.gen_w1),
        ("gen_b1", params.gen_b1),
        ("gen_w2", params.gen_w2),
        ("gen_b2", params.gen_b2),
        ("u_add", params.u_add),
        ("u_mul", params.u_mul),
    ]:
        for m, arr in enumerate(seq):
            arrays[f"param_{prefix}_{m}"] = arr
    return arrays


def _arrays_to_params(arrays: Mapping[str, np.ndarray], cfg: SCMConfig) -> SCMParams:
    return SCMParams(
        graph=arrays["param_graph"].astype(np.float32),
        state_mask=arrays["param_state_mask"].astype(np.float32),
        modality_mask=arrays["param_modality_mask"].astype(np.float32),
        label_weights=arrays.get("param_label_weights", true_label_weights()).astype(np.float32),
        alpha_vec=arrays["param_alpha_vec"].astype(np.float32),
        u_label_dirs=[arrays[f"param_u_label_dir_{m}"].astype(np.float32) for m in range(cfg.n_modalities)],
        gen_w1=[arrays[f"param_gen_w1_{m}"].astype(np.float32) for m in range(cfg.n_modalities)],
        gen_b1=[arrays[f"param_gen_b1_{m}"].astype(np.float32) for m in range(cfg.n_modalities)],
        gen_w2=[arrays[f"param_gen_w2_{m}"].astype(np.float32) for m in range(cfg.n_modalities)],
        gen_b2=[arrays[f"param_gen_b2_{m}"].astype(np.float32) for m in range(cfg.n_modalities)],
        u_add=[arrays[f"param_u_add_{m}"].astype(np.float32) for m in range(cfg.n_modalities)],
        u_mul=[arrays[f"param_u_mul_{m}"].astype(np.float32) for m in range(cfg.n_modalities)],
        delta_w1=arrays["param_delta_w1"].astype(np.float32),
        delta_b1=arrays["param_delta_b1"].astype(np.float32),
        delta_w2=arrays["param_delta_w2"].astype(np.float32),
        delta_b2=arrays["param_delta_b2"].astype(np.float32),
    )


def save_fixed_dataset(path: str | Path, cfg: SCMConfig) -> Tuple[Dict[str, BCSplit], SCMParams]:
    splits, params = make_scm_dataset(cfg)
    arrays: Dict[str, np.ndarray] = {
        "config_json": np.array(json.dumps(cfg.to_dict())),
    }
    arrays.update(_params_to_arrays(params))
    for split_name, split in splits.items():
        arrays[f"{split_name}_y"] = split.y.numpy()
        arrays[f"{split_name}_z"] = split.z.numpy()
        arrays[f"{split_name}_s"] = split.s.numpy()
        arrays[f"{split_name}_delta"] = split.delta.numpy()
        arrays[f"{split_name}_s_tilde"] = split.s_tilde.numpy()
        arrays[f"{split_name}_obs_mask"] = split.obs_mask.numpy()
        for m, x in enumerate(split.x):
            arrays[f"{split_name}_x_{m}"] = x.numpy()
        for m, x in enumerate(split.x_intervened):
            arrays[f"{split_name}_x_intervened_{m}"] = x.numpy()
        for m, u in enumerate(split.u):
            arrays[f"{split_name}_u_{m}"] = u.numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return splits, params


def _load_split(name: str, arrays: Mapping[str, np.ndarray], cfg: SCMConfig) -> BCSplit:
    return BCSplit(
        x=[torch.tensor(arrays[f"{name}_x_{m}"].astype(np.float32)) for m in range(cfg.n_modalities)],
        x_intervened=[torch.tensor(arrays[f"{name}_x_intervened_{m}"].astype(np.float32)) for m in range(cfg.n_modalities)],
        y=torch.tensor(arrays[f"{name}_y"].astype(np.float32)),
        z=torch.tensor(arrays[f"{name}_z"].astype(np.float32)),
        s=torch.tensor(arrays[f"{name}_s"].astype(np.float32)),
        u=[torch.tensor(arrays[f"{name}_u_{m}"].astype(np.float32)) for m in range(cfg.n_modalities)],
        delta=torch.tensor(arrays[f"{name}_delta"].astype(np.float32)),
        s_tilde=torch.tensor(arrays[f"{name}_s_tilde"].astype(np.float32)),
        obs_mask=torch.tensor(arrays[f"{name}_obs_mask"].astype(np.float32)),
    )


def load_fixed_dataset(path: str | Path) -> Tuple[Dict[str, BCSplit], SCMParams, SCMConfig]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    cfg = SCMConfig(**json.loads(str(arrays["config_json"].item())))
    params = _arrays_to_params(arrays, cfg)
    splits = {name: _load_split(name, arrays, cfg) for name in SPLIT_NAMES}
    return splits, params, cfg


def split_u_label_correlations(split: BCSplit) -> List[float]:
    y = (2.0 * split.y.numpy().reshape(-1) - 1.0).astype(np.float64)
    out: List[float] = []
    for u in split.u:
        score = u.numpy().mean(axis=1)
        out.append(float(np.corrcoef(score, y)[0, 1]))
    return out


def split_missing_label_correlations(split: BCSplit) -> List[float]:
    y = (2.0 * split.y.numpy().reshape(-1) - 1.0).astype(np.float64)
    out: List[float] = []
    for m in range(split.obs_mask.shape[1]):
        observed = split.obs_mask.numpy()[:, m]
        if float(np.std(observed)) < 1e-8:
            out.append(float("nan"))
        else:
            out.append(float(np.corrcoef(observed, y)[0, 1]))
    return out


def split_sizes(splits: Mapping[str, BCSplit]) -> Dict[str, int]:
    return {name: int(split.y.shape[0]) for name, split in splits.items()}


def split_true_certificate_rate(split: BCSplit, params: SCMParams) -> float:
    values = [
        _true_certificate(params.modality_mask, params.state_mask, row)
        for row in split.obs_mask.numpy()
    ]
    return float(np.mean(values))


def audit_environment_shifts(
    splits: Mapping[str, BCSplit],
    params: SCMParams,
    cfg: SCMConfig,
) -> Dict[str, object]:
    """Fail-fast audit for the three requested structure-preserving shifts."""
    train_corr = np.abs(np.asarray(split_u_label_correlations(splits["train"]), dtype=np.float64))
    concept_corr = np.abs(np.asarray(split_u_label_correlations(splits["test_concept_shift"]), dtype=np.float64))
    train_var = float(np.mean([u.numpy().var() for u in splits["train"].u]))
    domain_var = float(np.mean([u.numpy().var() for u in splits["test_domain_shift"].u]))
    missing_rate = split_true_certificate_rate(splits["test_missing_information"], params)
    label_rates = {name: float(split.y.mean()) for name, split in splits.items()}
    label_rate_span = float(max(label_rates.values()) - min(label_rates.values()))
    style_matrix = _domain_style_matrix(cfg)
    style_distance = float(np.linalg.norm(style_matrix - np.eye(cfg.u_dim, dtype=np.float32)))
    passed = (
        float(train_corr.mean()) > 0.30
        and float(concept_corr.max()) < 0.10
        and domain_var > 1.5 * train_var
        and style_distance > 0.25
        and abs(missing_rate - cfg.missing_certified_fraction) < 0.08
        and label_rate_span < 0.06
    )
    return {
        "passed": bool(passed),
        "train_abs_u_label_correlation_mean": float(train_corr.mean()),
        "concept_abs_u_label_correlation_max": float(concept_corr.max()),
        "train_private_residual_variance": train_var,
        "domain_private_residual_variance": domain_var,
        "domain_style_matrix_distance_from_identity": style_distance,
        "missing_true_certificate_rate": missing_rate,
        "missing_target_certificate_rate": cfg.missing_certified_fraction,
        "label_rates": label_rates,
        "label_rate_span": label_rate_span,
    }


def core_structure_fingerprint(params: SCMParams) -> str:
    """Hash every environment-invariant part of the synthetic SCM.

    Nuisance distribution parameters (shortcut strength, variance, and
    missingness) are deliberately excluded. The fingerprint covers P(Z)'s
    structural coefficients, the invariant label mechanism, incidence, the
    core modality generators, and the stable way U enters each modality.
    """
    digest = hashlib.sha256(b"sfm-scm-core-v1")
    arrays: List[np.ndarray] = [
        params.graph,
        params.state_mask,
        params.modality_mask,
        params.label_weights,
    ]
    for sequence in (params.gen_w1, params.gen_b1, params.gen_w2, params.gen_b2, params.u_add, params.u_mul):
        arrays.extend(sequence)
    for array in arrays:
        value = np.ascontiguousarray(array.astype(np.float32))
        digest.update(str(value.shape).encode("utf-8"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _true_certificate(incidence: np.ndarray, task_mask: np.ndarray, observed: np.ndarray) -> float:
    selected = np.flatnonzero(task_mask > 0)
    if selected.size == 0:
        return 0.0
    pair = incidence[:, :, None] * (1.0 - incidence[:, None, :])
    pair = pair * observed[:, None, None]
    witness = pair.max(axis=0)
    values = [witness[j, np.arange(incidence.shape[1]) != j].min() for j in selected]
    return float(np.min(values))


def audit_core_structure(params: SCMParams, cfg: SCMConfig, tolerance: float = 1e-6) -> Dict[str, object]:
    """Executable audit of the shared-structure assumptions used by the theorem."""
    rng = np.random.default_rng(cfg.seed + 4049)
    probe_z = rng.normal(size=(64, cfg.k)).astype(np.float32)
    nonedge_effects: List[float] = []
    active_effects: List[float] = []
    per_modality: Dict[str, Dict[str, float]] = {}
    for m in range(cfg.n_modalities):
        baseline = _fixed_mlp(probe_z, params, m)
        modality_nonedge: List[float] = []
        modality_active: List[float] = []
        for j in range(cfg.k):
            perturbed = probe_z.copy()
            perturbed[:, j] += 0.37
            effect = float(np.max(np.abs(_fixed_mlp(perturbed, params, m) - baseline)))
            if params.modality_mask[m, j] > 0:
                active_effects.append(effect)
                modality_active.append(effect)
            else:
                nonedge_effects.append(effect)
                modality_nonedge.append(effect)
        per_modality[str(m)] = {
            "max_nonedge_effect": float(max(modality_nonedge, default=0.0)),
            "min_active_edge_effect": float(min(modality_active, default=0.0)),
        }

    label_baseline = probe_z @ params.label_weights
    label_irrelevant_effects: List[float] = []
    for j in np.flatnonzero(params.state_mask == 0):
        perturbed = probe_z.copy()
        perturbed[:, j] += 0.37
        label_irrelevant_effects.append(float(np.max(np.abs(perturbed @ params.label_weights - label_baseline))))

    full_observed = np.ones(cfg.n_modalities, dtype=np.float32)
    # Modalities 1 and 2 alone cannot separate Z3 from Z5 in the design.
    witness_missing = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    fingerprint = core_structure_fingerprint(params)
    max_nonedge = float(max(nonedge_effects, default=0.0))
    min_active = float(min(active_effects, default=0.0))
    max_label_irrelevant = float(max(label_irrelevant_effects, default=0.0))
    full_certificate = _true_certificate(params.modality_mask, params.state_mask, full_observed)
    missing_certificate = _true_certificate(params.modality_mask, params.state_mask, witness_missing)
    passed = (
        max_nonedge <= tolerance
        and max_label_irrelevant <= tolerance
        and min_active > tolerance
        and full_certificate == 1.0
        and missing_certificate == 0.0
    )
    return {
        "passed": bool(passed),
        "core_fingerprint": fingerprint,
        "environment_fingerprints": {name: fingerprint for name in SPLIT_NAMES},
        "max_nonedge_core_effect": max_nonedge,
        "min_active_edge_core_effect": min_active,
        "max_irrelevant_label_effect": max_label_irrelevant,
        "full_modality_certificate": full_certificate,
        "witness_missing_certificate": missing_certificate,
        "per_modality": per_modality,
        "environment_changes_only": {
            "test_concept_shift": "all P_e(U_m|Y) shortcut coefficients are removed; P(Y|Z_C) stays fixed",
            "test_domain_shift": "P_e(U_m) becomes scaled heavy-tailed and the private residual action is rotated",
            "test_missing_information": "P_e(O) mixes fully observed and witness-breaking [1,1,0,0] patterns independently of Y",
        },
    }
