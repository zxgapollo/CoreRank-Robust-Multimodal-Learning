from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import torch


SPLIT_NAMES = ("train", "val", "test_id", "test_ood_a", "test_ood_b", "test_ood_c", "test_ood_d")
OOD_SPLITS = ("test_ood_a", "test_ood_b", "test_ood_c", "test_ood_d")


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
    ood_var_scale: float = 2.5
    ood_a_modality: int = 0
    ood_b_modality: int = 1
    ood_c_modality: int = 2
    ood_d_modality: int = 3
    ood_a_alpha_scale: float = 0.35
    ood_b_mean_shift: float = 1.25
    ood_d_base_drop: float = 0.25
    ood_d_gap_scale: float = 0.50
    missing_base: float = 0.75
    missing_gap: float = 0.20
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
        for name in ("noise_std", "proto_noise_std", "alpha", "ood_var_scale"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")
        for name in ("ood_a_modality", "ood_b_modality", "ood_c_modality", "ood_d_modality"):
            value = getattr(self, name)
            if not 0 <= value < self.n_modalities:
                raise ValueError(f"{name} must be in [0, {self.n_modalities}), got {value}")
        for name in ("ood_a_alpha_scale", "ood_d_gap_scale"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.ood_b_mean_shift < 0:
            raise ValueError("ood_b_mean_shift must be non-negative")
        if not 0.0 <= self.ood_d_base_drop <= 1.0:
            raise ValueError("ood_d_base_drop must be in [0, 1]")
        if not 0.0 <= self.missing_base <= 1.0:
            raise ValueError("missing_base must be in [0, 1]")
        if not 0.0 <= self.missing_gap <= 1.0:
            raise ValueError("missing_gap must be in [0, 1]")

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SCMParams:
    graph: np.ndarray
    state_mask: np.ndarray
    modality_mask: np.ndarray
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


def _sample_labels(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    logits = 1.5 * z[:, 2] + 1.2 * z[:, 4] - 0.8 * z[:, 5]
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


def _u_regime_settings(regime: str, alpha: float, modality: int, cfg: SCMConfig) -> Tuple[float, float, float]:
    """Return same-sign label coefficient, noise scale, and independent mean shift.

    OOD here is an observation-layer shift. It never flips labels, never changes
    the latent graph, and never uses a negative label shortcut coefficient.
    """
    coef = alpha
    eta_scale = 1.0
    mean_shift = 0.0
    if regime == "ood_a" and modality == cfg.ood_a_modality:
        coef = cfg.ood_a_alpha_scale * alpha
    if regime == "ood_b" and modality == cfg.ood_b_modality:
        mean_shift = cfg.ood_b_mean_shift
    if regime == "ood_c" and modality == cfg.ood_c_modality:
        eta_scale = cfg.ood_var_scale
    return coef, eta_scale, mean_shift


def _missing_prob(regime: str, y: np.ndarray, cfg: SCMConfig, modality: int) -> np.ndarray:
    y = y.reshape(-1)
    modality_offset = np.array([0.02, -0.02, 0.00, -0.04], dtype=np.float32)[modality]
    if regime == "ood_d" and modality == cfg.ood_d_modality:
        p = cfg.missing_base - cfg.ood_d_base_drop + modality_offset + cfg.ood_d_gap_scale * cfg.missing_gap * y
    else:
        p = cfg.missing_base + modality_offset + cfg.missing_gap * y
    return np.clip(p, 0.05, 0.99).astype(np.float32)


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
    y = _sample_labels(z, rng)
    y_sign = (2.0 * y - 1.0).astype(np.float32)
    s = (z * params.state_mask[None, :]).astype(np.float32)

    xs: List[np.ndarray] = []
    us: List[np.ndarray] = []
    obs_cols: List[np.ndarray] = []
    for m in range(cfg.n_modalities):
        coef, eta_scale, mean_shift = _u_regime_settings(regime, float(params.alpha_vec[m]), m, cfg)
        eta = rng.normal(0, eta_scale, size=(n, cfg.u_dim)).astype(np.float32)
        nuisance_dir = _orthogonal_nuisance_dir(params.u_label_dirs[m])
        u = eta + coef * y_sign[:, None] * params.u_label_dirs[m][None, :] + mean_shift * nuisance_dir[None, :]
        x_bar = _fixed_mlp(z, params, m)
        additive = u @ params.u_add[m]
        multiplicative = u @ params.u_mul[m]
        noise = rng.normal(0, cfg.noise_std, size=(n, cfg.x_dim)).astype(np.float32)
        x = x_bar + cfg.additive_u_strength * additive + cfg.multiplicative_u_strength * multiplicative * x_bar + noise
        p_obs = _missing_prob(regime, y, cfg, m)
        obs = rng.binomial(1, p_obs).astype(np.float32)
        xs.append(x.astype(np.float32))
        us.append(u.astype(np.float32))
        obs_cols.append(obs[:, None])

    delta = _delta_from_u(us, cfg, params)
    s_tilde = s + delta + rng.normal(0, cfg.proto_noise_std, size=s.shape).astype(np.float32)
    obs_mask = np.concatenate(obs_cols, axis=1).astype(np.float32)

    return BCSplit(
        x=[torch.tensor(x) for x in xs],
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


def make_scm_dataset(cfg: SCMConfig) -> Tuple[Dict[str, BCSplit], SCMParams]:
    params = _make_params(cfg)
    rng = np.random.default_rng(cfg.seed + 2027)
    sizes = {
        "train": cfg.n_train,
        "val": cfg.n_val,
        "test_id": cfg.n_test,
        "test_ood_a": cfg.n_test,
        "test_ood_b": cfg.n_test,
        "test_ood_c": cfg.n_test,
        "test_ood_d": cfg.n_test,
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
        for m, u in enumerate(split.u):
            arrays[f"{split_name}_u_{m}"] = u.numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return splits, params


def _load_split(name: str, arrays: Mapping[str, np.ndarray], cfg: SCMConfig) -> BCSplit:
    return BCSplit(
        x=[torch.tensor(arrays[f"{name}_x_{m}"].astype(np.float32)) for m in range(cfg.n_modalities)],
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
    return [float(np.corrcoef(split.obs_mask.numpy()[:, m], y)[0, 1]) for m in range(split.obs_mask.shape[1])]


def split_sizes(splits: Mapping[str, BCSplit]) -> Dict[str, int]:
    return {name: int(split.y.shape[0]) for name, split in splits.items()}
