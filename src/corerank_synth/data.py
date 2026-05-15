from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import torch


@dataclass
class SyntheticConfig:
    scenario: str = "complementary"  # complementary | redundant | biased | domain
    seed: int = 0
    n_train: int = 5000
    n_val: int = 1000
    n_test: int = 2000
    z_dim: int = 6
    u_dim: int = 3
    n_modalities: int = 3
    x_dim: int = 16
    noise_std: float = 0.35
    bias_strength: float = 0.0
    biased_modality: int = 0
    train_bias_corr: float = 0.85
    test_bias_corr: float = -0.50
    domain_shift_strength: float = 0.0
    domain_shifted_modality: int = 0
    core_graph_strength: float = 0.35
    label_nonlinear: bool = True
    standardize: bool = True

    def __post_init__(self) -> None:
        valid_scenarios = {"complementary", "redundant", "biased", "domain"}
        if self.scenario not in valid_scenarios:
            raise ValueError(f"scenario must be one of {sorted(valid_scenarios)}, got {self.scenario!r}")
        for name in ("n_train", "n_val", "n_test", "z_dim", "u_dim", "n_modalities", "x_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if not 0 <= self.biased_modality < self.n_modalities:
            raise ValueError(f"biased_modality must be in [0, {self.n_modalities}), got {self.biased_modality}")
        if not 0 <= self.domain_shifted_modality < self.n_modalities:
            raise ValueError(f"domain_shifted_modality must be in [0, {self.n_modalities}), got {self.domain_shifted_modality}")
        if self.noise_std <= 0:
            raise ValueError(f"noise_std must be positive, got {self.noise_std}")
        for name in ("train_bias_corr", "test_bias_corr"):
            value = getattr(self, name)
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [-1, 1], got {value}")

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SyntheticParams:
    footprint: np.ndarray  # [M, r]
    core_graph: np.ndarray  # [r, r], core_graph[child, parent] is a directed effect.
    A: List[np.ndarray]    # each [d, r]
    B: List[np.ndarray]    # each [d, q]
    bias_vec: List[np.ndarray]  # each [d]
    domain_vec: List[np.ndarray]  # each [d]
    beta_y: np.ndarray     # [r]
    noise_std: float


@dataclass
class SyntheticSplit:
    x: List[torch.Tensor]
    y: torch.Tensor
    z: torch.Tensor
    u: List[torch.Tensor]
    bias: List[torch.Tensor]
    domain: torch.Tensor


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, split: SyntheticSplit):
        self.split = split
        self.n = split.y.shape[0]
        self.m = len(split.x)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": [xm[idx] for xm in self.split.x],
            "y": self.split.y[idx],
            "z": self.split.z[idx],
            "u": [um[idx] for um in self.split.u],
            "bias": [bm[idx] for bm in self.split.bias],
            "domain": self.split.domain[idx],
        }


def collate_batch(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
    m = len(items[0]["x"])
    out = {
        "x": [torch.stack([it["x"][j] for it in items], dim=0) for j in range(m)],
        "y": torch.stack([it["y"] for it in items], dim=0).float(),
        "z": torch.stack([it["z"] for it in items], dim=0).float(),
        "u": [torch.stack([it["u"][j] for it in items], dim=0).float() for j in range(m)],
        "bias": [torch.stack([it["bias"][j] for it in items], dim=0).float() for j in range(m)],
        "domain": torch.stack([it["domain"] for it in items], dim=0).float(),
    }
    return out


def make_footprint(cfg: SyntheticConfig) -> np.ndarray:
    r, m = cfg.z_dim, cfg.n_modalities
    if r == 6 and m == 3:
        if cfg.scenario == "redundant":
            fp = np.array([
                [1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [1, 1, 1, 0, 0, 0],
            ], dtype=np.float32)
        else:
            fp = np.array([
                [1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 1, 0, 0, 1, 1],
            ], dtype=np.float32)
        return fp

    # Generic fallback: each modality observes about half of the dimensions and each dimension is covered.
    rng = np.random.default_rng(cfg.seed + 17)
    fp = np.zeros((m, r), dtype=np.float32)
    for j in range(r):
        fp[rng.integers(0, m), j] = 1.0
    for mm in range(m):
        cols = rng.choice(r, size=max(1, r // 2), replace=False)
        fp[mm, cols] = 1.0
    if cfg.scenario == "redundant" and m >= 2:
        fp[-1] = fp[0]
    return fp


def make_core_graph(cfg: SyntheticConfig) -> np.ndarray:
    """Return a simple acyclic latent causal graph among disease-core coordinates.

    The benchmark exposes this graph as ground truth for synthetic analysis, but the
    first CoreRank model does not try to identify it.
    """
    graph = np.zeros((cfg.z_dim, cfg.z_dim), dtype=np.float32)
    if cfg.z_dim >= 6:
        edges = [
            (2, 0, 1.00),
            (2, 1, -0.70),
            (3, 2, 0.85),
            (4, 1, 0.75),
            (5, 4, 0.90),
        ]
        for child, parent, weight in edges:
            graph[child, parent] = cfg.core_graph_strength * weight
    else:
        for child in range(1, cfg.z_dim):
            graph[child, child - 1] = cfg.core_graph_strength / max(1.0, np.sqrt(cfg.z_dim))
    return graph


def _make_params(cfg: SyntheticConfig) -> SyntheticParams:
    rng = np.random.default_rng(cfg.seed)
    fp = make_footprint(cfg)
    core_graph = make_core_graph(cfg)
    A, B, bias_vec, domain_vec = [], [], [], []
    for mm in range(cfg.n_modalities):
        Am = rng.normal(0, 1.0 / np.sqrt(max(1, fp[mm].sum())), size=(cfg.x_dim, cfg.z_dim)).astype(np.float32)
        Am *= fp[mm][None, :]
        # Normalize columns to avoid pathological weak dimensions.
        col_norm = np.linalg.norm(Am, axis=0, keepdims=True) + 1e-6
        Am = Am / col_norm * fp[mm][None, :]
        Bm = rng.normal(0, 0.7 / np.sqrt(cfg.u_dim), size=(cfg.x_dim, cfg.u_dim)).astype(np.float32)
        bv = rng.normal(0, 1.0, size=(cfg.x_dim,)).astype(np.float32)
        bv = bv / (np.linalg.norm(bv) + 1e-6)
        dv = rng.normal(0, 1.0, size=(cfg.x_dim,)).astype(np.float32)
        dv = dv / (np.linalg.norm(dv) + 1e-6)
        A.append(Am.astype(np.float32))
        B.append(Bm.astype(np.float32))
        bias_vec.append(bv.astype(np.float32))
        domain_vec.append(dv.astype(np.float32))
    beta_y = rng.normal(0, 1.0, size=(cfg.z_dim,)).astype(np.float32)
    beta_y = beta_y / (np.linalg.norm(beta_y) + 1e-6) * 1.5
    return SyntheticParams(fp, core_graph, A, B, bias_vec, domain_vec, beta_y, cfg.noise_std)


def _sample_core(n: int, cfg: SyntheticConfig, params: SyntheticParams, rng: np.random.Generator) -> np.ndarray:
    eps = rng.normal(0, 1, size=(n, cfg.z_dim)).astype(np.float32)
    # A[child, parent] means z_child <- A * z_parent + noise.
    transform = np.linalg.inv(np.eye(cfg.z_dim, dtype=np.float32) - params.core_graph)
    z = eps @ transform.T
    z = (z - z.mean(axis=0, keepdims=True)) / (z.std(axis=0, keepdims=True) + 1e-6)
    return z.astype(np.float32)


def _label_logits(z: np.ndarray, beta: np.ndarray, nonlinear: bool) -> np.ndarray:
    logits = z @ beta
    if nonlinear and z.shape[1] >= 4:
        logits = logits + 0.6 * np.sin(z[:, 0] * z[:, 1]) - 0.4 * z[:, 2] * z[:, 3]
    return logits.astype(np.float32)


def _generate_split(n: int, cfg: SyntheticConfig, params: SyntheticParams, split: str, rng: np.random.Generator) -> SyntheticSplit:
    z = _sample_core(n, cfg, params, rng)
    logits = _label_logits(z, params.beta_y, cfg.label_nonlinear)
    p = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, p).astype(np.float32)
    y_sign = 2.0 * y - 1.0

    rho = 0.0
    if cfg.scenario == "biased":
        rho = cfg.train_bias_corr if split in {"train", "val"} else cfg.test_bias_corr
    elif cfg.bias_strength > 0:
        rho = cfg.train_bias_corr if split in {"train", "val"} else cfg.test_bias_corr

    domain_mean = 1.0 if cfg.scenario == "domain" and split == "test" else 0.0
    domain = (domain_mean + 0.1 * rng.normal(0, 1, size=(n,))).astype(np.float32)

    xs, us, biases = [], [], []
    for mm in range(cfg.n_modalities):
        u = rng.normal(0, 1, size=(n, cfg.u_dim)).astype(np.float32)
        eps_b = rng.normal(0, 1, size=(n,)).astype(np.float32)
        b = rho * y_sign + np.sqrt(max(0.0, 1.0 - rho ** 2)) * eps_b
        pre = z @ params.A[mm].T + u @ params.B[mm].T
        core_private = np.tanh(pre)
        applied_bias_strength = cfg.bias_strength if mm == cfg.biased_modality else 0.0
        biased = applied_bias_strength * b[:, None] * params.bias_vec[mm][None, :]
        applied_domain_strength = cfg.domain_shift_strength if mm == cfg.domain_shifted_modality else 0.0
        domain_shift = applied_domain_strength * domain[:, None] * params.domain_vec[mm][None, :]
        noise = rng.normal(0, cfg.noise_std, size=(n, cfg.x_dim)).astype(np.float32)
        x = (core_private + biased + domain_shift + noise).astype(np.float32)
        xs.append(torch.tensor(x))
        us.append(torch.tensor(u))
        biases.append(torch.tensor(b[:, None].astype(np.float32)))

    return SyntheticSplit(
        x=xs,
        y=torch.tensor(y[:, None]),
        z=torch.tensor(z),
        u=us,
        bias=biases,
        domain=torch.tensor(domain[:, None].astype(np.float32)),
    )


def standardize_splits(train: SyntheticSplit, val: SyntheticSplit, test: SyntheticSplit) -> None:
    for m in range(len(train.x)):
        mean = train.x[m].mean(dim=0, keepdim=True)
        std = train.x[m].std(dim=0, keepdim=True).clamp_min(1e-5)
        train.x[m] = (train.x[m] - mean) / std
        val.x[m] = (val.x[m] - mean) / std
        test.x[m] = (test.x[m] - mean) / std


def make_synthetic_data(cfg: SyntheticConfig) -> Tuple[SyntheticSplit, SyntheticSplit, SyntheticSplit, SyntheticParams]:
    params = _make_params(cfg)
    rng = np.random.default_rng(cfg.seed + 123)
    train = _generate_split(cfg.n_train, cfg, params, "train", rng)
    val = _generate_split(cfg.n_val, cfg, params, "val", rng)
    test = _generate_split(cfg.n_test, cfg, params, "test", rng)
    if cfg.standardize:
        standardize_splits(train, val, test)
    return train, val, test, params


def true_fisher_for_split(split: SyntheticSplit, params: SyntheticParams, cfg: SyntheticConfig, modalities: List[int]) -> torch.Tensor:
    """Compute the average true nuisance-adjusted Fisher using the known nonlinear generator.

    This is used only for diagnostics. It assumes the synthetic tanh Gaussian generator.
    """
    z = split.z.numpy()
    n, r = z.shape
    q = cfg.u_dim
    K = np.zeros((n, r, r), dtype=np.float32)
    sigma2 = cfg.noise_std ** 2
    for mm in modalities:
        A = params.A[mm]
        B = params.B[mm]
        u = split.u[mm].numpy()
        pre = z @ A.T + u @ B.T
        dphi = 1.0 - np.tanh(pre) ** 2  # [n, d]
        for i in range(n):
            Dz = dphi[i][:, None] * A
            Du = dphi[i][:, None] * B
            Izz = Dz.T @ Dz / sigma2
            Izu = Dz.T @ Du / sigma2
            Iuu = Du.T @ Du / sigma2
            Icore = Izz - Izu @ np.linalg.pinv(Iuu + 1e-3 * np.eye(q, dtype=np.float32)) @ Izu.T
            K[i] += Icore.astype(np.float32)
    return torch.tensor(K)
