from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch


SCENARIOS = (
    "complementary",
    "redundant",
    "nuisance_only",
    "shortcut",
    "noisy_modality",
    "mediated_context",
)
SCENARIO_ALIASES = {
    "nuisance": "nuisance_only",
    "nuisance-only": "nuisance_only",
    "noisy": "noisy_modality",
    "noisy-modality": "noisy_modality",
    "mediated-context": "mediated_context",
    "context": "mediated_context",
    "context_mediation": "mediated_context",
}


def canonical_scenario(name: str) -> str:
    return SCENARIO_ALIASES.get(name, name)


@dataclass
class ISODataConfig:
    """Configuration for the ISO synthetic structural equation.

    The generated data follows

        Y ~ p(Y | S*)
        X_i = g_i(A_i S* + B_i U_i^tau + C_i Q^tau + eps_i^tau).

    Train/validation/id-test are source-domain draws. ``test`` is a target OOD
    draw where the latent state graph and label mechanism stay fixed while the
    observation-layer nuisance/residual variables may shift. ``Q`` is a shared
    shortcut/domain variable. In shortcut-like scenarios it is label-correlated
    in source splits and flipped or weakened at OOD test.
    """

    scenario: str = "complementary"
    seed: int = 0
    n_train: int = 1024
    n_val: int = 512
    n_test: int = 1024
    s_dim: int = 6
    u_dim: int = 3
    n_modalities: int = 3
    x_dim: int = 16
    noise_std: float = 0.35
    noisy_modality: int = 2
    noisy_noise_std: float = 1.80
    shortcut_modality: int = 2
    shortcut_strength: float = 0.0
    train_shortcut_corr: float = 0.85
    test_shortcut_corr: float = -0.65
    ood_residual_shift: float = 0.65
    train_nuisance_corr: float = 0.35
    test_nuisance_corr: float = -0.25
    ood_noise_multiplier: float = 1.35
    state_graph_strength: float = 0.25
    label_strength: float = 1.6
    label_nonlinear: bool = True
    standardize: bool = True

    def __post_init__(self) -> None:
        self.scenario = canonical_scenario(self.scenario)
        if self.scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}, got {self.scenario!r}")
        for name in ("n_train", "n_val", "n_test", "s_dim", "u_dim", "n_modalities", "x_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("noisy_modality", "shortcut_modality"):
            value = getattr(self, name)
            if not 0 <= value < self.n_modalities:
                raise ValueError(f"{name} must be in [0, {self.n_modalities}), got {value}")
        if self.noise_std <= 0 or self.noisy_noise_std <= 0:
            raise ValueError("noise_std and noisy_noise_std must be positive")
        for name in ("train_shortcut_corr", "test_shortcut_corr"):
            value = getattr(self, name)
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [-1, 1], got {value}")
        for name in ("train_nuisance_corr", "test_nuisance_corr"):
            value = getattr(self, name)
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [-1, 1], got {value}")
        if self.ood_residual_shift < 0.0:
            raise ValueError("ood_residual_shift must be non-negative")
        if self.ood_noise_multiplier <= 0.0:
            raise ValueError("ood_noise_multiplier must be positive")
        if self.scenario == "mediated_context" and self.shortcut_modality == 2 and self.n_modalities >= 1:
            self.shortcut_modality = 0
        if self.shortcut_strength == 0.0 and self.scenario in {"shortcut", "mediated_context"}:
            self.shortcut_strength = 2.5

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ISOParams:
    footprint: np.ndarray
    state_graph: np.ndarray
    A: List[np.ndarray]
    B: List[np.ndarray]
    C: List[np.ndarray]
    beta_y: np.ndarray
    noise_stds: np.ndarray
    target_noise_stds: np.ndarray


@dataclass
class ISOSplit:
    x: List[torch.Tensor]
    y: torch.Tensor
    s: torch.Tensor
    u: List[torch.Tensor]
    q: torch.Tensor


class ISODataset(torch.utils.data.Dataset):
    def __init__(self, split: ISOSplit):
        self.split = split
        self.n = int(split.y.shape[0])

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        return {
            "x": [xm[idx] for xm in self.split.x],
            "y": self.split.y[idx],
            "s": self.split.s[idx],
            "u": [um[idx] for um in self.split.u],
            "q": self.split.q[idx],
        }


def collate_iso_batch(items: Sequence[Dict[str, torch.Tensor | List[torch.Tensor]]]) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
    m = len(items[0]["x"])  # type: ignore[index]
    return {
        "x": [torch.stack([it["x"][j] for it in items], dim=0).float() for j in range(m)],  # type: ignore[index]
        "y": torch.stack([it["y"] for it in items], dim=0).float(),  # type: ignore[index]
        "s": torch.stack([it["s"] for it in items], dim=0).float(),  # type: ignore[index]
        "u": [torch.stack([it["u"][j] for it in items], dim=0).float() for j in range(m)],  # type: ignore[index]
        "q": torch.stack([it["q"] for it in items], dim=0).float(),  # type: ignore[index]
    }


def make_footprint(cfg: ISODataConfig) -> np.ndarray:
    """Return modality-to-state observability footprint G_d.

    Rows are modalities, columns are intrinsic state coordinates. A one means
    the modality receives direct signal from that state coordinate.
    """

    m, r = cfg.n_modalities, cfg.s_dim
    fp = np.zeros((m, r), dtype=np.float32)
    if m == 3 and r == 6:
        if cfg.scenario == "redundant":
            return np.array(
                [
                    [1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1],
                    [1, 1, 1, 0, 0, 0],
                ],
                dtype=np.float32,
            )
        if cfg.scenario == "mediated_context":
            return np.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 1, 0],
                    [0, 0, 1, 1, 1, 1],
                ],
                dtype=np.float32,
            )
        if cfg.scenario in {"nuisance_only", "shortcut"}:
            return np.array(
                [
                    [1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=np.float32,
            )
        return np.array(
            [
                [1, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 1, 0, 0, 1, 1],
            ],
            dtype=np.float32,
        )

    rng = np.random.default_rng(cfg.seed + 19)
    for j in range(r):
        fp[rng.integers(0, max(1, m - (1 if cfg.scenario in {"nuisance_only", "shortcut"} else 0))), j] = 1.0
    for mm in range(m):
        cols = rng.choice(r, size=max(1, r // 2), replace=False)
        fp[mm, cols] = 1.0
    if cfg.scenario == "redundant" and m >= 2:
        fp[-1] = fp[0]
    if cfg.scenario in {"nuisance_only", "shortcut"}:
        fp[-1] = 0.0
    return fp


def make_state_graph(cfg: ISODataConfig) -> np.ndarray:
    graph = np.zeros((cfg.s_dim, cfg.s_dim), dtype=np.float32)
    if cfg.scenario == "mediated_context" and cfg.s_dim >= 6:
        edges = [(1, 0, 0.9), (2, 1, 0.8), (3, 2, 0.75), (4, 2, 0.55), (5, 3, 0.35)]
        for child, parent, weight in edges:
            graph[child, parent] = cfg.state_graph_strength * weight
        return graph
    if cfg.s_dim >= 6:
        edges = [(1, 0, 0.7), (2, 0, -0.5), (3, 1, 0.6), (4, 2, 0.8), (5, 3, -0.4)]
        for child, parent, weight in edges:
            graph[child, parent] = cfg.state_graph_strength * weight
    else:
        for child in range(1, cfg.s_dim):
            graph[child, child - 1] = cfg.state_graph_strength / max(1.0, np.sqrt(cfg.s_dim))
    return graph


def _normalize_columns(mat: np.ndarray, active: np.ndarray, scale: float) -> np.ndarray:
    out = mat.copy()
    norms = np.linalg.norm(out, axis=0, keepdims=True) + 1e-6
    out = out / norms * active[None, :] * scale
    return out.astype(np.float32)


def make_params(cfg: ISODataConfig) -> ISOParams:
    rng = np.random.default_rng(cfg.seed)
    footprint = make_footprint(cfg)
    state_graph = make_state_graph(cfg)
    noise_stds = np.full(cfg.n_modalities, cfg.noise_std, dtype=np.float32)
    if cfg.scenario == "noisy_modality":
        noise_stds[cfg.noisy_modality] = cfg.noisy_noise_std
    target_noise_stds = noise_stds.copy()
    if cfg.scenario == "noisy_modality":
        target_noise_stds[cfg.noisy_modality] *= cfg.ood_noise_multiplier

    A: List[np.ndarray] = []
    B: List[np.ndarray] = []
    C: List[np.ndarray] = []
    for mm in range(cfg.n_modalities):
        active = footprint[mm]
        raw_A = rng.normal(0.0, 1.0, size=(cfg.x_dim, cfg.s_dim)).astype(np.float32)
        Am = _normalize_columns(raw_A * active[None, :], active, scale=1.25)
        nuisance_scale = 1.25 if (cfg.scenario == "nuisance_only" and active.sum() == 0) else 0.65
        Bm = rng.normal(0.0, nuisance_scale / np.sqrt(cfg.u_dim), size=(cfg.x_dim, cfg.u_dim)).astype(np.float32)
        c = rng.normal(0.0, 1.0, size=(cfg.x_dim,)).astype(np.float32)
        c = c / (np.linalg.norm(c) + 1e-6)
        if not (cfg.scenario in {"shortcut", "mediated_context"} and mm == cfg.shortcut_modality):
            c = np.zeros_like(c)
        A.append(Am)
        B.append(Bm)
        C.append(c.astype(np.float32))

    if cfg.scenario == "mediated_context" and cfg.s_dim >= 6:
        beta = np.array([0.0, 0.20, 0.70, 1.25, 0.75, 0.25], dtype=np.float32)
    else:
        beta = rng.normal(0.0, 1.0, size=(cfg.s_dim,)).astype(np.float32)
    beta = beta / (np.linalg.norm(beta) + 1e-6) * cfg.label_strength
    return ISOParams(
        footprint=footprint,
        state_graph=state_graph,
        A=A,
        B=B,
        C=C,
        beta_y=beta,
        noise_stds=noise_stds,
        target_noise_stds=target_noise_stds,
    )


def _sample_state(n: int, cfg: ISODataConfig, params: ISOParams, rng: np.random.Generator) -> np.ndarray:
    eps = rng.normal(0.0, 1.0, size=(n, cfg.s_dim)).astype(np.float32)
    transform = np.linalg.inv(np.eye(cfg.s_dim, dtype=np.float32) - params.state_graph)
    s = eps @ transform.T
    s = (s - s.mean(axis=0, keepdims=True)) / (s.std(axis=0, keepdims=True) + 1e-6)
    return s.astype(np.float32)


def _label_logits(s: np.ndarray, beta: np.ndarray, nonlinear: bool) -> np.ndarray:
    logits = s @ beta
    if nonlinear and s.shape[1] >= 4:
        logits = logits + 0.45 * np.sin(s[:, 0] * s[:, 1]) - 0.35 * s[:, 2] * s[:, 3]
    return logits.astype(np.float32)


def _sample_shortcut(y: np.ndarray, cfg: ISODataConfig, split: str, rng: np.random.Generator) -> np.ndarray:
    if cfg.scenario not in {"shortcut", "mediated_context"}:
        return rng.normal(0.0, 1.0, size=(y.shape[0],)).astype(np.float32)
    rho = cfg.train_shortcut_corr if _is_source_split(split) else cfg.test_shortcut_corr
    eps = rng.normal(0.0, 1.0, size=(y.shape[0],)).astype(np.float32)
    y_sign = 2.0 * y.astype(np.float32) - 1.0
    q = rho * y_sign + np.sqrt(max(0.0, 1.0 - rho * rho)) * eps
    return ((q - q.mean()) / (q.std() + 1e-6)).astype(np.float32)


def _is_source_split(split: str) -> bool:
    return split in {"train", "val", "id_test"}


def _sample_nuisance(
    n: int,
    cfg: ISODataConfig,
    split: str,
    modality: int,
    y: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    u = rng.normal(0.0, 1.0, size=(n, cfg.u_dim)).astype(np.float32)
    if cfg.u_dim == 0:
        return u

    rho = cfg.train_nuisance_corr if _is_source_split(split) else cfg.test_nuisance_corr
    if abs(rho) > 0.0:
        eps = rng.normal(0.0, 1.0, size=(n,)).astype(np.float32)
        y_sign = 2.0 * y.astype(np.float32) - 1.0
        u[:, 0] = rho * y_sign + np.sqrt(max(0.0, 1.0 - rho * rho)) * eps

    if not _is_source_split(split) and cfg.ood_residual_shift > 0.0:
        u[:, modality % cfg.u_dim] += cfg.ood_residual_shift
    return u.astype(np.float32)


def _noise_std_for_split(params: ISOParams, split: str, modality: int) -> float:
    if _is_source_split(split):
        return float(params.noise_stds[modality])
    return float(params.target_noise_stds[modality])


def _generate_split(n: int, cfg: ISODataConfig, params: ISOParams, split: str, rng: np.random.Generator) -> ISOSplit:
    s = _sample_state(n, cfg, params, rng)
    logits = _label_logits(s, params.beta_y, cfg.label_nonlinear)
    prob = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, prob).astype(np.float32)
    q = _sample_shortcut(y, cfg, split, rng)

    xs: List[torch.Tensor] = []
    us: List[torch.Tensor] = []
    for mm in range(cfg.n_modalities):
        u = _sample_nuisance(n, cfg, split, mm, y, rng)
        noise_std = _noise_std_for_split(params, split, mm)
        eps = rng.normal(0.0, noise_std, size=(n, cfg.x_dim)).astype(np.float32)
        pre = s @ params.A[mm].T + u @ params.B[mm].T
        pre = pre + cfg.shortcut_strength * q[:, None] * params.C[mm][None, :]
        x = np.tanh(pre + eps).astype(np.float32)
        xs.append(torch.tensor(x))
        us.append(torch.tensor(u))
    return ISOSplit(
        x=xs,
        y=torch.tensor(y[:, None]),
        s=torch.tensor(s),
        u=us,
        q=torch.tensor(q[:, None]),
    )


def standardize_splits(train: ISOSplit, *others: ISOSplit) -> None:
    for mm in range(len(train.x)):
        mean = train.x[mm].mean(dim=0, keepdim=True)
        std = train.x[mm].std(dim=0, keepdim=True).clamp_min(1e-5)
        train.x[mm] = (train.x[mm] - mean) / std
        for split in others:
            split.x[mm] = (split.x[mm] - mean) / std


def make_iso_data(cfg: ISODataConfig, include_id_test: bool = True) -> Tuple[ISOSplit, ISOSplit, ISOSplit | None, ISOSplit, ISOParams]:
    params = make_params(cfg)
    rng = np.random.default_rng(cfg.seed + 1234)
    train = _generate_split(cfg.n_train, cfg, params, "train", rng)
    val = _generate_split(cfg.n_val, cfg, params, "val", rng)
    id_test = _generate_split(cfg.n_test, cfg, params, "id_test", rng) if include_id_test else None
    test = _generate_split(cfg.n_test, cfg, params, "test", rng)
    if cfg.standardize:
        if id_test is None:
            standardize_splits(train, val, test)
        else:
            standardize_splits(train, val, id_test, test)
    return train, val, id_test, test, params
