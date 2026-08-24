from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import torch


MODALITIES = ("pet", "mri", "cog", "demo")
SPLITS = ("train", "val", "test_id", "test_ood_shortcut", "test_ood_noise")


@dataclass
class ADSCMConfig:
    seed: int = 0
    n_train: int = 5000
    n_val: int = 1000
    n_test: int = 2000
    x_dim: int = 8
    hidden_gen_dim: int = 20
    base_noise: float = 0.25
    disease_noise_scale: float = 1.0
    demo_noise_scale: float = 1.0
    demo_to_s_strength: float = 0.35
    shortcut_strength: float = 0.55
    shortcut_test_strength: float = 0.10
    noise_train_scale: float = 1.0
    noise_ood_scale: float = 1.0
    noise_modality: str = "mri"
    standardize_x: bool = True

    @property
    def n_modalities(self) -> int:
        return len(MODALITIES)

    @property
    def k(self) -> int:
        return 6

    def __post_init__(self) -> None:
        for name in ("n_train", "n_val", "n_test", "x_dim", "hidden_gen_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.noise_modality not in MODALITIES:
            raise ValueError(f"noise_modality must be one of {MODALITIES}")
        for name in (
            "base_noise",
            "disease_noise_scale",
            "demo_noise_scale",
            "demo_to_s_strength",
            "shortcut_strength",
            "shortcut_test_strength",
            "noise_train_scale",
            "noise_ood_scale",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ADParams:
    graph: np.ndarray
    state_mask: np.ndarray
    modality_mask: np.ndarray
    gen_w1: List[np.ndarray]
    gen_b1: List[np.ndarray]
    gen_w2: List[np.ndarray]
    gen_b2: List[np.ndarray]


@dataclass
class ADSplit:
    x: List[torch.Tensor]
    y: torch.Tensor
    latent: torch.Tensor
    state: torch.Tensor
    clinical_score: torch.Tensor
    label_noise: torch.Tensor
    shortcut_component: torch.Tensor


class ADDataset(torch.utils.data.Dataset):
    def __init__(self, split: ADSplit):
        self.split = split

    def __len__(self) -> int:
        return int(self.split.y.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        return {
            "x": [xm[idx] for xm in self.split.x],
            "y": self.split.y[idx],
            "latent": self.split.latent[idx],
            "state": self.split.state[idx],
            "clinical_score": self.split.clinical_score[idx],
            "label_noise": self.split.label_noise[idx],
            "shortcut_component": self.split.shortcut_component[idx],
        }


def collate_ad_batch(items: List[Dict[str, torch.Tensor | List[torch.Tensor]]]) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
    m = len(items[0]["x"])  # type: ignore[arg-type]
    return {
        "x": [torch.stack([it["x"][j] for it in items], dim=0).float() for j in range(m)],  # type: ignore[index]
        "y": torch.stack([it["y"] for it in items], dim=0).long(),  # type: ignore[arg-type]
        "latent": torch.stack([it["latent"] for it in items], dim=0).float(),  # type: ignore[arg-type]
        "state": torch.stack([it["state"] for it in items], dim=0).float(),  # type: ignore[arg-type]
        "clinical_score": torch.stack([it["clinical_score"] for it in items], dim=0).float(),  # type: ignore[arg-type]
        "label_noise": torch.stack([it["label_noise"] for it in items], dim=0).float(),  # type: ignore[arg-type]
        "shortcut_component": torch.stack([it["shortcut_component"] for it in items], dim=0).float(),  # type: ignore[arg-type]
    }


def true_graph() -> np.ndarray:
    # Latent order: D, S, Z1 amyloid, Z2 tau, Z3 neurodegeneration, Z4 reserve.
    graph = np.zeros((6, 6), dtype=np.float32)
    for child, parent in [
        (1, 0),  # D -> S
        (2, 1),  # S -> amyloid
        (3, 1),  # S -> tau
        (3, 2),  # amyloid -> tau
        (4, 1),  # S -> neurodegeneration
        (4, 3),  # tau -> neurodegeneration
        (4, 0),  # D -> neurodegeneration
    ]:
        graph[child, parent] = 1.0
    return graph


def true_modality_mask() -> np.ndarray:
    mask = np.zeros((4, 6), dtype=np.float32)
    mask[0, [2, 3]] = 1.0  # PET <- amyloid, tau
    mask[1, [4]] = 1.0  # MRI <- neurodegeneration
    mask[2, [4, 5]] = 1.0  # Cog <- neurodegeneration, reserve
    mask[3, [0]] = 1.0  # Demo <- demographic/risk D
    return mask


def true_state_mask() -> np.ndarray:
    # Label-relevant clean disease state: Z2 tau, Z3 neurodegeneration, Z4 reserve.
    mask = np.zeros(6, dtype=np.float32)
    mask[[3, 4, 5]] = 1.0
    return mask


def _make_params(cfg: ADSCMConfig) -> ADParams:
    rng = np.random.default_rng(cfg.seed + 511)
    masks = true_modality_mask()
    gen_w1: List[np.ndarray] = []
    gen_b1: List[np.ndarray] = []
    gen_w2: List[np.ndarray] = []
    gen_b2: List[np.ndarray] = []
    for m in range(cfg.n_modalities):
        w1 = rng.normal(0, 0.9 / np.sqrt(cfg.k), size=(cfg.k, cfg.hidden_gen_dim)).astype(np.float32)
        w1 *= masks[m, :, None]
        gen_w1.append(w1)
        gen_b1.append(rng.normal(0, 0.10, size=(cfg.hidden_gen_dim,)).astype(np.float32))
        gen_w2.append(rng.normal(0, 0.8 / np.sqrt(cfg.hidden_gen_dim), size=(cfg.hidden_gen_dim, cfg.x_dim)).astype(np.float32))
        gen_b2.append(rng.normal(0, 0.05, size=(cfg.x_dim,)).astype(np.float32))
    return ADParams(true_graph(), true_state_mask(), masks, gen_w1, gen_b1, gen_w2, gen_b2)


def _sample_latents(
    n: int,
    rng: np.random.Generator,
    demo_to_s_strength: float,
    demo_to_neuro_strength: float,
) -> Tuple[np.ndarray, np.ndarray]:
    eps = rng.normal(0, 1, size=(n, 6)).astype(np.float32)
    d = eps[:, 0]
    reserve = eps[:, 5]
    s = np.tanh(demo_to_s_strength * d + 0.8 * eps[:, 1])
    amyloid = np.tanh(1.15 * s + 0.35 * eps[:, 2])
    tau = np.tanh(0.75 * s + 0.95 * amyloid + 0.35 * eps[:, 3])
    shortcut_component = demo_to_neuro_strength * d
    neuro = np.tanh(0.55 * s + 0.85 * tau + shortcut_component + 0.35 * eps[:, 4])
    latent = np.stack([d, s, amyloid, tau, neuro, reserve], axis=1).astype(np.float32)
    return latent, shortcut_component[:, None].astype(np.float32)


def _state_from_latent(latent: np.ndarray) -> np.ndarray:
    return (latent * true_state_mask()[None, :]).astype(np.float32)


def _clinical_score(latent: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    tau = latent[:, 3]
    neuro = latent[:, 4]
    reserve = latent[:, 5]
    # Label SCM: U_Y enters the structural equation for observed diagnosis.
    label_noise = rng.normal(0, 1, size=tau.shape).astype(np.float32)
    score = 1.0 * tau + 1.25 * neuro - 0.85 * reserve + 0.20 * label_noise
    return score.astype(np.float32), label_noise[:, None].astype(np.float32)


def _labels(score: np.ndarray) -> np.ndarray:
    y = np.zeros(score.shape[0], dtype=np.int64)
    y[score >= -0.45] = 1
    y[score >= 0.75] = 2
    return y


def _fixed_mlp(latent: np.ndarray, params: ADParams, modality: int) -> np.ndarray:
    h = np.tanh(latent @ params.gen_w1[modality] + params.gen_b1[modality])
    return np.tanh(h @ params.gen_w2[modality] + params.gen_b2[modality]).astype(np.float32)


def _split_settings(split: str, cfg: ADSCMConfig) -> Tuple[float, Dict[str, float]]:
    demo_to_neuro = cfg.shortcut_strength
    noise = {
        "pet": cfg.base_noise * cfg.disease_noise_scale,
        "mri": cfg.base_noise * cfg.disease_noise_scale,
        "cog": cfg.base_noise * cfg.disease_noise_scale,
        "demo": cfg.base_noise * cfg.demo_noise_scale,
    }
    if split == "test_ood_shortcut":
        demo_to_neuro = cfg.shortcut_test_strength
    if split in {"train", "val", "test_id"}:
        noise[cfg.noise_modality] *= cfg.noise_train_scale
    if split == "test_ood_noise":
        noise[cfg.noise_modality] *= cfg.noise_ood_scale
    return demo_to_neuro, noise


def _generate_split(n: int, split: str, cfg: ADSCMConfig, params: ADParams, rng: np.random.Generator) -> ADSplit:
    demo_to_neuro, noise = _split_settings(split, cfg)
    latent, shortcut_component = _sample_latents(
        n,
        rng,
        demo_to_s_strength=cfg.demo_to_s_strength,
        demo_to_neuro_strength=demo_to_neuro,
    )
    state = _state_from_latent(latent)
    score, label_noise = _clinical_score(latent, rng)
    y = _labels(score)
    xs: List[torch.Tensor] = []
    for m, name in enumerate(MODALITIES):
        x = _fixed_mlp(latent, params, m)
        x = x + rng.normal(0, noise[name], size=x.shape).astype(np.float32)
        xs.append(torch.tensor(x.astype(np.float32)))
    return ADSplit(
        x=xs,
        y=torch.tensor(y),
        latent=torch.tensor(latent),
        state=torch.tensor(state),
        clinical_score=torch.tensor(score[:, None]),
        label_noise=torch.tensor(label_noise),
        shortcut_component=torch.tensor(shortcut_component),
    )


def _standardize(splits: Mapping[str, ADSplit]) -> None:
    train = splits["train"]
    for m in range(len(train.x)):
        mean = train.x[m].mean(dim=0, keepdim=True)
        std = train.x[m].std(dim=0, keepdim=True).clamp_min(1e-5)
        for split in splits.values():
            split.x[m] = (split.x[m] - mean) / std


def make_ad_dataset(cfg: ADSCMConfig) -> Tuple[Dict[str, ADSplit], ADParams]:
    params = _make_params(cfg)
    rng = np.random.default_rng(cfg.seed + 2029)
    sizes = {
        "train": cfg.n_train,
        "val": cfg.n_val,
        "test_id": cfg.n_test,
        "test_ood_shortcut": cfg.n_test,
        "test_ood_noise": cfg.n_test,
    }
    splits = {name: _generate_split(n, name, cfg, params, rng) for name, n in sizes.items()}
    if cfg.standardize_x:
        _standardize(splits)
    return splits, params


def save_ad_dataset(path: str | Path, cfg: ADSCMConfig) -> Tuple[Dict[str, ADSplit], ADParams]:
    splits, params = make_ad_dataset(cfg)
    arrays: Dict[str, np.ndarray] = {
        "config_json": np.array(json.dumps(cfg.to_dict())),
        "param_graph": params.graph,
        "param_state_mask": params.state_mask,
        "param_modality_mask": params.modality_mask,
    }
    for split_name, split in splits.items():
        arrays[f"{split_name}_y"] = split.y.numpy()
        arrays[f"{split_name}_latent"] = split.latent.numpy()
        arrays[f"{split_name}_state"] = split.state.numpy()
        arrays[f"{split_name}_clinical_score"] = split.clinical_score.numpy()
        arrays[f"{split_name}_label_noise"] = split.label_noise.numpy()
        arrays[f"{split_name}_shortcut_component"] = split.shortcut_component.numpy()
        for m, x in enumerate(split.x):
            arrays[f"{split_name}_x_{m}"] = x.numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return splits, params


def class_counts(split: ADSplit) -> List[int]:
    y = split.y.numpy()
    return [int((y == c).sum()) for c in range(3)]


def demo_shortcut_label_corr(split: ADSplit) -> float:
    demo_component = split.latent.numpy()[:, 0]
    y = split.y.numpy().astype(np.float64)
    return float(np.corrcoef(demo_component, y)[0, 1])


def shortcut_component_label_corr(split: ADSplit) -> float:
    comp = split.shortcut_component.numpy().reshape(-1)
    y = split.y.numpy().astype(np.float64)
    if float(np.std(comp)) < 1e-8:
        return 0.0
    return float(np.corrcoef(comp, y)[0, 1])
