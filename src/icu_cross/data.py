from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import MODALITIES


def cache_modalities(cache_root: str | Path) -> tuple[str, ...]:
    manifest_path = Path(cache_root) / "manifest.json"
    if not manifest_path.exists():
        return MODALITIES
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    modalities = tuple(str(value) for value in manifest["modalities"])
    if not modalities:
        raise ValueError(f"No modalities in {manifest_path}")
    return modalities


def load_raw_split(
    cache_root: str | Path,
    dataset: str,
    split: str,
    modalities: Sequence[str] | None = None,
) -> Dict[str, object]:
    modalities = tuple(modalities or cache_modalities(cache_root))
    split_dir = Path(cache_root) / dataset / split
    values = {
        modality: np.load(split_dir / f"{modality}.npy", mmap_mode="r")
        for modality in modalities
    }
    return {
        "values": values,
        "mask": np.load(split_dir / "mask.npy", mmap_mode="r"),
        "labels": np.load(split_dir / "labels.npy", mmap_mode="r"),
        "ids": (split_dir / "ids.txt").read_text(encoding="utf-8").splitlines(),
    }


def fit_source_statistics(
    cache_root: str | Path,
    modalities: Sequence[str] | None = None,
) -> Dict[str, object]:
    modalities = tuple(modalities or cache_modalities(cache_root))
    raw = load_raw_split(cache_root, "mimic4", "train", modalities)
    mask = raw["mask"]
    statistics: Dict[str, object] = {"fitted_on": "mimic4/train", "modalities": {}}
    manifest_path = Path(cache_root) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    for index, modality in enumerate(modalities):
        values = np.asarray(raw["values"][modality], dtype=np.float64)
        observed = np.asarray(mask[:, index], dtype=bool)
        selected = values[observed]
        if len(selected) == 0:
            raise ValueError(f"No observed source samples for {modality}")
        if manifest.get("pre_normalized", False):
            # Hourly METRE caches already use source-train-only normalization.
            # Keep only the last feature axis so broadcasting also works for [T,D].
            mean = np.zeros(values.shape[-1], dtype=np.float64)
            std = np.ones(values.shape[-1], dtype=np.float64)
        else:
            flattened = selected.reshape(-1, selected.shape[-1])
            mean = flattened.mean(axis=0)
            std = flattened.std(axis=0)
        std = np.where(std > 1e-6, std, 1.0)
        statistics["modalities"][modality] = {"mean": mean.tolist(), "std": std.tolist()}
    return statistics


class ICUFeatureDataset(Dataset):
    def __init__(
        self,
        cache_root: str | Path,
        dataset: str,
        split: str,
        statistics: Dict[str, object],
        modalities: Sequence[str] | None = None,
    ):
        self.modalities = tuple(modalities or cache_modalities(cache_root))
        raw = load_raw_split(cache_root, dataset, split, self.modalities)
        self.ids: Sequence[str] = raw["ids"]
        self.mask = raw["mask"]
        self.labels = raw["labels"]
        self.values = raw["values"]
        self.means = {
            modality: np.asarray(statistics["modalities"][modality]["mean"], dtype=np.float32)
            for modality in self.modalities
        }
        self.stds = {
            modality: np.asarray(statistics["modalities"][modality]["std"], dtype=np.float32)
            for modality in self.modalities
        }
        if len(self.ids) != len(self.labels):
            raise ValueError(f"ID/label mismatch for {dataset}/{split}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, str]:
        mask = np.asarray(self.mask[index], dtype=np.float32)
        modalities = []
        for modality_index, modality in enumerate(self.modalities):
            raw = np.asarray(self.values[modality][index], dtype=np.float32)
            normalized = (raw - self.means[modality]) / self.stds[modality]
            normalized = normalized * mask[modality_index]
            modalities.append(torch.from_numpy(np.array(normalized, dtype=np.float32, copy=True)))
        return (
            modalities,
            torch.from_numpy(np.array(mask, dtype=np.float32, copy=True)),
            torch.tensor(float(self.labels[index]), dtype=torch.float32),
            self.ids[index],
        )


def save_statistics(statistics: Dict[str, object], path: str | Path) -> None:
    Path(path).write_text(json.dumps(statistics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
