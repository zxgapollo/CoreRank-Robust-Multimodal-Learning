from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


LABEL_MAP = {"CN": 0, "MCI": 1, "AD": 2}
LABEL_NAMES = ["CN", "MCI", "AD"]

FEATURE_GROUPS: Dict[str, Sequence[str]] = {
    "demographics": (
        "mri_age",
        "mri_field_strength_t",
        "demographics_ptgender",
        "demographics_pteducat",
        "demographics_ptmarry",
        "demographics_ptethcat",
        "demographics_ptraccat",
    ),
    "cognition": (
        "mmse_mmscore",
        "cdr_cdglobal",
        "cdr_cdrsb",
        "faq_faqtotal",
        "moca_moca",
        "adas_totscore",
        "adas_total13",
        "neurobat_limmtotal",
        "neurobat_ldeltotal",
        "neurobat_avtotb",
        "neurobat_bnttotal",
        "neurobat_traascor",
        "neurobat_trabscor",
        "neurobat_dspanfor",
        "neurobat_dspanbac",
        "neurobat_catanimsc",
    ),
    "behavior": (
        "npi_npitotal",
        "npiq_npiscore",
        "gds_gdtotal",
    ),
    "genetics_history": (
        "apoe_e4_count",
        "apoe_e2_count",
        "medical_history_mhpsych",
        "medical_history_mh2neurl",
        "medical_history_mh3head",
        "medical_history_mh4card",
        "medical_history_mh5resp",
        "medical_history_mh9endo",
        "medical_history_mh10gast",
        "medical_history_mh12rena",
        "medical_history_mh14alch",
        "medical_history_mh15drug",
        "medical_history_mh16smok",
        "medical_history_mh17mali",
    ),
}


def _apoe_allele_count(value: object, allele: str) -> float:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).replace(" ", "").replace("|", "/")
    alleles = [part for part in text.split("/") if part]
    if not alleles:
        return np.nan
    return float(sum(part == allele for part in alleles))


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "demographics_ptgender" in out:
        out["demographics_ptgender"] = (
            out["demographics_ptgender"].astype(str).str.upper().map(
                {"2": 0.0, "F": 0.0, "FEMALE": 0.0, "1": 1.0, "M": 1.0, "MALE": 1.0}
            )
        )
    genotype = out.get("apoe_genotype", pd.Series(index=out.index, dtype=object))
    out["apoe_e4_count"] = genotype.map(lambda value: _apoe_allele_count(value, "4"))
    out["apoe_e2_count"] = genotype.map(lambda value: _apoe_allele_count(value, "2"))
    for columns in FEATURE_GROUPS.values():
        for column in columns:
            if column not in out:
                out[column] = np.nan
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def prepare_split_manifest(
    master_csv: str | Path,
    cache_root: str | Path,
    split_csv: str | Path,
    summary_json: str | Path,
    seed: int = 2026,
) -> pd.DataFrame:
    frame = pd.read_csv(master_csv, dtype=str, keep_default_na=True)
    frame = frame[frame["diagnosis_3class"].isin(LABEL_MAP)].copy()
    if "has_core_multimodal" in frame:
        frame = frame[pd.to_numeric(frame["has_core_multimodal"], errors="coerce").fillna(0) == 1].copy()
    frame["label"] = frame["diagnosis_3class"].map(LABEL_MAP).astype(int)
    cache_root = Path(cache_root)
    frame["image_cache"] = frame.apply(
        lambda row: str(cache_root / row["subject_id"] / f"{row['subject_id']}_I{row['image_id']}_T1w_brain_96.npy"),
        axis=1,
    )
    frame = frame.sort_values("subject_id").reset_index(drop=True)

    train_idx, remainder_idx = train_test_split(
        np.arange(len(frame)), test_size=0.30, random_state=seed, stratify=frame["label"]
    )
    val_idx, test_idx = train_test_split(
        remainder_idx,
        test_size=0.50,
        random_state=seed + 1,
        stratify=frame.iloc[remainder_idx]["label"],
    )
    split = np.full(len(frame), "", dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"
    frame["split"] = split

    columns = ["subject_id", "image_id", "diagnosis_3class", "label", "split", "image_cache"]
    split_path = Path(split_csv)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    frame[columns].to_csv(split_path, index=False)

    counts = {
        split_name: frame[frame["split"] == split_name]["diagnosis_3class"].value_counts().sort_index().to_dict()
        for split_name in ("train", "val", "test")
    }
    summary = {
        "seed": seed,
        "subjects": int(len(frame)),
        "split_sizes": frame["split"].value_counts().to_dict(),
        "class_counts": counts,
        "subject_unique": bool(frame["subject_id"].is_unique),
        "master_csv": str(master_csv),
        "cache_root": str(cache_root),
        "split_csv": str(split_path),
    }
    summary_path = Path(summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return frame[columns]


def load_and_transform_features(
    master_csv: str | Path,
    split_csv: str | Path,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray, Dict[str, object]]:
    master = _numeric_frame(pd.read_csv(master_csv, dtype=str, keep_default_na=True))
    split = pd.read_csv(split_csv, dtype={"subject_id": str, "image_id": str})
    split["label"] = pd.to_numeric(split["label"], errors="raise").astype(int)
    frame = split.merge(master, on=["subject_id", "image_id"], how="left", validate="one_to_one", suffixes=("", "_master"))
    train_mask = frame["split"].eq("train").to_numpy()
    groups: Dict[str, np.ndarray] = {}
    availability: List[np.ndarray] = [np.ones(len(frame), dtype=np.float32)]
    statistics: Dict[str, object] = {}

    for group_name, columns in FEATURE_GROUPS.items():
        values = frame[list(columns)].to_numpy(dtype=np.float64)
        observed = np.isfinite(values)
        group_available = observed.any(axis=1).astype(np.float32)
        availability.append(group_available)
        train_values = values[train_mask]
        medians = np.nanmedian(train_values, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        filled = np.where(observed, values, medians[None, :])
        means = filled[train_mask].mean(axis=0)
        stds = filled[train_mask].std(axis=0)
        stds = np.where(stds > 1e-6, stds, 1.0)
        standardized = (filled - means[None, :]) / stds[None, :]
        groups[group_name] = np.concatenate([standardized, observed.astype(np.float64)], axis=1).astype(np.float32)
        statistics[group_name] = {
            "columns": list(columns),
            "median": medians.tolist(),
            "mean": means.tolist(),
            "std": stds.tolist(),
        }

    return frame, groups, np.stack(availability, axis=1), statistics


class ADNIDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        groups: Dict[str, np.ndarray],
        availability: np.ndarray,
        split: str,
        train: bool,
        seed: int,
    ):
        self.indices = np.flatnonzero(frame["split"].eq(split).to_numpy())
        self.frame = frame.reset_index(drop=True)
        self.groups = groups
        self.availability = availability
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> Dict[str, object]:
        index = int(self.indices[position])
        row = self.frame.iloc[index]
        image_path = Path(str(row["image_cache"]))
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image = np.load(image_path, allow_pickle=False).astype(np.float32, copy=False)
        image = torch.from_numpy(np.ascontiguousarray(image[None, ...]))
        if self.train:
            if random.random() < 0.5:
                image = torch.flip(image, dims=(1,))
            if random.random() < 0.5:
                image = torch.flip(image, dims=(3,))
            image = image * (0.95 + 0.10 * random.random())
            if random.random() < 0.25:
                image = image + 0.01 * torch.randn_like(image)
        return {
            "image": image,
            "groups": {name: torch.from_numpy(values[index]) for name, values in self.groups.items()},
            "availability": torch.from_numpy(self.availability[index]),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "subject_id": str(row["subject_id"]),
        }
