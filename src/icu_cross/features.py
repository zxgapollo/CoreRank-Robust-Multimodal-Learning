from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


MODALITIES = ("demographics", "diagnosis", "procedure", "medication", "labs")
CODE_MODALITIES = ("diagnosis", "procedure", "medication")
CODE_DIM = 768
LAB_DIM = 14

TYPE_MAP = {
    "mimic4": {
        "diagnoses_icd": "diagnosis",
        "procedures_icd": "procedure",
        "prescriptions": "medication",
    },
    "eicu": {
        "diagnosis": "diagnosis",
        "treatment": "procedure",
        "medication": "medication",
    },
}


@dataclass(frozen=True)
class FeaturePaths:
    data_root: Path
    muse_src: Path
    output_root: Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_embedding_matrix(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        count_text, dim_text = handle.readline().strip().split()
        count, dim = int(count_text), int(dim_text)
        matrix = np.empty((count, dim), dtype=np.float32)
        for row in range(count):
            line = handle.readline()
            if not line:
                raise ValueError(f"Embedding file ended at row {row}: {path}")
            values = np.fromstring(line, sep=" ", dtype=np.float32)
            if values.size != dim:
                raise ValueError(f"Embedding row {row} has {values.size} values, expected {dim}")
            matrix[row] = values
    if dim != CODE_DIM:
        raise ValueError(f"Expected {CODE_DIM}-D shared ClinicalBERT embeddings, got {dim}")
    return matrix


def read_ids(path: str | Path) -> list[str]:
    return [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def broad_ethnicity(value: object) -> str:
    text = str(value or "").strip().upper()
    if "BLACK" in text or "AFRICAN" in text:
        return "black"
    if "ASIAN" in text:
        return "asian"
    if "HISPANIC" in text or "LATINO" in text or "SOUTH AMERICAN" in text:
        return "hispanic"
    if "NATIVE" in text or "AMERICAN INDIAN" in text:
        return "native"
    if "WHITE" in text or "CAUCASIAN" in text or "PORTUGUESE" in text:
        return "white"
    if text in {"", "UNKNOWN", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER", "OTHER/UNKNOWN"}:
        return "unknown"
    return "other"


def demographic_vector(admission: object) -> np.ndarray:
    gender = str(getattr(admission, "gender", "") or "").strip().upper()
    gender_names = ("female", "male", "unknown")
    gender_name = "female" if gender in {"F", "FEMALE"} else "male" if gender in {"M", "MALE"} else "unknown"
    ethnicity_names = ("white", "black", "asian", "hispanic", "native", "other", "unknown")
    age = float(getattr(admission, "age")) / 100.0
    return np.asarray(
        [age]
        + [float(gender_name == name) for name in gender_names]
        + [float(broad_ethnicity(getattr(admission, "ethnicity", "")) == name) for name in ethnicity_names],
        dtype=np.float32,
    )


def split_codes(admission: object, dataset: str) -> Dict[str, list[str]]:
    grouped: Dict[str, list[str]] = {name: [] for name in CODE_MODALITIES}
    types, codes = admission.trajectory
    for source_type, code in zip(types, codes):
        destination = TYPE_MAP[dataset].get(source_type)
        if destination is None:
            raise ValueError(f"Unexpected {dataset} code source {source_type!r}")
        grouped[destination].append(code)
    return grouped


def code_vector(codes: Sequence[str], vocab: object, embeddings: np.ndarray) -> tuple[np.ndarray, float]:
    if not codes:
        return np.zeros(CODE_DIM + 1, dtype=np.float32), 0.0
    indices = np.fromiter((vocab(code) for code in codes), dtype=np.int64, count=len(codes))
    pooled = embeddings[indices].mean(axis=0, dtype=np.float32)
    result = np.empty(CODE_DIM + 1, dtype=np.float32)
    result[:-1] = pooled
    result[-1] = np.log1p(len(codes)) / np.log1p(512.0)
    return result, 1.0


def lab_vector(labvectors: object, dataset: str) -> tuple[np.ndarray, float]:
    if labvectors is None:
        return np.zeros(LAB_DIM, dtype=np.float32), 0.0
    values = np.asarray(labvectors, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        return np.zeros(LAB_DIM, dtype=np.float32), 0.0
    if dataset == "mimic4":
        if values.shape[1] % 2:
            raise ValueError(f"MIMIC lab width must be even, got {values.shape}")
        values = values[:, 0::2]
    finite = np.isfinite(values)
    observed = finite & (np.abs(values) > 1e-8)
    observed_values = values[observed]
    last = values[-1]
    last_observed = np.isfinite(last) & (np.abs(last) > 1e-8)
    last_values = last[last_observed]

    def stats(array: np.ndarray) -> list[float]:
        if array.size == 0:
            return [0.0] * 5
        return [
            float(np.mean(array)),
            float(np.std(array)),
            float(np.min(array)),
            float(np.median(array)),
            float(np.max(array)),
        ]

    consecutive = observed[1:] & observed[:-1]
    changes = np.abs(values[1:] - values[:-1])[consecutive] if len(values) > 1 else np.empty(0, dtype=np.float32)
    summary = np.asarray(
        stats(observed_values)
        + stats(last_values)
        + [
            np.log1p(values.shape[0]) / np.log1p(50.0),
            float(observed.mean()),
            float(last_observed.mean()),
            float(np.mean(changes)) if changes.size else 0.0,
        ],
        dtype=np.float32,
    )
    if summary.shape != (LAB_DIM,) or not np.isfinite(summary).all():
        raise ValueError(f"Invalid lab summary for {dataset}: {summary}")
    return summary, 1.0


def _load_pickle(path: Path, muse_src: Path) -> object:
    if str(muse_src) not in sys.path:
        sys.path.insert(0, str(muse_src))
    with path.open("rb") as handle:
        return pickle.load(handle)


def _patient_id(admission: object) -> str:
    return str(getattr(admission, "patient_id"))


def _atomic_json(data: Mapping[str, object], path: Path) -> None:
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def build_dataset_cache(
    dataset: str,
    splits: Iterable[str],
    paths: FeaturePaths,
    task: str = "mortality",
) -> Dict[str, object]:
    if dataset not in TYPE_MAP:
        raise ValueError(dataset)
    dataset_dir = paths.data_root / dataset
    object_name = "hosp_adm_dict_muse_full.pkl" if dataset == "mimic4" else "icu_stay_dict.pkl"
    object_path = dataset_dir / object_name
    embedding_path = dataset_dir / "embeddings.txt"
    vocab_path = dataset_dir / "vocab.pkl"
    admissions = _load_pickle(object_path, paths.muse_src)
    vocab = _load_pickle(vocab_path, paths.muse_src)
    embeddings = read_embedding_matrix(embedding_path)
    if len(vocab) != embeddings.shape[0]:
        raise ValueError(f"{dataset}: vocab {len(vocab)} != embedding rows {len(embeddings)}")

    audit: Dict[str, object] = {
        "dataset": dataset,
        "task": task,
        "label_definition": (
            "death within 90 days after a live hospital discharge"
            if dataset == "mimic4"
            else "death by ICU discharge"
        ),
        "source_object": str(object_path),
        "source_embedding": str(embedding_path),
        "source_embedding_sha256": sha256_file(embedding_path),
        "source_vocab": str(vocab_path),
        "source_vocab_sha256": sha256_file(vocab_path),
        "embedding_model": "emilyalsentzer/Bio_ClinicalBERT pooler_output (legacy MUSE file)",
        "modalities": list(MODALITIES),
        "excluded_dataset_specific_modalities": ["discharge_note"] if dataset == "mimic4" else ["APACHE_APS"],
        "splits": {},
    }
    seen_ids: set[str] = set()
    seen_patients: Dict[str, set[str]] = {}

    for split in splits:
        ids_path = dataset_dir / f"task:{task}" / f"{split}_admission_ids.txt"
        identifiers = read_ids(ids_path)
        duplicate_ids = seen_ids.intersection(identifiers)
        if duplicate_ids:
            raise ValueError(f"{dataset}: {len(duplicate_ids)} IDs overlap earlier splits")
        seen_ids.update(identifiers)
        split_dir = paths.output_root / dataset / split
        split_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = split_dir / f".building-{os.getpid()}"
        temp_dir.mkdir(parents=True, exist_ok=False)
        count = len(identifiers)
        arrays = {
            "demographics": np.lib.format.open_memmap(temp_dir / "demographics.npy", mode="w+", dtype=np.float32, shape=(count, 11)),
            "diagnosis": np.lib.format.open_memmap(temp_dir / "diagnosis.npy", mode="w+", dtype=np.float16, shape=(count, CODE_DIM + 1)),
            "procedure": np.lib.format.open_memmap(temp_dir / "procedure.npy", mode="w+", dtype=np.float16, shape=(count, CODE_DIM + 1)),
            "medication": np.lib.format.open_memmap(temp_dir / "medication.npy", mode="w+", dtype=np.float16, shape=(count, CODE_DIM + 1)),
            "labs": np.lib.format.open_memmap(temp_dir / "labs.npy", mode="w+", dtype=np.float32, shape=(count, LAB_DIM)),
            "mask": np.lib.format.open_memmap(temp_dir / "mask.npy", mode="w+", dtype=np.uint8, shape=(count, len(MODALITIES))),
            "labels": np.lib.format.open_memmap(temp_dir / "labels.npy", mode="w+", dtype=np.uint8, shape=(count,)),
        }
        patients: list[str] = []
        positives = 0
        for row, identifier in enumerate(identifiers):
            admission = admissions[identifier]
            arrays["demographics"][row] = demographic_vector(admission)
            arrays["mask"][row, 0] = 1
            grouped = split_codes(admission, dataset)
            for modality_index, modality in enumerate(CODE_MODALITIES, start=1):
                vector, available = code_vector(grouped[modality], vocab, embeddings)
                arrays[modality][row] = vector
                arrays["mask"][row, modality_index] = int(available)
            vector, available = lab_vector(getattr(admission, "labvectors", None), dataset)
            arrays["labs"][row] = vector
            arrays["mask"][row, 4] = int(available)
            label = int(bool(getattr(admission, task)))
            arrays["labels"][row] = label
            positives += label
            patients.append(_patient_id(admission))
            if (row + 1) % 10000 == 0:
                print(f"{dataset}/{split}: {row + 1}/{count}", flush=True)

        for array in arrays.values():
            array.flush()
        (temp_dir / "ids.txt").write_text("\n".join(identifiers) + "\n", encoding="utf-8")
        (temp_dir / "patient_ids.txt").write_text("\n".join(patients) + "\n", encoding="utf-8")
        for child in temp_dir.iterdir():
            os.replace(child, split_dir / child.name)
        temp_dir.rmdir()
        split_patients = set(patients)
        overlap = {name: len(split_patients & values) for name, values in seen_patients.items()}
        seen_patients[split] = split_patients
        mask = np.load(split_dir / "mask.npy", mmap_mode="r")
        audit["splits"][split] = {
            "n": count,
            "positives": positives,
            "prevalence": positives / max(count, 1),
            "unique_patients": len(split_patients),
            "patient_overlap_with_prior_splits": overlap,
            "modality_availability": {
                modality: float(mask[:, index].mean()) for index, modality in enumerate(MODALITIES)
            },
        }

    audit_path = paths.output_root / dataset / "audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(audit, audit_path)
    return audit


def validate_cache(root: str | Path, dataset: str, splits: Sequence[str]) -> Dict[str, object]:
    root = Path(root)
    audit = json.loads((root / dataset / "audit.json").read_text(encoding="utf-8"))
    dims = {"demographics": 11, "diagnosis": 769, "procedure": 769, "medication": 769, "labs": LAB_DIM}
    for split in splits:
        labels = np.load(root / dataset / split / "labels.npy", mmap_mode="r")
        mask = np.load(root / dataset / split / "mask.npy", mmap_mode="r")
        if mask.shape != (len(labels), len(MODALITIES)):
            raise ValueError(f"Bad mask shape for {dataset}/{split}: {mask.shape}")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError(f"Non-binary labels in {dataset}/{split}")
        for modality, dim in dims.items():
            values = np.load(root / dataset / split / f"{modality}.npy", mmap_mode="r")
            if values.shape != (len(labels), dim):
                raise ValueError(f"Bad {modality} shape for {dataset}/{split}: {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite {modality} values in {dataset}/{split}")
    return audit
