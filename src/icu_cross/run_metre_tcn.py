from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .metre_tcn import METRETemporalConv
from .run_experiment import binary_metrics


TEMPORAL_MODALITIES = ("bedside", "laboratory", "medications", "procedures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the public-code METRE TCN on the aligned ICU cache.")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--channels", type=int, nargs="+", default=[256, 256, 256, 256])
    return parser.parse_args()


class METREDataset(Dataset):
    def __init__(self, root: Path, dataset: str, split: str):
        directory = root / dataset / split
        self.values = [
            np.load(directory / f"{name}.npy", mmap_mode="r")
            for name in TEMPORAL_MODALITIES
        ]
        self.labels = np.load(directory / "labels.npy", mmap_mode="r")
        self.ids = (directory / "ids.txt").read_text(encoding="utf-8").splitlines()
        if len(self.ids) != len(self.labels):
            raise ValueError(f"ID/label mismatch in {dataset}/{split}")
        trailing = {array.shape[1] for array in self.values}
        if trailing != {48}:
            raise ValueError(f"METRE expects 48 hourly bins, found {trailing}")
        if sum(array.shape[2] for array in self.values) != 200:
            raise ValueError("METRE expects exactly 200 time-dependent input channels")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        values = np.concatenate(
            [np.asarray(array[index], dtype=np.float32) for array in self.values], axis=1
        )
        return (
            torch.from_numpy(np.array(values.T, dtype=np.float32, copy=True)),
            torch.tensor(int(self.labels[index]), dtype=torch.long),
            self.ids[index],
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    identifiers: list[str] = []
    for values, target, ids in loader:
        logits = model(values.to(device, non_blocking=True))[:, -1, :]
        labels.append(target.numpy())
        probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        identifiers.extend(ids)
    return np.concatenate(labels), np.concatenate(probabilities), identifiers


def write_predictions(
    path: Path,
    identifiers: Sequence[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label", "probability"])
        writer.writerows(
            (identifier, int(label), float(probability))
            for identifier, label, probability in zip(identifiers, labels, probabilities)
        )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    cache_root, output_dir = Path(args.cache_root), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("temporal_channel_audit", {}).get("total") != 200:
        raise ValueError("Cache is not compatible with the public METRE 200-channel schema")

    datasets = {
        "mimic_train": METREDataset(cache_root, "mimic4", "train"),
        "mimic_val": METREDataset(cache_root, "mimic4", "val"),
        "mimic_test": METREDataset(cache_root, "mimic4", "test"),
        "eicu_test": METREDataset(cache_root, "eicu", "test"),
    }
    labels = np.asarray(datasets["mimic_train"].labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=2)
    class_weights = len(labels) / (2.0 * np.maximum(counts, 1))
    sample_weights = torch.as_tensor(class_weights[labels], dtype=torch.double)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(labels), replacement=True, generator=generator
    )
    loaders = {
        "mimic_train": DataLoader(
            datasets["mimic_train"], batch_size=args.batch_size, sampler=sampler,
            num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
        ),
        **{
            name: DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
            )
            for name, dataset in datasets.items() if name != "mimic_train"
        },
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU is required")
    model = METRETemporalConv(
        inputs=200, channels=args.channels, kernel_size=args.kernel_size,
        dropout=args.dropout, classes=2,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    config = {
        **vars(args),
        "model": "METRE public-code TCN",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "input_modalities_concatenated": list(TEMPORAL_MODALITIES),
        "input_shape": [200, 48],
        "static_input_used": False,
        "sampler": "official class-balanced WeightedRandomSampler",
        "checkpoint_selection": "MIMIC validation AUROC only",
        "protocol_difference_from_public_code": (
            "fixed leakage-free train/validation/test split; no test-set checkpoint selection; one seed"
        ),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    best_auroc = -np.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, float | int]] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for values, target, _ in loaders["mimic_train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(values.to(device, non_blocking=True))[:, -1, :]
            loss = F.cross_entropy(logits, target.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        val_labels, val_probabilities, _ = predict(model, loaders["mimic_val"], device)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(batches, 1),
            "mimic_val_auroc": float(roc_auc_score(val_labels, val_probabilities)),
            "mimic_val_auprc": float(average_precision_score(val_labels, val_probabilities)),
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if record["mimic_val_auroc"] > best_auroc:
            best_auroc = float(record["mimic_val_auroc"])
            best_epoch = epoch
            stale = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, output_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    metrics: dict[str, object] = {
        "model": "metre_tcn",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "selection_metric": "MIMIC validation AUROC",
        "classification_threshold": 0.5,
    }
    for name in ("mimic_val", "mimic_test", "eicu_test"):
        split_labels, probabilities, identifiers = predict(model, loaders[name], device)
        metrics[name] = binary_metrics(split_labels, probabilities, 0.5)
        write_predictions(
            output_dir / f"{name}_predictions.csv.gz", identifiers, split_labels, probabilities
        )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

