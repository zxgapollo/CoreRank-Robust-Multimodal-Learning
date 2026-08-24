from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader

from .data import ICUFeatureDataset, cache_modalities, fit_source_statistics, save_statistics
from .features import MODALITIES, validate_cache
from .models import MultimodalTransformer, SPMNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MIMIC-IV to eICU five-modality mortality transfer.")
    parser.add_argument("--model", choices=("spmnet", "transformer"), required=True)
    parser.add_argument("--encoder", choices=("matched", "metre_shared"), default="matched")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--metre-channels", type=int, nargs="+", default=[256, 256, 256, 256])
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--modality-dropout", type=float, default=0.15)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ece_score(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    score = 0.0
    for lower in np.linspace(0.0, 1.0, bins, endpoint=False):
        selected = (probabilities >= lower) & (probabilities < lower + 1.0 / bins)
        if selected.any():
            score += float(selected.mean()) * abs(float(labels[selected].mean()) - float(probabilities[selected].mean()))
    return score


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    false_positive, true_positive, thresholds = roc_curve(labels, probabilities)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    index = int(np.argmax((true_positive - false_positive)[finite]))
    return float(thresholds[finite][index])


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> Dict[str, object]:
    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    prevalence = float(labels.mean())
    auprc = float(average_precision_score(labels, probabilities))
    return {
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "prevalence": prevalence,
        "threshold_from_mimic_val": float(threshold),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": auprc,
        "auprc_lift": auprc / max(prevalence, 1e-12),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "sensitivity": float(tp / max(1, tp + fn)),
        "specificity": float(tn / max(1, tn + fp)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece_10bin": ece_score(labels, probabilities),
        "confusion_matrix": matrix.tolist(),
    }


def move_batch(
    batch: Tuple[Sequence[torch.Tensor], torch.Tensor, torch.Tensor, Sequence[str]],
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, Sequence[str]]:
    modalities, availability, labels, identifiers = batch
    return (
        [value.to(device, non_blocking=True) for value in modalities],
        availability.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
        identifiers,
    )


def spmnet_loss(
    model: SPMNet,
    output: Mapping[str, object],
    labels: torch.Tensor,
    pos_weight: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, float]]:
    logits = output["logits"]
    classification = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    availability = output["effective_availability"]
    reconstruction = logits.new_zeros(())
    denominator = availability.sum().clamp_min(1.0)
    for index, (target, predicted) in enumerate(zip(output["embeddings"], output["reconstructions"])):
        per_sample = (predicted - target.detach()).pow(2).mean(dim=1)
        reconstruction = reconstruction + (per_sample * availability[:, index]).sum() / denominator
    z_mu = output["z_mu"]
    z_logvar = output["z_logvar"]
    kl = 0.5 * (torch.exp(z_logvar) + z_mu.pow(2) - 1.0 - z_logvar).sum(dim=1).mean()
    regularization = model.regularization()
    total = (
        classification
        + 0.10 * reconstruction
        + 0.002 * kl
        + 0.005 * regularization["sparsity"]
        + 0.10 * regularization["witness"]
        + 0.05 * regularization["task_floor"]
    )
    return total, {
        "classification": float(classification.detach()),
        "reconstruction": float(reconstruction.detach()),
        "kl": float(kl.detach()),
        "sparsity": float(regularization["sparsity"].detach()),
        "witness": float(regularization["witness"].detach()),
    }


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    all_labels: List[np.ndarray] = []
    all_probabilities: List[np.ndarray] = []
    all_ids: list[str] = []
    for batch in loader:
        modalities, availability, labels, identifiers = move_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            if model_name == "spmnet":
                logits = model(modalities, availability, sample=False)["logits"]
            else:
                logits = model(modalities, availability)
        all_labels.append(labels.cpu().numpy())
        all_probabilities.append(torch.sigmoid(logits).float().cpu().numpy())
        all_ids.extend(identifiers)
    return np.concatenate(all_labels), np.concatenate(all_probabilities), all_ids


def write_predictions(path: Path, ids: Sequence[str], labels: np.ndarray, probabilities: np.ndarray) -> None:
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label", "probability"])
        writer.writerows((identifier, int(label), float(probability)) for identifier, label, probability in zip(ids, labels, probabilities))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    cache_root = Path(args.cache_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modalities = cache_modalities(cache_root)
    manifest = (
        json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
        if (cache_root / "manifest.json").exists()
        else {}
    )
    temporal_names = set(manifest.get("temporal_modalities", ()))
    temporal_flags = [name in temporal_names for name in modalities]
    if modalities == MODALITIES:
        validate_cache(cache_root, "mimic4", ("train", "val", "test"))
        validate_cache(cache_root, "eicu", ("test",))
    statistics = fit_source_statistics(cache_root, modalities)
    save_statistics(statistics, output_dir / "source_normalization.json")

    datasets = {
        "mimic_train": ICUFeatureDataset(cache_root, "mimic4", "train", statistics, modalities),
        "mimic_val": ICUFeatureDataset(cache_root, "mimic4", "val", statistics, modalities),
        "mimic_test": ICUFeatureDataset(cache_root, "mimic4", "test", statistics, modalities),
        "eicu_test": ICUFeatureDataset(cache_root, "eicu", "test", statistics, modalities),
    }
    loaders = {
        name: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "mimic_train",
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
            drop_last=name == "mimic_train",
        )
        for name, dataset in datasets.items()
    }
    modality_dims = [datasets["mimic_train"].values[name].shape[-1] for name in modalities]
    model: nn.Module
    if args.model == "spmnet":
        model = SPMNet(
            modality_dims, hidden=args.hidden, latent=args.latent,
            temporal_modalities=temporal_flags,
            encoder_kind=args.encoder, metre_channels=args.metre_channels,
        )
    else:
        model = MultimodalTransformer(
            modality_dims, hidden=args.hidden, temporal_modalities=temporal_flags,
            encoder_kind=args.encoder, metre_channels=args.metre_channels,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU is required")
    model.to(device)

    train_labels = np.asarray(datasets["mimic_train"].labels, dtype=np.float32)
    positives = float(train_labels.sum())
    pos_weight = torch.tensor((len(train_labels) - positives) / max(positives, 1.0), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    config = vars(args).copy()
    config.update({
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "modalities": list(modalities),
        "temporal_modalities": sorted(temporal_names),
        "modality_dims": modality_dims,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "source_train_positive_weight": float(pos_weight),
        "protocol": "train MIMIC-IV only; select epoch/threshold on MIMIC-IV val only; zero-shot test on MIMIC-IV and eICU",
        "label_warning": (
            "MIMIC-IV target is 90-day post-discharge mortality; eICU target is ICU mortality in the supplied MUSE preprocessing."
            if not (cache_root / "manifest.json").exists()
            else json.loads((cache_root / "manifest.json").read_text(encoding="utf-8")).get("label_definition")
        ),
    })
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    best_score = -math.inf
    best_epoch = 0
    stale = 0
    history: List[Dict[str, object]] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in loaders["mimic_train"]:
            modalities, availability, labels, _ = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                if args.model == "spmnet":
                    output = model(
                        modalities,
                        availability,
                        sample=True,
                        modality_dropout=args.modality_dropout,
                    )
                    loss, _ = spmnet_loss(model, output, labels, pos_weight)
                else:
                    logits = model(modalities, availability)
                    loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            batches += 1
        scheduler.step()
        val_labels, val_probabilities, _ = predict(model, loaders["mimic_val"], device, args.model)
        val_auroc = float(roc_auc_score(val_labels, val_probabilities))
        val_auprc = float(average_precision_score(val_labels, val_probabilities))
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(batches, 1),
            "mimic_val_auroc": val_auroc,
            "mimic_val_auprc": val_auprc,
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if val_auprc > best_score + 1e-5:
            best_score = val_auprc
            best_epoch = epoch
            stale = 0
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, output_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    val_labels, val_probabilities, val_ids = predict(model, loaders["mimic_val"], device, args.model)
    threshold = select_threshold(val_labels, val_probabilities)
    metrics: Dict[str, object] = {
        "model": args.model,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "selection_metric": "MIMIC-IV validation AUPRC",
        "threshold_selection": "Youden J on MIMIC-IV validation only",
        "mimic_val": binary_metrics(val_labels, val_probabilities, threshold),
    }
    write_predictions(output_dir / "mimic_val_predictions.csv.gz", val_ids, val_labels, val_probabilities)
    for name in ("mimic_test", "eicu_test"):
        labels, probabilities, identifiers = predict(model, loaders[name], device, args.model)
        metrics[name] = binary_metrics(labels, probabilities, threshold)
        write_predictions(output_dir / f"{name}_predictions.csv.gz", identifiers, labels, probabilities)
    if args.model == "spmnet":
        metrics["structure"] = {
            "incidence_soft": model.incidence().detach().cpu().tolist(),
            "incidence_hard": (model.incidence() >= 0.5).int().detach().cpu().tolist(),
            "task_mask_soft": model.task_mask().detach().cpu().tolist(),
            "task_mask_hard": (model.task_mask() >= 0.5).int().detach().cpu().tolist(),
        }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
