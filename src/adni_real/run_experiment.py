from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.utils.data import DataLoader

from .data import ADNIDataset, FEATURE_GROUPS, LABEL_NAMES, load_and_transform_features, prepare_split_manifest
from .models import MultimodalTransformer, SPMNet


ABLATIONS = (
    "full",
    "no_incidence",
    "no_task_mask",
    "no_private",
    "no_reconstruction",
    "no_sparsity",
    "no_witness",
    "no_modality_dropout",
    "mean_fusion",
    "direct_bypass",
)


def resolve_spmnet_configuration(args: argparse.Namespace) -> Dict[str, object]:
    """Resolve one controlled ablation into explicit model and loss settings."""
    config: Dict[str, object] = {
        "incidence_mode": "learned",
        "task_mode": "learned",
        "fusion": "poe",
        "use_private": True,
        "direct_bypass": False,
        "reconstruction_weight": float(args.reconstruction_weight),
        "kl_weight": float(args.kl_weight),
        "incidence_sparsity_weight": float(args.sparsity_weight),
        "task_sparsity_weight": float(args.sparsity_weight),
        "witness_weight": float(args.witness_weight),
        "task_floor_weight": float(args.task_floor_weight),
        "modality_dropout": float(args.modality_dropout),
    }
    if args.ablation == "no_incidence":
        config.update(incidence_mode="all", incidence_sparsity_weight=0.0, witness_weight=0.0)
    elif args.ablation == "no_task_mask":
        config.update(task_mode="all", task_sparsity_weight=0.0, task_floor_weight=0.0)
    elif args.ablation == "no_private":
        config["use_private"] = False
    elif args.ablation == "no_reconstruction":
        config["reconstruction_weight"] = 0.0
    elif args.ablation == "no_sparsity":
        config.update(incidence_sparsity_weight=0.0, task_sparsity_weight=0.0)
    elif args.ablation == "no_witness":
        config["witness_weight"] = 0.0
    elif args.ablation == "no_modality_dropout":
        config["modality_dropout"] = 0.0
    elif args.ablation == "mean_fusion":
        config["fusion"] = "mean"
    elif args.ablation == "direct_bypass":
        config["direct_bypass"] = True
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SPMNet or a matched multimodal Transformer on ADNI.")
    parser.add_argument("--model", choices=("spmnet", "transformer"), default="spmnet")
    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--split-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--private", type=int, default=8)
    parser.add_argument("--ablation", choices=ABLATIONS, default="full")
    parser.add_argument("--modality-dropout", type=float, default=0.15)
    parser.add_argument("--reconstruction-weight", type=float, default=0.20)
    parser.add_argument("--kl-weight", type=float, default=0.002)
    parser.add_argument("--sparsity-weight", type=float, default=0.005)
    parser.add_argument("--witness-weight", type=float, default=0.10)
    parser.add_argument("--task-floor-weight", type=float, default=0.05)
    parser.add_argument("--exclude-mri", action="store_true", help="Mask T1 MRI for all splits while retaining tabular groups.")
    parser.add_argument(
        "--exclude-groups",
        nargs="*",
        choices=tuple(FEATURE_GROUPS),
        default=(),
        help="Tabular feature groups to remove while retaining MRI.",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.model != "spmnet" and args.ablation != "full":
        parser.error("--ablation is only valid with --model spmnet")
    for name in (
        "modality_dropout",
        "reconstruction_weight",
        "kl_weight",
        "sparsity_weight",
        "witness_weight",
        "task_floor_weight",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be nonnegative")
    if args.modality_dropout >= 1:
        parser.error("--modality-dropout must be less than 1")
    return args


def multiclass_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, object]:
    predictions = probabilities.argmax(axis=1)
    binary = label_binarize(labels, classes=np.arange(len(LABEL_NAMES)))
    matrix = confusion_matrix(labels, predictions, labels=np.arange(len(LABEL_NAMES)))
    recalls = recall_score(labels, predictions, labels=np.arange(len(LABEL_NAMES)), average=None, zero_division=0)
    specificity: List[float] = []
    for index in range(len(LABEL_NAMES)):
        tp = matrix[index, index]
        fn = matrix[index].sum() - tp
        fp = matrix[:, index].sum() - tp
        tn = matrix.sum() - tp - fn - fp
        specificity.append(float(tn / max(1, tn + fp)))
    try:
        auroc = float(roc_auc_score(binary, probabilities, average="macro", multi_class="ovr"))
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(binary, probabilities, average="macro"))
    except ValueError:
        auprc = float("nan")
    confidence = probabilities.max(axis=1)
    correctness = (predictions == labels).astype(np.float64)
    clipped = np.clip(probabilities, 1e-8, 1.0)
    nll = float(-np.log(clipped[np.arange(len(labels)), labels]).mean())
    one_hot = np.eye(len(LABEL_NAMES), dtype=np.float64)[labels]
    brier = float(np.square(probabilities - one_hot).sum(axis=1).mean())
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        selected = (confidence >= lower) & (confidence < lower + 0.1)
        if selected.any():
            ece += float(selected.mean()) * abs(float(correctness[selected].mean()) - float(confidence[selected].mean()))
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "macro_auroc_ovr": auroc,
        "macro_auprc_ovr": auprc,
        "nll": nll,
        "brier_multiclass": brier,
        "ece_10bin": ece,
        "sensitivity_by_class": {name: float(value) for name, value in zip(LABEL_NAMES, recalls)},
        "specificity_by_class": {name: float(value) for name, value in zip(LABEL_NAMES, specificity)},
        "confusion_matrix": matrix.tolist(),
    }


def move_batch(
    batch: Mapping[str, object],
    device: torch.device,
    group_names: Sequence[str],
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
    image = batch["image"].to(device, non_blocking=True)
    group_mapping = batch["groups"]
    groups = [group_mapping[name].to(device, non_blocking=True) for name in group_names]
    availability = batch["availability"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True)
    return image, groups, availability, labels


def spmnet_loss(
    model: SPMNet,
    output: Mapping[str, object],
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    weights: Mapping[str, float],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    logits = output["logits"]
    classification = F.cross_entropy(logits, labels, weight=class_weights)
    embeddings = output["embeddings"]
    reconstructions = output["reconstructions"]
    availability = output["effective_availability"]
    reconstruction = logits.new_zeros(())
    denominator = availability.sum().clamp_min(1.0)
    for index, (target, predicted) in enumerate(zip(embeddings, reconstructions)):
        per_sample = (predicted - target.detach()).pow(2).mean(dim=1)
        reconstruction = reconstruction + (per_sample * availability[:, index]).sum() / denominator
    z_mu = output["z_mu"]
    z_logvar = output["z_logvar"]
    kl = 0.5 * (torch.exp(z_logvar) + z_mu.pow(2) - 1.0 - z_logvar).sum(dim=1).mean()
    regularization = model.regularization()
    total = (
        classification
        + weights["reconstruction_weight"] * reconstruction
        + weights["kl_weight"] * kl
        + weights["incidence_sparsity_weight"] * regularization["incidence_sparsity"]
        + weights["task_sparsity_weight"] * regularization["task_sparsity"]
        + weights["witness_weight"] * regularization["witness"]
        + weights["task_floor_weight"] * regularization["task_floor"]
    )
    parts = {
        "classification": float(classification.detach()),
        "reconstruction": float(reconstruction.detach()),
        "kl": float(kl.detach()),
        "incidence_sparsity": float(regularization["incidence_sparsity"].detach()),
        "task_sparsity": float(regularization["task_sparsity"].detach()),
        "witness": float(regularization["witness"].detach()),
        "task_floor": float(regularization["task_floor"].detach()),
    }
    return total, parts


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
    group_names: Sequence[str],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    model.eval()
    labels_all: List[np.ndarray] = []
    probabilities_all: List[np.ndarray] = []
    rows: List[Dict[str, object]] = []
    for batch in loader:
        image, groups, availability, labels = move_batch(batch, device, group_names)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            if model_name == "spmnet":
                logits = model(image, groups, availability, sample=False)["logits"]
            else:
                logits = model(image, groups, availability)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        label_values = labels.cpu().numpy()
        labels_all.append(label_values)
        probabilities_all.append(probabilities)
        for subject, label, probability in zip(batch["subject_id"], label_values, probabilities):
            rows.append(
                {
                    "subject_id": subject,
                    "label": int(label),
                    "label_name": LABEL_NAMES[int(label)],
                    "prediction": int(np.argmax(probability)),
                    "prediction_name": LABEL_NAMES[int(np.argmax(probability))],
                    **{f"prob_{name}": float(value) for name, value in zip(LABEL_NAMES, probability)},
                }
            )
    labels_np = np.concatenate(labels_all)
    probabilities_np = np.concatenate(probabilities_all)
    return multiclass_metrics(labels_np, probabilities_np), rows


def write_predictions(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    spmnet_config = resolve_spmnet_configuration(args) if args.model == "spmnet" else {}
    split_path = Path(args.split_csv)
    if args.prepare_only or not split_path.exists():
        prepare_split_manifest(args.master_csv, args.cache_root, split_path, args.split_summary, args.seed)
    if args.prepare_only:
        return

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_configuration = {
        "cli": vars(args),
        "resolved_spmnet": spmnet_config,
        "selection_metric": "validation macro_auroc_ovr",
        "test_selection_prohibited": True,
    }
    (output_dir / "config.json").write_text(
        json.dumps(run_configuration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    frame, all_groups, all_availability, all_statistics = load_and_transform_features(args.master_csv, split_path)
    excluded_groups = set(args.exclude_groups)
    group_names = [name for name in FEATURE_GROUPS if name not in excluded_groups]
    if not group_names:
        raise ValueError("At least one tabular feature group must remain; MRI is always retained")
    groups = {name: all_groups[name] for name in group_names}
    availability_columns = [0] + [index + 1 for index, name in enumerate(FEATURE_GROUPS) if name in group_names]
    availability = all_availability[:, availability_columns].copy()
    if args.exclude_mri:
        availability[:, 0] = 0.0
    statistics = {name: all_statistics[name] for name in group_names}
    statistics["configuration"] = {
        "image_modality": None if args.exclude_mri else "T1 MRI",
        "mri_forced_unavailable": bool(args.exclude_mri),
        "included_tabular_groups": group_names,
        "excluded_tabular_groups": sorted(excluded_groups),
    }
    (output_dir / "feature_statistics.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    missing_cache = [path for path in frame["image_cache"].astype(str) if not Path(path).is_file()]
    if missing_cache:
        raise FileNotFoundError(f"{len(missing_cache)} cached MRI arrays are missing; first={missing_cache[0]}")

    datasets = {
        split: ADNIDataset(frame, groups, availability, split, train=(split == "train"), seed=args.seed)
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
            drop_last=(split == "train"),
        )
        for split, dataset in datasets.items()
    }

    group_dims = [groups[name].shape[1] for name in group_names]
    if args.model == "spmnet":
        model: nn.Module = SPMNet(
            group_dims,
            hidden=args.hidden,
            latent=args.latent,
            private=args.private,
            incidence_mode=str(spmnet_config["incidence_mode"]),
            task_mode=str(spmnet_config["task_mode"]),
            fusion=str(spmnet_config["fusion"]),
            use_private=bool(spmnet_config["use_private"]),
            direct_bypass=bool(spmnet_config["direct_bypass"]),
        )
    else:
        model = MultimodalTransformer(group_dims, hidden=args.hidden)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU is required for this experiment")
    model.to(device)

    train_labels = frame.loc[frame["split"].eq("train"), "label"].to_numpy(dtype=int)
    counts = np.bincount(train_labels, minlength=len(LABEL_NAMES))
    class_weights = torch.tensor(len(train_labels) / (len(LABEL_NAMES) * counts), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_score = -math.inf
    best_epoch = -1
    stale = 0
    history: List[Dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        part_totals: Dict[str, float] = {}
        batches = 0
        for batch in loaders["train"]:
            image, group_values, observed, labels = move_batch(batch, device, group_names)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                if args.model == "spmnet":
                    output = model(
                        image,
                        group_values,
                        observed,
                        sample=True,
                        modality_dropout=float(spmnet_config["modality_dropout"]),
                    )
                    loss, parts = spmnet_loss(model, output, labels, class_weights, spmnet_config)
                    for name, value in parts.items():
                        part_totals[name] = part_totals.get(name, 0.0) + value
                else:
                    logits = model(image, group_values, observed)
                    loss = F.cross_entropy(logits, labels, weight=class_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach())
            batches += 1
        scheduler.step()
        val_metrics, _ = evaluate(model, loaders["val"], device, args.model, group_names)
        score = float(val_metrics["macro_auroc_ovr"])
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, batches),
            "lr": scheduler.get_last_lr()[0],
            **{f"train_{name}": value / max(1, batches) for name, value in part_totals.items()},
            **val_metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if score > best_score + 1e-4:
            best_score = score
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "resolved_spmnet": spmnet_config,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                output_dir / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    result: Dict[str, object] = {
        "model": args.model,
        "ablation": args.ablation,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "modalities": ([] if args.exclude_mri else ["T1 MRI"]) + group_names,
        "mri_forced_unavailable": bool(args.exclude_mri),
        "excluded_groups": sorted(excluded_groups),
        "configuration": run_configuration,
        "splits": {},
    }
    for split in ("val", "test"):
        metrics, predictions = evaluate(model, loaders[split], device, args.model, group_names)
        result["splits"][split] = metrics
        write_predictions(output_dir / f"predictions_{split}.csv", predictions)
    if args.model == "spmnet":
        soft_incidence = model.incidence().detach().cpu()
        soft_task = model.task_mask().detach().cpu()
        hard_incidence = (soft_incidence >= 0.5).to(torch.int64)
        hard_task = (soft_task >= 0.5).to(torch.int64)
        result["structure"] = {
            "incidence": soft_incidence.tolist(),
            "task_mask": soft_task.tolist(),
            "hard_incidence": hard_incidence.tolist(),
            "hard_task_mask": hard_task.tolist(),
            "active_incidence_fraction": float(hard_incidence.float().mean()),
            "active_task_dimensions": int(hard_task.sum()),
            "active_modalities_per_latent": hard_incidence.sum(dim=0).tolist(),
            **{name: float(value.detach()) for name, value in model.regularization().items()},
        }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
