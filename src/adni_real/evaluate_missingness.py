from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ADNIDataset, FEATURE_GROUPS, load_and_transform_features
from .models import SPMNet
from .run_experiment import multiclass_metrics, move_batch


SCALAR_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "macro_auroc_ovr",
    "macro_auprc_ovr",
    "ece_10bin",
    "nll",
    "brier_multiclass",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ADNI SPMNet checkpoints under forced missing modalities.")
    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--split-csv", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--no-dropout-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--seeds", nargs="*", type=int, default=[2026, 2027, 2028, 2029, 2030])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--random-replicates", type=int, default=10)
    return parser.parse_args()


def scenario_names(modalities: Sequence[str], replicates: int) -> List[str]:
    names = ["natural"]
    names.extend(f"drop_{name.lower().replace(' ', '_')}" for name in modalities)
    names.extend(["mri_only", "tabular_only"])
    for drop_rate in (25, 50, 75):
        names.extend(f"random_non_mri_drop_{drop_rate:02d}_rep_{replicate:02d}" for replicate in range(replicates))
    return names


def stable_random_keep(subject_id: str, scenario: str, modality_count: int, drop_rate: float) -> np.ndarray:
    digest = hashlib.sha256(f"adni-spmnet-missing-v1|{scenario}|{subject_id}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(seed)
    keep = np.ones(modality_count, dtype=np.float32)
    keep[1:] = (rng.random(modality_count - 1) >= drop_rate).astype(np.float32)
    return keep


def forced_keep(
    scenario: str,
    subject_ids: Sequence[str],
    modalities: Sequence[str],
) -> np.ndarray:
    modality_count = len(modalities)
    keep = np.ones((len(subject_ids), modality_count), dtype=np.float32)
    if scenario == "natural":
        return keep
    if scenario == "mri_only":
        keep[:, 1:] = 0.0
        return keep
    if scenario == "tabular_only":
        keep[:, 0] = 0.0
        return keep
    if scenario.startswith("drop_"):
        target = scenario.removeprefix("drop_")
        slugs = [name.lower().replace(" ", "_") for name in modalities]
        keep[:, slugs.index(target)] = 0.0
        return keep
    prefix = "random_non_mri_drop_"
    if scenario.startswith(prefix):
        drop_rate = int(scenario[len(prefix) : len(prefix) + 2]) / 100.0
        return np.stack(
            [stable_random_keep(subject_id, scenario, modality_count, drop_rate) for subject_id in subject_ids],
            axis=0,
        )
    raise ValueError(f"Unknown missingness scenario: {scenario}")


def default_resolved_config() -> Dict[str, object]:
    return {
        "incidence_mode": "learned",
        "task_mode": "learned",
        "fusion": "poe",
        "use_private": True,
        "direct_bypass": False,
    }


def build_model(checkpoint: Mapping[str, object], group_dims: Sequence[int]) -> SPMNet:
    saved_args = checkpoint.get("args", {})
    resolved = default_resolved_config()
    resolved.update(checkpoint.get("resolved_spmnet", {}))
    model = SPMNet(
        group_dims,
        hidden=int(saved_args.get("hidden", 128)),
        latent=int(saved_args.get("latent", 32)),
        private=int(saved_args.get("private", 8)),
        incidence_mode=str(resolved["incidence_mode"]),
        task_mode=str(resolved["task_mode"]),
        fusion=str(resolved["fusion"]),
        use_private=bool(resolved["use_private"]),
        direct_bypass=bool(resolved["direct_bypass"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    loader: DataLoader,
    device: torch.device,
    group_names: Sequence[str],
    scenarios: Sequence[str],
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, float]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    group_dims = [loader.dataset.groups[name].shape[1] for name in group_names]
    model = build_model(checkpoint, group_dims).to(device)
    model.eval()
    probabilities: Dict[str, List[np.ndarray]] = {scenario: [] for scenario in scenarios}
    labels_all: List[np.ndarray] = []
    availability_totals = {scenario: 0.0 for scenario in scenarios}
    availability_denominator = 0
    modalities = ["T1 MRI", *group_names]

    for batch in loader:
        image, groups, natural_availability, labels = move_batch(batch, device, group_names)
        with torch.cuda.amp.autocast(enabled=True):
            embeddings = model.stem(image, groups)
            for scenario in scenarios:
                keep_np = forced_keep(scenario, list(batch["subject_id"]), modalities)
                keep = torch.from_numpy(keep_np).to(device=device, dtype=natural_availability.dtype)
                observed = natural_availability * keep
                logits = model.forward_from_embeddings(embeddings, observed, sample=False)["logits"]
                probabilities[scenario].append(torch.softmax(logits, dim=1).cpu().numpy())
                availability_totals[scenario] += float(observed.sum().cpu())
        labels_all.append(labels.cpu().numpy())
        availability_denominator += labels.numel() * len(modalities)

    labels_np = np.concatenate(labels_all)
    metrics = {
        scenario: multiclass_metrics(labels_np, np.concatenate(chunks))
        for scenario, chunks in probabilities.items()
    }
    observed_fractions = {
        scenario: availability_totals[scenario] / max(1, availability_denominator)
        for scenario in scenarios
    }
    return metrics, observed_fractions


def mean_sd(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A CUDA GPU is required for missing-modality evaluation")

    frame, groups, availability, _ = load_and_transform_features(args.master_csv, args.split_csv)
    group_names = list(FEATURE_GROUPS)
    dataset = ADNIDataset(frame, groups, availability, "test", train=False, seed=0)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    modalities = ["T1 MRI", *group_names]
    scenarios = scenario_names(modalities, args.random_replicates)
    roots = {
        "full": Path(args.baseline_root) / "spmnet",
        "no_modality_dropout": Path(args.no_dropout_root),
    }

    nested: Dict[str, Dict[int, Dict[str, Dict[str, object]]]] = {}
    observed_nested: Dict[str, Dict[int, Dict[str, float]]] = {}
    rows: List[Dict[str, object]] = []
    for model_name, root in roots.items():
        nested[model_name] = {}
        observed_nested[model_name] = {}
        for seed in args.seeds:
            metrics, observed = evaluate_checkpoint(
                root / f"seed_{seed}" / "best.pt",
                loader,
                device,
                group_names,
                scenarios,
            )
            nested[model_name][seed] = metrics
            observed_nested[model_name][seed] = observed
            for scenario in scenarios:
                row: Dict[str, object] = {
                    "model": model_name,
                    "seed": seed,
                    "scenario": scenario,
                    "observed_fraction": observed[scenario],
                }
                row.update({metric: metrics[scenario][metric] for metric in SCALAR_METRICS})
                rows.append(row)

    aggregate: Dict[str, object] = {}
    for model_name in roots:
        aggregate[model_name] = {}
        for scenario in scenarios:
            aggregate[model_name][scenario] = {
                "observed_fraction": mean_sd(
                    np.asarray([observed_nested[model_name][seed][scenario] for seed in args.seeds])
                ),
                **{
                    metric: mean_sd(
                        np.asarray([nested[model_name][seed][scenario][metric] for seed in args.seeds])
                    )
                    for metric in SCALAR_METRICS
                },
            }

    robustness_curves: Dict[str, object] = {}
    for model_name in roots:
        robustness_curves[model_name] = {}
        for metric in SCALAR_METRICS:
            points: List[Dict[str, float]] = []
            for drop_rate in (0, 25, 50, 75):
                if drop_rate == 0:
                    values = np.asarray(
                        [nested[model_name][seed]["natural"][metric] for seed in args.seeds], dtype=np.float64
                    )
                else:
                    values = np.asarray(
                        [
                            np.mean(
                                [
                                    nested[model_name][seed][f"random_non_mri_drop_{drop_rate:02d}_rep_{rep:02d}"][metric]
                                    for rep in range(args.random_replicates)
                                ]
                            )
                            for seed in args.seeds
                        ],
                        dtype=np.float64,
                    )
                points.append({"available_fraction": 1.0 - drop_rate / 100.0, **mean_sd(values)})
            x = np.asarray([point["available_fraction"] for point in reversed(points)])
            y = np.asarray([point["mean"] for point in reversed(points)])
            robustness_curves[model_name][metric] = {
                "points": points,
                "area_under_availability_curve": float(np.trapz(y, x) / (x[-1] - x[0])),
            }

    summary = {
        "status": "complete",
        "seeds": list(args.seeds),
        "modalities": modalities,
        "scenario_count": len(scenarios),
        "random_mask_policy": "Subject-specific deterministic SHA-256 masks; MRI retained in random non-MRI stress tests.",
        "aggregate": aggregate,
        "robustness_curves": robustness_curves,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "complete", "scenario_count": len(scenarios), "seeds": args.seeds}, indent=2))


if __name__ == "__main__":
    main()

