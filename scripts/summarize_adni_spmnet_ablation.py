#!/usr/bin/env python3
"""Aggregate paired multi-seed SPMNet ablations for the ADNI age-OOD benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from scipy import stats


DEFAULT_VARIANTS = (
    "no_incidence",
    "no_task_mask",
    "no_private",
    "no_reconstruction",
    "no_sparsity",
    "no_witness",
    "no_modality_dropout",
    "mean_fusion",
    "direct_bypass",
    "no_demographics",
    "no_cognition",
    "no_behavior",
    "no_genetics_history",
    "no_mri",
    "latent_16",
    "latent_64",
    "private_4",
    "private_16",
    "modality_dropout_030",
)

METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "macro_auroc_ovr",
    "macro_auprc_ovr",
    "ece_10bin",
    "nll",
    "brier_multiclass",
)

LOWER_IS_BETTER = {"ece_10bin", "nll", "brier_multiclass"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--variants", nargs="*", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[2026, 2027, 2028, 2029, 2030])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-table", required=True)
    parser.add_argument("--output-per-seed", required=True)
    return parser.parse_args()


def proper_scores(prediction_path: Path) -> Dict[str, float]:
    rows = list(csv.DictReader(prediction_path.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"No prediction rows in {prediction_path}")
    labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    probabilities = np.asarray(
        [[float(row["prob_CN"]), float(row["prob_MCI"]), float(row["prob_AD"])] for row in rows],
        dtype=np.float64,
    )
    probabilities = np.clip(probabilities, 1e-8, 1.0)
    one_hot = np.eye(3, dtype=np.float64)[labels]
    return {
        "nll": float(-np.log(probabilities[np.arange(len(labels)), labels]).mean()),
        "brier_multiclass": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
    }


def load_run(path: Path, expected_seed: int) -> Dict[str, object]:
    metrics_path = path / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if int(payload["seed"]) != expected_seed:
        raise ValueError(f"Seed mismatch in {metrics_path}: {payload['seed']} != {expected_seed}")
    for split in ("val", "test"):
        scores = payload["splits"][split]
        missing = [metric for metric in METRICS if metric not in scores]
        if set(missing).issubset({"nll", "brier_multiclass"}):
            scores.update(proper_scores(path / f"predictions_{split}.csv"))
        still_missing = [metric for metric in METRICS if metric not in scores]
        if still_missing:
            raise ValueError(f"Missing metrics in {metrics_path}: {still_missing}")
    return payload


def mean_sd(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    }


def paired_summary(variant: np.ndarray, baseline: np.ndarray, lower_is_better: bool) -> Dict[str, object]:
    differences = variant - baseline
    n = len(differences)
    sd = float(differences.std(ddof=1)) if n > 1 else 0.0
    sem = float(stats.sem(differences)) if n > 1 else 0.0
    margin = float(stats.t.ppf(0.975, n - 1) * sem) if n > 1 else 0.0
    paired_t = stats.ttest_rel(variant, baseline)
    try:
        wilcoxon = stats.wilcoxon(differences, zero_method="wilcox", alternative="two-sided", method="auto")
        wilcoxon_result = {"statistic": float(wilcoxon.statistic), "p": float(wilcoxon.pvalue)}
    except ValueError:
        wilcoxon_result = {"statistic": None, "p": None}
    wins = differences < 0 if lower_is_better else differences > 0
    losses = differences > 0 if lower_is_better else differences < 0
    return {
        "difference_definition": "variant minus full SPMNet",
        "paired_differences": differences.tolist(),
        "mean_difference": float(differences.mean()),
        "sd_difference": sd,
        "cohens_dz": float(differences.mean() / sd) if sd > 0 else None,
        "mean_difference_95ci": [float(differences.mean() - margin), float(differences.mean() + margin)],
        "paired_t": {"statistic": float(paired_t.statistic), "p": float(paired_t.pvalue)},
        "wilcoxon": wilcoxon_result,
        "variant_wins": int(wins.sum()),
        "ties": int((differences == 0).sum()),
        "full_wins": int(losses.sum()),
    }


def holm_adjust(p_values: Mapping[str, float]) -> Dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: Dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        value = min(1.0, (count - rank) * p_values[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    baseline_root = Path(args.baseline_root) / "spmnet"
    seeds = list(args.seeds)
    variants = list(args.variants)

    runs: Dict[str, Dict[int, Dict[str, object]]] = {"full": {}}
    for seed in seeds:
        runs["full"][seed] = load_run(baseline_root / f"seed_{seed}", seed)
    for variant in variants:
        runs[variant] = {}
        for seed in seeds:
            runs[variant][seed] = load_run(run_root / variant / f"seed_{seed}", seed)

    per_seed_rows: List[Dict[str, object]] = []
    for variant, seed_payloads in runs.items():
        for seed in seeds:
            payload = seed_payloads[seed]
            for split in ("val", "test"):
                row: Dict[str, object] = {
                    "variant": variant,
                    "seed": seed,
                    "split": split,
                    "best_epoch": int(payload["best_epoch"]),
                    "parameter_count": int(payload["parameter_count"]),
                }
                row.update({metric: payload["splits"][split][metric] for metric in METRICS})
                per_seed_rows.append(row)

    aggregate: Dict[str, object] = {}
    paired: Dict[str, object] = {}
    table_rows: List[Dict[str, object]] = []
    primary_p_values: Dict[str, float] = {}
    for variant, seed_payloads in runs.items():
        aggregate[variant] = {
            split: {
                metric: mean_sd(
                    np.asarray([seed_payloads[seed]["splits"][split][metric] for seed in seeds], dtype=np.float64)
                )
                for metric in METRICS
            }
            for split in ("val", "test")
        }
        aggregate[variant]["test_minus_val"] = {
            metric: mean_sd(
                np.asarray(
                    [
                        seed_payloads[seed]["splits"]["test"][metric]
                        - seed_payloads[seed]["splits"]["val"][metric]
                        for seed in seeds
                    ],
                    dtype=np.float64,
                )
            )
            for metric in METRICS
        }
        aggregate[variant]["parameter_count"] = mean_sd(
            np.asarray([seed_payloads[seed]["parameter_count"] for seed in seeds], dtype=np.float64)
        )
        aggregate[variant]["best_epoch"] = mean_sd(
            np.asarray([seed_payloads[seed]["best_epoch"] for seed in seeds], dtype=np.float64)
        )

        if variant == "full":
            continue
        paired[variant] = {}
        table_row: Dict[str, object] = {"variant": variant}
        for metric in METRICS:
            variant_values = np.asarray(
                [seed_payloads[seed]["splits"]["test"][metric] for seed in seeds], dtype=np.float64
            )
            baseline_values = np.asarray(
                [runs["full"][seed]["splits"]["test"][metric] for seed in seeds], dtype=np.float64
            )
            comparison = paired_summary(variant_values, baseline_values, metric in LOWER_IS_BETTER)
            paired[variant][metric] = comparison
            table_row[f"{metric}_mean"] = float(variant_values.mean())
            table_row[f"{metric}_sd"] = float(variant_values.std(ddof=1))
            table_row[f"{metric}_delta"] = comparison["mean_difference"]
            table_row[f"{metric}_delta_ci_low"] = comparison["mean_difference_95ci"][0]
            table_row[f"{metric}_delta_ci_high"] = comparison["mean_difference_95ci"][1]
        table_row["parameter_count_mean"] = aggregate[variant]["parameter_count"]["mean"]
        table_row["best_epoch_mean"] = aggregate[variant]["best_epoch"]["mean"]
        table_rows.append(table_row)
        primary_p_values[variant] = paired[variant]["macro_auroc_ovr"]["paired_t"]["p"]

    holm = holm_adjust(primary_p_values)
    for row in table_rows:
        row["macro_auroc_ovr_paired_t_holm_p"] = holm[str(row["variant"])]
    for variant, value in holm.items():
        paired[variant]["macro_auroc_ovr"]["paired_t_holm_p"] = value

    summary = {
        "status": "complete",
        "run_root": str(run_root),
        "baseline_root": str(baseline_root),
        "seeds": seeds,
        "variants": variants,
        "aggregate": aggregate,
        "paired_test_comparison": paired,
        "statistical_guardrails": {
            "primary_metric": "test macro_auroc_ovr",
            "checkpoint_selection": "source-validation macro_auroc_ovr only",
            "paired_n": len(seeds),
            "multiplicity": "Holm correction across ablations for the primary metric only",
            "small_sample_warning": "Five paired seeds are low-powered; emphasize paired effects, intervals, and consistency.",
            "test_set_reuse": "All ablations use the same locked age-OOD test set; do not tune from these results.",
        },
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(Path(args.output_table), table_rows)
    write_csv(Path(args.output_per_seed), per_seed_rows)
    print(json.dumps({"status": "complete", "variants": variants, "seeds": seeds}, indent=2))


if __name__ == "__main__":
    main()

