#!/usr/bin/env python3
"""Aggregate paired multi-seed ADNI age-OOD experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats


METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "macro_auroc_ovr",
    "macro_auprc_ovr",
    "ece_10bin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def mean_sd(values: np.ndarray) -> Dict[str, float]:
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0}


def paired_summary(spmnet: np.ndarray, transformer: np.ndarray, lower_is_better: bool) -> Dict[str, object]:
    difference = spmnet - transformer
    n = len(difference)
    sem = float(stats.sem(difference)) if n > 1 else 0.0
    margin = float(stats.t.ppf(0.975, n - 1) * sem) if n > 1 else 0.0
    paired_t = stats.ttest_rel(spmnet, transformer)
    try:
        wilcoxon = stats.wilcoxon(difference, zero_method="wilcox", alternative="two-sided", method="auto")
        wilcoxon_result = {"statistic": float(wilcoxon.statistic), "p": float(wilcoxon.pvalue)}
    except ValueError:
        wilcoxon_result = {"statistic": None, "p": None}
    wins = difference < 0 if lower_is_better else difference > 0
    losses = difference > 0 if lower_is_better else difference < 0
    return {
        "difference_definition": "SPMNet minus Transformer",
        "paired_differences": difference.tolist(),
        "mean_difference": float(difference.mean()),
        "sd_difference": float(difference.std(ddof=1)) if n > 1 else 0.0,
        "mean_difference_95ci": [float(difference.mean() - margin), float(difference.mean() + margin)],
        "paired_t": {"statistic": float(paired_t.statistic), "p": float(paired_t.pvalue)},
        "wilcoxon": wilcoxon_result,
        "spmnet_wins": int(wins.sum()),
        "ties": int((difference == 0).sum()),
        "transformer_wins": int(losses.sum()),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.run_root)
    records: List[Dict[str, object]] = []
    nested: Dict[str, Dict[int, Dict[str, object]]] = {}
    for model in ("spmnet", "transformer"):
        nested[model] = {}
        for path in sorted((root / model).glob("seed_*/metrics.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            seed = int(payload["seed"])
            nested[model][seed] = payload
            for split in ("val", "test"):
                record: Dict[str, object] = {
                    "model": model,
                    "seed": seed,
                    "split": split,
                    "best_epoch": int(payload["best_epoch"]),
                }
                record.update({name: payload["splits"][split][name] for name in METRICS})
                records.append(record)

    seed_sets = {model: sorted(values) for model, values in nested.items()}
    if seed_sets["spmnet"] != seed_sets["transformer"] or len(seed_sets["spmnet"]) < 2:
        raise ValueError(f"Expected matching multi-seed results, found {seed_sets}")
    seeds = seed_sets["spmnet"]

    aggregate: Dict[str, object] = {}
    for model in ("spmnet", "transformer"):
        aggregate[model] = {}
        for split in ("val", "test"):
            aggregate[model][split] = {
                metric: mean_sd(np.asarray([nested[model][seed]["splits"][split][metric] for seed in seeds]))
                for metric in METRICS
            }
        aggregate[model]["test_minus_val"] = {
            metric: mean_sd(
                np.asarray(
                    [
                        nested[model][seed]["splits"]["test"][metric]
                        - nested[model][seed]["splits"]["val"][metric]
                        for seed in seeds
                    ]
                )
            )
            for metric in METRICS
        }

    paired = {}
    for metric in METRICS:
        spmnet_values = np.asarray([nested["spmnet"][seed]["splits"]["test"][metric] for seed in seeds])
        transformer_values = np.asarray([nested["transformer"][seed]["splits"]["test"][metric] for seed in seeds])
        paired[metric] = paired_summary(spmnet_values, transformer_values, lower_is_better=metric == "ece_10bin")

    modalities = {
        model: sorted({tuple(nested[model][seed].get("modalities", [])) for seed in seeds})
        for model in ("spmnet", "transformer")
    }
    summary = {
        "status": "complete",
        "run_root": str(root),
        "seeds": seeds,
        "modalities_by_model": {model: [list(item) for item in values] for model, values in modalities.items()},
        "aggregate": aggregate,
        "paired_test_comparison": paired,
        "statistical_guardrails": {
            "paired_n": len(seeds),
            "small_sample_warning": "With five paired seeds, inference is low-powered; emphasize effect sizes, confidence intervals, and consistency.",
            "selection": "Checkpoints selected only by source-domain validation macro-AUROC; age-OOD test was not used for selection.",
        },
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
