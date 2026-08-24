#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


METRICS = (
    "auroc", "auprc", "auprc_lift", "balanced_accuracy", "f1", "sensitivity", "specificity",
    "brier", "ece_10bin",
)
DOMAINS = ("mimic_test", "eicu_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    critical = float(stats.t.ppf(0.975, df=len(array) - 1)) if len(array) > 1 else 0.0
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "seeds": int(len(array)),
        "ci95_half_width": float(critical * array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.run_root)
    records = {}
    rows = []
    for model in ("spmnet", "transformer"):
        for path in sorted((root / model).glob("seed_*/metrics.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            seed = int(data["seed"])
            records[(model, seed)] = data
            for domain in DOMAINS:
                rows.append({"model": model, "seed": seed, "domain": domain, **{metric: data[domain][metric] for metric in METRICS}})
    if not records:
        raise FileNotFoundError(f"No metrics found under {root}")
    aggregate = {"models": {}, "paired_spmnet_minus_transformer": {}}
    for model in ("spmnet", "transformer"):
        aggregate["models"][model] = {}
        for domain in DOMAINS:
            aggregate["models"][model][domain] = {
                metric: summary([row[metric] for row in rows if row["model"] == model and row["domain"] == domain])
                for metric in METRICS
            }
    common_seeds = sorted(set(seed for model, seed in records if model == "spmnet") & set(seed for model, seed in records if model == "transformer"))
    aggregate["paired_seeds"] = common_seeds
    for domain in DOMAINS:
        aggregate["paired_spmnet_minus_transformer"][domain] = {}
        for metric in METRICS:
            deltas = [records[("spmnet", seed)][domain][metric] - records[("transformer", seed)][domain][metric] for seed in common_seeds]
            metric_summary = summary(deltas)
            if len(deltas) > 1:
                metric_summary["paired_ttest_pvalue_two_sided"] = float(stats.ttest_1samp(deltas, popmean=0.0).pvalue)
                nonzero = [value for value in deltas if value != 0]
                metric_summary["exact_sign_test_pvalue_two_sided"] = float(
                    stats.binomtest(sum(value > 0 for value in nonzero), len(nonzero), 0.5).pvalue
                ) if nonzero else 1.0
            aggregate["paired_spmnet_minus_transformer"][domain][metric] = metric_summary
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output.with_suffix(".csv")).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
