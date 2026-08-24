from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate SFM-Net multi-seed runs.")
    parser.add_argument("--base-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    frames = []
    for path in sorted(base_dir.glob("seed_*/metrics.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "seed", int(path.parent.name.split("_")[-1]))
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No seed_*/metrics.csv files found under {base_dir}")

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(base_dir / "metrics_all.csv", index=False)
    numeric = [column for column in combined.select_dtypes(include="number").columns if column != "seed"]
    summary = combined.groupby(["method", "split"])[numeric].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(base_dir / "summary_by_method_split.csv", index=False)

    primary_columns = [
        "method",
        "split",
        "auc_mean",
        "auc_std",
        "brier_mean",
        "brier_std",
        "relevant_factor_mcc_mean",
        "incidence_f1_perm_mean",
        "task_f1_perm_mean",
        "certificate_rate_mean",
        "certified_auc_mean",
        "true_certificate_rate_mean",
        "true_certified_auc_mean",
        "true_uncertified_auc_mean",
        "paired_state_mse_mean",
    ]
    primary_columns = [column for column in primary_columns if column in summary.columns]
    primary_splits = [
        "test_id",
        "test_concept_shift",
        "test_domain_shift",
        "test_missing_information",
    ]
    primary = summary[summary["split"].isin(primary_splits)][primary_columns]
    payload = {
        "n_seeds": int(combined["seed"].nunique()),
        "seed_values": sorted(int(seed) for seed in combined["seed"].unique()),
        "metrics_all": str(base_dir / "metrics_all.csv"),
        "summary_csv": str(base_dir / "summary_by_method_split.csv"),
        "primary_results": primary.to_dict(orient="records"),
    }
    with open(base_dir / "aggregate_summary.json", "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
