#!/usr/bin/env python3
"""Build a class-controlled age-shift split for the real ADNI cohort."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split


LABEL_MAP = {"CN": 0, "MCI": 1, "AD": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-csv", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    return parser.parse_args()


def finite(values: Iterable[object]) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=np.float64)


def numeric_comparison(source: Iterable[object], target: Iterable[object]) -> Dict[str, object]:
    a = finite(source)
    b = finite(target)
    result: Dict[str, object] = {
        "train_n": int(a.size),
        "test_n": int(b.size),
        "train_mean": float(a.mean()) if a.size else None,
        "test_mean": float(b.mean()) if b.size else None,
        "train_sd": float(a.std(ddof=1)) if a.size > 1 else None,
        "test_sd": float(b.std(ddof=1)) if b.size > 1 else None,
        "train_median": float(np.median(a)) if a.size else None,
        "test_median": float(np.median(b)) if b.size else None,
        "train_q25_q75": [float(value) for value in np.quantile(a, [0.25, 0.75])] if a.size else None,
        "test_q25_q75": [float(value) for value in np.quantile(b, [0.25, 0.75])] if b.size else None,
    }
    if a.size < 2 or b.size < 2:
        result.update({"welch_t": None, "welch_p": None, "cohens_d": None, "ks_statistic": None, "ks_p": None})
        return result
    t_result = stats.ttest_ind(a, b, equal_var=False)
    ks_result = stats.ks_2samp(a, b, alternative="two-sided", method="auto")
    pooled_variance = ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2)
    pooled_sd = math.sqrt(max(0.0, pooled_variance))
    result.update(
        {
            "mean_difference_test_minus_train": float(b.mean() - a.mean()),
            "welch_t": float(t_result.statistic),
            "welch_p": float(t_result.pvalue),
            "cohens_d_test_minus_train": float((b.mean() - a.mean()) / pooled_sd) if pooled_sd > 0 else None,
            "ks_statistic": float(ks_result.statistic),
            "ks_p": float(ks_result.pvalue),
        }
    )
    return result


def categorical_comparison(train: pd.Series, test: pd.Series) -> Dict[str, object]:
    train_values = train.fillna("<missing>").astype(str)
    test_values = test.fillna("<missing>").astype(str)
    categories = sorted(set(train_values) | set(test_values))
    table = np.asarray(
        [
            [(train_values == category).sum() for category in categories],
            [(test_values == category).sum() for category in categories],
        ],
        dtype=np.int64,
    )
    chi2, p_value, _, expected = stats.chi2_contingency(table)
    n = int(table.sum())
    denominator = max(1, min(table.shape) - 1)
    cramers_v = math.sqrt(float(chi2) / (n * denominator)) if n else None
    result = {
        "categories": categories,
        "train_counts": table[0].tolist(),
        "test_counts": table[1].tolist(),
        "chi2": float(chi2),
        "chi2_p": float(p_value),
        "cramers_v": cramers_v,
        "minimum_expected_count": float(expected.min()),
    }
    nonmissing_categories = [category for category in categories if category != "<missing>"]
    if len(nonmissing_categories) == 2:
        binary_table = np.asarray(
            [
                [(train_values == category).sum() for category in nonmissing_categories],
                [(test_values == category).sum() for category in nonmissing_categories],
            ],
            dtype=np.int64,
        )
        odds_ratio, fisher_p = stats.fisher_exact(binary_table)
        result["fisher_exact_nonmissing_binary"] = {
            "categories": nonmissing_categories,
            "table": binary_table.tolist(),
            "odds_ratio": float(odds_ratio),
            "p": float(fisher_p),
        }
    return result


def main() -> None:
    args = parse_args()
    if not (0.0 < args.test_fraction < 1.0):
        raise ValueError("test-fraction must be between 0 and 1")
    if not (0.0 < args.val_fraction < 1.0 - args.test_fraction):
        raise ValueError("val-fraction must be positive and leave room for training")

    frame = pd.read_csv(args.master_csv, dtype=str, keep_default_na=True)
    initial_rows = len(frame)
    frame = frame[frame["diagnosis_3class"].isin(LABEL_MAP)].copy()
    if "has_core_multimodal" in frame:
        frame = frame[pd.to_numeric(frame["has_core_multimodal"], errors="coerce").fillna(0).eq(1)].copy()
    frame["mri_age_numeric"] = pd.to_numeric(frame["mri_age"], errors="coerce")
    missing_age = int(frame["mri_age_numeric"].isna().sum())
    frame = frame[frame["mri_age_numeric"].notna()].copy()
    frame["label"] = frame["diagnosis_3class"].map(LABEL_MAP).astype(int)
    frame = frame.sort_values("subject_id").reset_index(drop=True)
    if not frame["subject_id"].is_unique:
        raise ValueError("The age-OOD cohort must contain one baseline image per subject")

    rng = np.random.default_rng(args.seed)
    test_indices = []
    for diagnosis in LABEL_MAP:
        block = frame[frame["diagnosis_3class"].eq(diagnosis)].copy()
        block["tie_break"] = rng.random(len(block))
        block = block.sort_values(["mri_age_numeric", "tie_break"], ascending=[False, True])
        test_count = max(1, int(round(len(block) * args.test_fraction)))
        test_indices.extend(block.index[:test_count].tolist())

    test_indices = np.asarray(sorted(test_indices), dtype=np.int64)
    source_indices = np.setdiff1d(np.arange(len(frame)), test_indices)
    relative_val_fraction = args.val_fraction / (1.0 - args.test_fraction)
    train_indices, val_indices = train_test_split(
        source_indices,
        test_size=relative_val_fraction,
        random_state=args.seed,
        stratify=frame.loc[source_indices, "label"],
    )
    frame["split"] = ""
    frame.loc[train_indices, "split"] = "train"
    frame.loc[val_indices, "split"] = "val"
    frame.loc[test_indices, "split"] = "test"

    cache_root = Path(args.cache_root)
    frame["image_cache"] = frame.apply(
        lambda row: str(cache_root / row["subject_id"] / f"{row['subject_id']}_I{row['image_id']}_T1w_brain_96.npy"),
        axis=1,
    )
    missing_cache = [path for path in frame["image_cache"] if not Path(path).is_file()]
    if missing_cache:
        raise FileNotFoundError(f"{len(missing_cache)} MRI caches are missing; first={missing_cache[0]}")

    train = frame[frame["split"].eq("train")]
    validation = frame[frame["split"].eq("val")]
    test = frame[frame["split"].eq("test")]
    age_by_class = {
        diagnosis: numeric_comparison(
            train.loc[train["diagnosis_3class"].eq(diagnosis), "mri_age_numeric"],
            test.loc[test["diagnosis_3class"].eq(diagnosis), "mri_age_numeric"],
        )
        for diagnosis in LABEL_MAP
    }
    report = {
        "protocol": "class-conditional age shift: oldest 15% within each diagnosis reserved for OOD test; source pool randomly split into train/validation",
        "seed": args.seed,
        "initial_rows": int(initial_rows),
        "eligible_subjects": int(len(frame)),
        "excluded_missing_age": missing_age,
        "subject_unique": bool(frame["subject_id"].is_unique),
        "cache_complete": not missing_cache,
        "split_sizes": frame["split"].value_counts().to_dict(),
        "class_counts": {
            split: frame.loc[frame["split"].eq(split), "diagnosis_3class"].value_counts().sort_index().to_dict()
            for split in ("train", "val", "test")
        },
        "primary_age_shift": numeric_comparison(train["mri_age_numeric"], test["mri_age_numeric"]),
        "age_shift_by_diagnosis": age_by_class,
        "secondary_numeric_diagnostics": {
            column: numeric_comparison(train[column], test[column])
            for column in ("demographics_pteducat", "mri_field_strength_t")
        },
        "categorical_diagnostics": {
            "diagnosis_3class": categorical_comparison(train["diagnosis_3class"], test["diagnosis_3class"]),
            "demographics_ptgender": categorical_comparison(train["demographics_ptgender"], test["demographics_ptgender"]),
        },
        "source_validation_diagnostics": {
            "age": numeric_comparison(train["mri_age_numeric"], validation["mri_age_numeric"]),
            "diagnosis_3class": categorical_comparison(train["diagnosis_3class"], validation["diagnosis_3class"]),
            "demographics_ptgender": categorical_comparison(
                train["demographics_ptgender"], validation["demographics_ptgender"]
            ),
        },
        "interpretation_guardrails": [
            "Age is the prespecified shift variable; secondary tests are diagnostic rather than confirmatory.",
            "Class-conditional selection keeps class proportions comparable but uses diagnosis labels to construct the benchmark.",
            "A significant t-test alone is insufficient; Cohen's d and the KS statistic quantify shift magnitude and shape.",
            "This is a stress-test distribution shift, not an estimate of prospective clinical prevalence.",
        ],
    }

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["subject_id", "image_id", "diagnosis_3class", "label", "split", "image_cache"]
    frame[columns].to_csv(output_path, index=False)
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
