from __future__ import annotations

import argparse
import json
from pathlib import Path

from .features import FeaturePaths, build_dataset_cache, validate_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build aligned five-modality MIMIC-IV/eICU caches.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--muse-src", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--datasets", nargs="+", choices=("mimic4", "eicu"), default=("mimic4", "eicu"))
    parser.add_argument("--mimic-splits", nargs="+", default=("train", "val", "test"))
    parser.add_argument("--eicu-splits", nargs="+", default=("test",))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = FeaturePaths(Path(args.data_root), Path(args.muse_src), Path(args.output_root))
    paths.output_root.mkdir(parents=True, exist_ok=True)
    result = {}
    for dataset in args.datasets:
        splits = args.mimic_splits if dataset == "mimic4" else args.eicu_splits
        audit_path = paths.output_root / dataset / "audit.json"
        if audit_path.exists():
            print(f"Validating existing {dataset} cache", flush=True)
            result[dataset] = validate_cache(paths.output_root, dataset, splits)
        else:
            result[dataset] = build_dataset_cache(dataset, splits, paths)
            validate_cache(paths.output_root, dataset, splits)
    summary = paths.output_root / "cache_summary.json"
    summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
