from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate training on the METRE cache alignment audit.")
    parser.add_argument("--cache-root", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.cache_root) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = manifest["cross_dataset_alignment"]
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    if not audit["passes_expected_empty_check"]:
        raise SystemExit(
            "Cross-dataset alignment audit failed: "
            f"unexpected_eicu_absent={audit['unexpected_eicu_absent']}, "
            f"official_empty_but_present={audit['official_empty_but_present']}"
        )


if __name__ == "__main__":
    main()
