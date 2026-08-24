#!/usr/bin/env bash
#SBATCH --partition=high
#SBATCH --account=datalabgrp
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}
CAUSAL_ROOT=${CAUSAL_ROOT:-/group/datalabgrp/xgzhu/CausalMedical}
PYTHON_BIN=${PYTHON_BIN:-/group/datalabgrp/xgzhu/env/corerank_synth/bin/python}
CACHE_ROOT=${CACHE_ROOT:-$PROJECT_DIR/outputs/icu_cross_mortality_v1/cache}

cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR/src" "$PYTHON_BIN" -m icu_cross.build_cache \
  --data-root "$CAUSAL_ROOT/Dataset/MUSE/processed_data" \
  --muse-src "$CAUSAL_ROOT/Code/MUSE/src" \
  --output-root "$CACHE_ROOT" \
  --mimic-splits train val test \
  --eicu-splits test
