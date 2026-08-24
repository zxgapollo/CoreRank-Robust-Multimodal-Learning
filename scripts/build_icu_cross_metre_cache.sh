#!/usr/bin/env bash
#SBATCH --partition=high
#SBATCH --account=datalabgrp
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=18:00:00

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}
CAUSAL_ROOT=${CAUSAL_ROOT:-/group/datalabgrp/xgzhu/CausalMedical}
PYTHON_BIN=${PYTHON_BIN:-/group/datalabgrp/xgzhu/env/corerank_synth/bin/python}
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/outputs/icu_cross_metre_v1}
CACHE_ROOT=${CACHE_ROOT:-$RUN_ROOT/cache}

cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR/src" "$PYTHON_BIN" -m icu_cross.build_metre_cache \
  --mimic-root "$CAUSAL_ROOT/Dataset/raw_data/MIMIC-IV2/physionet.org/files/mimiciv/2.0" \
  --eicu-root "$CAUSAL_ROOT/Dataset/raw_data/eICU/physionet.org/files/eicu-crd/2.0" \
  --output-root "$CACHE_ROOT" \
  --observation-hours 48 \
  --gap-hours 6 \
  --max-los-hours 240 \
  --seed 2026 \
  --chunksize "${CHUNKSIZE:-1000000}"
