#!/usr/bin/env bash
#SBATCH --partition=gpu-a100-h
#SBATCH --account=datalabgrp
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/outputs/icu_cross_metre_hourly_multimodal_v1}
CACHE_ROOT=${CACHE_ROOT:-$RUN_ROOT/cache}
PYTHON_BIN=${PYTHON_BIN:-/group/datalabgrp/xgzhu/env/corerank_synth/bin/python}
SEED=${SEED:-2026}

cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR/src" "$PYTHON_BIN" -m icu_cross.run_metre_tcn \
  --cache-root "$CACHE_ROOT" \
  --output-dir "$RUN_ROOT/metre_tcn/seed_$SEED" \
  --seed "$SEED" \
  --epochs "${EPOCHS:-150}" \
  --patience "${PATIENCE:-5}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --workers "${WORKERS:-6}" \
  --lr "${LR:-0.001}" \
  --kernel-size 3 \
  --dropout 0.2 \
  --channels 256 256 256 256
