#!/usr/bin/env bash
#SBATCH --partition=gpu-a100-h
#SBATCH --account=datalabgrp
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -euo pipefail

MODEL=${MODEL:?Set MODEL=spmnet or MODEL=transformer}
SEED=${SEED:?Set SEED}
ENCODER=${ENCODER:-matched}
OUTPUT_MODEL_NAME=${OUTPUT_MODEL_NAME:-$MODEL}
PROJECT_DIR=${PROJECT_DIR:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}
PYTHON_BIN=${PYTHON_BIN:-/group/datalabgrp/xgzhu/env/corerank_synth/bin/python}
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/outputs/icu_cross_mortality_v1}
CACHE_ROOT=${CACHE_ROOT:-$RUN_ROOT/cache}

cd "$PROJECT_DIR"
PYTHONPATH="$PROJECT_DIR/src" "$PYTHON_BIN" -m icu_cross.run_experiment \
  --model "$MODEL" \
  --encoder "$ENCODER" \
  --seed "$SEED" \
  --cache-root "$CACHE_ROOT" \
  --output-dir "$RUN_ROOT/$OUTPUT_MODEL_NAME/seed_$SEED" \
  --epochs "${EPOCHS:-30}" \
  --patience "${PATIENCE:-6}" \
  --batch-size "${BATCH_SIZE:-512}" \
  --workers "${WORKERS:-6}" \
  --hidden "${HIDDEN:-128}" \
  --metre-channels 256 256 256 256 \
  --amp
