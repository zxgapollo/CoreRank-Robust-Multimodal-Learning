#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}
PYTHON_BIN=${PYTHON_BIN:-/group/datalabgrp/xgzhu/env/corerank_synth/bin/python}
BASE=${1:-outputs/main_grid_gpu}
EPOCHS=${EPOCHS:-60}
N_TRAIN=${N_TRAIN:-5000}
N_VAL=${N_VAL:-1000}
N_TEST=${N_TEST:-2000}
BATCH_SIZE=${BATCH_SIZE:-256}

cd "$PROJECT_DIR"
export PYTHONNOUSERSITE=1

for SCENARIO in complementary redundant biased domain; do
  for SEED in 0 1 2 3 4; do
    OUT="$BASE/${SCENARIO}_seed${SEED}"
    "$PYTHON_BIN" -m corerank_synth.run_experiment \
      --scenario "$SCENARIO" \
      --seed "$SEED" \
      --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" \
      --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
      --z-dim 6 --u-dim 3 --x-dim 16 \
      --recon-reduction mean --label-weight 2.0 \
      --structural-weight 0.2 --dag-weight 0.1 --graph-l1-weight 0.001 \
      --structural-warmup-epochs 2 \
      --bias-invariance-weight 0.2 --domain-invariance-weight 0.2 \
      --rank-kappa 0.5 --sparse-budget 9.0 \
      --gate-anneal-epochs "$EPOCHS" --gate-temperature-min 0.2 \
      --gate-binary-weight 0.01 \
      --output-dir "$OUT"
  done
done
