#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=${PYTHONPATH:-}:$(pwd)/src

BASE=${1:-outputs/main_grid}
mkdir -p "$BASE"

for SCENARIO in complementary redundant biased domain; do
  for SEED in 0 1 2 3 4; do
    OUT="$BASE/${SCENARIO}_seed${SEED}"
    python -m corerank_synth.run_experiment \
      --scenario "$SCENARIO" \
      --seed "$SEED" \
      --n-train 5000 --n-val 1000 --n-test 2000 \
      --epochs 60 --batch-size 256 \
      --z-dim 6 --u-dim 3 --x-dim 16 \
      --rank-kappa 0.5 --sparse-budget 9.0 \
      --gate-anneal-epochs 60 --gate-temperature-min 0.2 \
      --gate-binary-weight 0.01 \
      --output-dir "$OUT"
  done
done
