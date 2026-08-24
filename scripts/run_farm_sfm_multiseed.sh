#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}
PYTHON_BIN=${PYTHON_BIN:-/group/datalabgrp/xgzhu/env/corerank_synth/bin/python}
BASE=${1:-outputs/sfm_selective_factors_a100_v1}

if [[ "$BASE" != /* ]]; then
  BASE="$PROJECT_DIR/$BASE"
fi

# The remote project contains a legacy top-level bc_mcsgn/ copy. Running from
# src/ makes module resolution unambiguous without deleting that old package.
cd "$PROJECT_DIR/src"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_DIR/src"

for SEED in 0 1 2 3 4; do
  "$PYTHON_BIN" -m bc_mcsgn.run_experiment \
    --output-dir "$BASE/seed_$SEED" \
    --force-generate \
    --seed "$SEED" \
    --n-train 5000 \
    --n-val 1000 \
    --n-test 2000 \
    --x-dim 10 \
    --baseline-epochs 50 \
    --correction-epochs 80 \
    --batch-size 256 \
    --hidden-dim 64 \
    --methods concat,concat_paired,sfm_self,sfm_net,sfm_oracle \
    --paired-intervention-weight 2.0 \
    --device cuda \
    --no-plots
done

"$PYTHON_BIN" -m bc_mcsgn.aggregate_runs --base-dir "$BASE"
