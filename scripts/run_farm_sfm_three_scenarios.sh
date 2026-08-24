#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}
PYTHON_BIN=${PYTHON_BIN:-/group/datalabgrp/xgzhu/env/corerank_synth/bin/python}
BASE=${1:-outputs/sfm_three_scenarios_a100_v1}

if [[ "$BASE" != /* ]]; then
  BASE="$PROJECT_DIR/$BASE"
fi

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
    --methods concat,concat_paired,late_fusion,multimodal_transformer,sfm_self,sfm_self_oracle,sfm_net,sfm_oracle \
    --concept-shortcut-scale 0.0 \
    --domain-noise-scale 2.5 \
    --domain-tail-df 3.5 \
    --domain-style-angle-degrees 120 \
    --domain-action-scale 1.25 \
    --missing-certified-fraction 0.50 \
    --paired-intervention-weight 2.0 \
    --device cuda \
    --no-plots
done

"$PYTHON_BIN" -m bc_mcsgn.aggregate_runs --base-dir "$BASE"
