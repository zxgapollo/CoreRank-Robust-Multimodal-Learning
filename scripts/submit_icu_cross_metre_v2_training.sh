#!/usr/bin/env bash

set -euo pipefail

CACHE_JOB_ID=${CACHE_JOB_ID:?Set CACHE_JOB_ID to the active cache-build job}
PROJECT_DIR=${PROJECT_DIR:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/outputs/icu_cross_metre_hourly_multimodal_v2}
PYTHON_BIN=${PYTHON_BIN:-/group/datalabgrp/xgzhu/env/corerank_synth/bin/python}
SBATCH_BIN=${SBATCH_BIN:-/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/slurm/bin/sbatch}
SEED=${SEED:-2026}

mkdir -p "$RUN_ROOT/logs"

audit_job=$(
  "$SBATCH_BIN" --parsable \
    --job-name=metre_v2_audit \
    --dependency="afterok:$CACHE_JOB_ID" \
    --partition=high \
    --account=datalabgrp \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:15:00 \
    --output="$RUN_ROOT/logs/alignment_%j.out" \
    --error="$RUN_ROOT/logs/alignment_%j.err" \
    --wrap="cd '$PROJECT_DIR' && PYTHONPATH='$PROJECT_DIR/src' '$PYTHON_BIN' -m icu_cross.check_metre_alignment --cache-root '$RUN_ROOT/cache'"
)

spmnet_job=$(
  "$SBATCH_BIN" --parsable \
    --job-name=spmnet_metre_v2 \
    --dependency="afterok:$audit_job" \
    --account=datalabgrp \
    --output="$RUN_ROOT/logs/spmnet_2026_%j.out" \
    --error="$RUN_ROOT/logs/spmnet_2026_%j.err" \
    --export="ALL,PROJECT_DIR=$PROJECT_DIR,RUN_ROOT=$RUN_ROOT,MODEL=spmnet,SEED=$SEED" \
    "$PROJECT_DIR/scripts/train_icu_cross_a100.sh"
)

transformer_job=$(
  "$SBATCH_BIN" --parsable \
    --job-name=transformer_metre_v2 \
    --dependency="afterok:$audit_job" \
    --account=ctbrowngrp \
    --output="$RUN_ROOT/logs/transformer_2026_%j.out" \
    --error="$RUN_ROOT/logs/transformer_2026_%j.err" \
    --export="ALL,PROJECT_DIR=$PROJECT_DIR,RUN_ROOT=$RUN_ROOT,MODEL=transformer,SEED=$SEED" \
    "$PROJECT_DIR/scripts/train_icu_cross_a100.sh"
)

printf 'audit_job=%s\nspmnet_job=%s\ntransformer_job=%s\n' \
  "$audit_job" "$spmnet_job" "$transformer_job"
