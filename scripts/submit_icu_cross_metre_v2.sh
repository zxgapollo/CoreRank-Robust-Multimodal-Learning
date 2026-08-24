#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/outputs/icu_cross_metre_hourly_multimodal_v2}
PYTHON_BIN=${PYTHON_BIN:-/group/datalabgrp/xgzhu/env/corerank_synth/bin/python}
SBATCH_BIN=${SBATCH_BIN:-/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/slurm/bin/sbatch}

mkdir -p "$RUN_ROOT/logs"

test_job=$(
  "$SBATCH_BIN" --parsable \
    --job-name=metre_v2_tests \
    --partition=high \
    --account=datalabgrp \
    --cpus-per-task=4 \
    --mem=16G \
    --time=00:30:00 \
    --output="$RUN_ROOT/logs/tests_%j.out" \
    --error="$RUN_ROOT/logs/tests_%j.err" \
    --wrap="cd '$PROJECT_DIR' && PYTHONPATH='$PROJECT_DIR/src' '$PYTHON_BIN' -m pytest tests/test_icu_cross.py -q"
)

cache_job=$(
  "$SBATCH_BIN" --parsable \
    --job-name=metre_cache_v2 \
    --dependency="afterok:$test_job" \
    --output="$RUN_ROOT/logs/cache_%j.out" \
    --error="$RUN_ROOT/logs/cache_%j.err" \
    --export="ALL,PROJECT_DIR=$PROJECT_DIR,RUN_ROOT=$RUN_ROOT" \
    "$PROJECT_DIR/scripts/build_icu_cross_metre_hourly_cache.sh"
)

printf 'test_job=%s\ncache_job=%s\n' "$test_job" "$cache_job"
