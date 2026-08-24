#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}
PYTHON_BIN=${PYTHON_BIN:-/group/datalabgrp/xgzhu/env/corerank_synth/bin/python}
BASE=${1:-outputs/iso_smoke_gpu}

cd "$PROJECT_DIR"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/group/datalabgrp/xgzhu/.cache}
export TORCH_HOME=${TORCH_HOME:-$XDG_CACHE_HOME/torch}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}
export MPLBACKEND=${MPLBACKEND:-Agg}
mkdir -p "$TORCH_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

"$PYTHON_BIN" -m iso_synth.run_experiment \
  --scenarios complementary,redundant,nuisance_only,shortcut,noisy_modality,mediated_context \
  --n-train-grid 128 \
  --seeds 0 \
  --n-val 128 \
  --n-test 256 \
  --ood-residual-shift 0.65 \
  --train-nuisance-corr 0.35 \
  --test-nuisance-corr -0.25 \
  --ood-noise-multiplier 1.35 \
  --epochs 3 \
  --batch-size 128 \
  --hidden-dim 32 \
  --output-dir "$BASE"
