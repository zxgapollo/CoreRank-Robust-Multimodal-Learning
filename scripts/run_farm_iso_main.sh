#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}
PYTHON_BIN=${PYTHON_BIN:-/group/datalabgrp/xgzhu/env/corerank_synth/bin/python}
BASE=${1:-outputs/iso_main_grid_gpu}
EPOCHS=${EPOCHS:-30}
N_TRAIN_GRID=${N_TRAIN_GRID:-128,512,2048}
SEEDS=${SEEDS:-0,1,2}
N_VAL=${N_VAL:-512}
N_TEST=${N_TEST:-1024}
BATCH_SIZE=${BATCH_SIZE:-256}
HIDDEN_DIM=${HIDDEN_DIM:-64}
SCENARIOS=${SCENARIOS:-complementary,redundant,nuisance_only,shortcut,noisy_modality,mediated_context}
OOD_RESIDUAL_SHIFT=${OOD_RESIDUAL_SHIFT:-0.65}
TRAIN_NUISANCE_CORR=${TRAIN_NUISANCE_CORR:-0.35}
TEST_NUISANCE_CORR=${TEST_NUISANCE_CORR:--0.25}
OOD_NOISE_MULTIPLIER=${OOD_NOISE_MULTIPLIER:-1.35}

cd "$PROJECT_DIR"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/group/datalabgrp/xgzhu/.cache}
export TORCH_HOME=${TORCH_HOME:-$XDG_CACHE_HOME/torch}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$XDG_CACHE_HOME/matplotlib}
export MPLBACKEND=${MPLBACKEND:-Agg}
mkdir -p "$TORCH_HOME" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

"$PYTHON_BIN" -m iso_synth.run_experiment \
  --scenarios "$SCENARIOS" \
  --n-train-grid "$N_TRAIN_GRID" \
  --seeds "$SEEDS" \
  --n-val "$N_VAL" \
  --n-test "$N_TEST" \
  --ood-residual-shift "$OOD_RESIDUAL_SHIFT" \
  --train-nuisance-corr "$TRAIN_NUISANCE_CORR" \
  --test-nuisance-corr "$TEST_NUISANCE_CORR" \
  --ood-noise-multiplier "$OOD_NOISE_MULTIPLIER" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --hidden-dim "$HIDDEN_DIM" \
  --output-dir "$BASE"
