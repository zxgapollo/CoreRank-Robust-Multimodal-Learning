#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}
PYTHON_BIN=${PYTHON_BIN:-/group/datalabgrp/xgzhu/env/corerank_synth/bin/python}
BASE=${1:-outputs/smoke_gpu}

cd "$PROJECT_DIR"
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/group/datalabgrp/xgzhu/.cache}
export TORCH_HOME=${TORCH_HOME:-$XDG_CACHE_HOME/torch}
mkdir -p "$TORCH_HOME" "$XDG_CACHE_HOME"

"$PYTHON_BIN" -m corerank_synth.run_experiment \
  --scenario complementary \
  --n-train 512 --n-val 256 --n-test 256 \
  --epochs 3 --batch-size 128 \
  --n-modalities 4 \
  --recon-reduction mean --label-weight 2.0 --beta-z 0.01 \
  --structural-weight 0.0 --dag-weight 0.1 --graph-l1-weight 0.001 \
  --best-id-tolerance 0.02 \
  --eval-fisher-batches 0 --eval-true-fisher-samples 1 \
  --gate-anneal-epochs 3 --gate-temperature-min 0.3 \
  --sparse-warmup-epochs 1 --gate-binary-weight 0.01 \
  --output-dir "$BASE/complementary"

"$PYTHON_BIN" -m corerank_synth.run_experiment \
  --scenario shortcut \
  --biased-modality 3 \
  --bias-strength 2.5 \
  --train-bias-corr 0.85 \
  --test-bias-corr -0.50 \
  --n-train 512 --n-val 256 --n-test 256 \
  --epochs 3 --batch-size 128 \
  --n-modalities 4 \
  --recon-reduction mean --label-weight 2.0 --beta-z 0.01 \
  --structural-weight 0.0 --dag-weight 0.1 --graph-l1-weight 0.001 \
  --best-id-tolerance 0.02 \
  --eval-fisher-batches 0 --eval-true-fisher-samples 1 \
  --gate-anneal-epochs 3 --gate-temperature-min 0.3 \
  --sparse-warmup-epochs 1 --gate-binary-weight 0.01 \
  --output-dir "$BASE/shortcut"

"$PYTHON_BIN" -m corerank_synth.run_experiment \
  --scenario measurement \
  --domain-shifted-modality 0 \
  --domain-shift-strength 2.0 \
  --n-train 512 --n-val 256 --n-test 256 \
  --epochs 3 --batch-size 128 \
  --n-modalities 4 \
  --recon-reduction mean --label-weight 2.0 --beta-z 0.01 \
  --structural-weight 0.0 --dag-weight 0.1 --graph-l1-weight 0.001 \
  --best-id-tolerance 0.02 \
  --eval-fisher-batches 0 --eval-true-fisher-samples 1 \
  --gate-anneal-epochs 3 --gate-temperature-min 0.3 \
  --sparse-warmup-epochs 1 --gate-binary-weight 0.01 \
  --output-dir "$BASE/measurement"
