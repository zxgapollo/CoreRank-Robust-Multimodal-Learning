#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}
PYTHON_BIN=${PYTHON_BIN:-/group/datalabgrp/xgzhu/env/corerank_synth/bin/python}
BASE=${1:-outputs/smoke_gpu}

cd "$PROJECT_DIR"
export PYTHONNOUSERSITE=1

"$PYTHON_BIN" -m corerank_synth.run_experiment \
  --scenario complementary \
  --n-train 512 --n-val 256 --n-test 256 \
  --epochs 3 --batch-size 128 \
  --gate-anneal-epochs 3 --gate-temperature-min 0.3 \
  --sparse-warmup-epochs 1 --gate-binary-weight 0.01 \
  --output-dir "$BASE/complementary"

"$PYTHON_BIN" -m corerank_synth.run_experiment \
  --scenario biased \
  --biased-modality 0 \
  --bias-strength 2.0 \
  --train-bias-corr 0.85 \
  --test-bias-corr -0.50 \
  --n-train 512 --n-val 256 --n-test 256 \
  --epochs 3 --batch-size 128 \
  --gate-anneal-epochs 3 --gate-temperature-min 0.3 \
  --sparse-warmup-epochs 1 --gate-binary-weight 0.01 \
  --output-dir "$BASE/biased"

"$PYTHON_BIN" -m corerank_synth.run_experiment \
  --scenario domain \
  --domain-shifted-modality 0 \
  --domain-shift-strength 1.5 \
  --n-train 512 --n-val 256 --n-test 256 \
  --epochs 3 --batch-size 128 \
  --gate-anneal-epochs 3 --gate-temperature-min 0.3 \
  --sparse-warmup-epochs 1 --gate-binary-weight 0.01 \
  --output-dir "$BASE/domain"
