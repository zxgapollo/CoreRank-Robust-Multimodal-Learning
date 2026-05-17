#!/usr/bin/env bash
set -euo pipefail

BASE=${1:-outputs/smoke}

python -m corerank_synth.run_experiment \
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

python -m corerank_synth.run_experiment \
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

python -m corerank_synth.run_experiment \
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
