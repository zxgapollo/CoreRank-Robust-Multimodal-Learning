#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-outputs/bc_mcsgn_corrected_a100_anchor_pilot}"

"${PYTHON:-python}" -m bc_mcsgn.run_experiment \
  --output-dir "${OUT_DIR}" \
  --force-generate \
  --seed 0 \
  --n-train 2000 \
  --n-val 500 \
  --n-test 1000 \
  --x-dim 10 \
  --warmup-epochs 25 \
  --baseline-epochs 25 \
  --correction-epochs 50 \
  --batch-size 256 \
  --hidden-dim 64 \
  --methods concat,warmup,bc_mcsgn \
  --proto-source true_biased \
  --fixed-graph \
  --fixed-masks \
  --state-anchor-weight 2.0 \
  --delta-anchor-weight 1.0 \
  --device cuda \
  --verbose
