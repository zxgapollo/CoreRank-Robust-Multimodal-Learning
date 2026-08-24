#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src "${PYTHON:-python3}" -m bc_mcsgn.run_experiment \
  --output-dir outputs/bc_mcsgn_smoke \
  --force-generate \
  --seed 0 \
  --n-train 512 \
  --n-val 128 \
  --n-test 256 \
  --x-dim 8 \
  --warmup-epochs 3 \
  --baseline-epochs 3 \
  --correction-epochs 3 \
  --batch-size 128 \
  --hidden-dim 32 \
  --methods concat,warmup,bc_mcsgn \
  --device auto \
  --no-plots
