#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
export PYTHONPATH="${PYTHONPATH:-}:src"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/iso_mpl}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
mkdir -p "$MPLCONFIGDIR"

"${PYTHON_BIN}" -m iso_synth.run_experiment \
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
  --output-dir outputs/iso_smoke
