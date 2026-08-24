#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-outputs/ad_scm_noise_sweep_a100}"

"${PYTHON:-python}" -m ad_scm.run_experiment \
  --output-dir "${OUT_DIR}" \
  --mode noise \
  --seed 0 \
  --n-train 10000 \
  --n-val 2000 \
  --n-test 4000 \
  --x-dim 8 \
  --base-noise 0.25 \
  --shortcut-strengths "0.55" \
  --noise-scales "1.0,1.5,2.0,3.0,4.0" \
  --noise-modality mri \
  --epochs 45 \
  --correction-epochs 45 \
  --batch-size 256 \
  --hidden-dim 64 \
  --modality-dropout 0.10 \
  --recon-weight 0.5 \
  --proto-weight 0.5 \
  --graph-l1-weight 0.001 \
  --mask-l1-weight 0.001 \
  --methods concat,no_demo,late_fusion,warmup,ad_bc_mcsgn \
  --device cuda \
  --verbose
