#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-outputs/ad_scm_shortcut_s_embed_a100}"

"${PYTHON:-python}" -m ad_scm.run_experiment \
  --output-dir "${OUT_DIR}" \
  --mode shortcut \
  --seed 0 \
  --n-train 10000 \
  --n-val 2000 \
  --n-test 4000 \
  --x-dim 8 \
  --base-noise 0.25 \
  --disease-noise-scale 2.75 \
  --demo-noise-scale 0.20 \
  --demo-to-s-strength 0.25 \
  --shortcut-strengths "0.25,1.0,2.0,3.0,4.0" \
  --shortcut-test-strength 0.25 \
  --noise-train-scale 1.0 \
  --epochs 45 \
  --correction-epochs 70 \
  --batch-size 256 \
  --hidden-dim 64 \
  --modality-dropout 0.15 \
  --recon-weight 1.0 \
  --proto-weight 0.5 \
  --graph-l1-weight 0.001 \
  --mask-l1-weight 0.001 \
  --edge-entropy-weight 0.001 \
  --mask-entropy-weight 0.002 \
  --methods no_demo,warmup,ad_bc_mcsgn_s_embed \
  --device cuda \
  --verbose
