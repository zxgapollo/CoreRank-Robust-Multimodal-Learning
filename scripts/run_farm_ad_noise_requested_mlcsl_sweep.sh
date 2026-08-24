#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-outputs/ad_scm_noise_requested_methods_a100_v1}"

"${PYTHON:-python}" -m ad_scm.run_experiment \
  --output-dir "${OUT_DIR}" \
  --mode noise \
  --seed 0 \
  --n-train 5000 \
  --n-val 1000 \
  --n-test 2000 \
  --x-dim 8 \
  --base-noise 0.25 \
  --disease-noise-scale 1.00 \
  --demo-noise-scale 0.20 \
  --demo-to-s-strength 0.25 \
  --shortcut-strengths "0.55" \
  --shortcut-test-strength 0.55 \
  --noise-scales "1.0,1.5,2.0,3.0,4.0" \
  --noise-modality mri \
  --epochs 35 \
  --correction-epochs 55 \
  --batch-size 256 \
  --hidden-dim 64 \
  --modality-dropout 0.10 \
  --recon-weight 1.0 \
  --proto-weight 0.5 \
  --graph-l1-weight 0.001 \
  --mask-l1-weight 0.001 \
  --edge-entropy-weight 0.001 \
  --mask-entropy-weight 0.002 \
  --counterfactual-demo-weight 0.25 \
  --methods demo_only,late_fusion_no_demo,concat_no_demo,concat,late_fusion,mlcsl,mlcsl_no_demo \
  --device cuda \
  --verbose
