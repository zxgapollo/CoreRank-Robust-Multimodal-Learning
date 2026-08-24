#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-outputs/ad_scm_shortcut_no_demo_method_a100}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

sbatch \
  --account=ctbrowngrp \
  --partition=gpu-a100-h \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=24G \
  --time=02:00:00 \
  --job-name=ad-nodemo \
  --output="${LOG_DIR}/%x-%j.out" \
  --wrap="export PYTHON=/group/datalabgrp/xgzhu/env/corerank_synth/bin/python && export PYTHONPATH=\$PWD/src && bash scripts/run_farm_ad_shortcut_no_demo_method.sh ${OUT_DIR}"
