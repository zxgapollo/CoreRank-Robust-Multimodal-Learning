#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}
BASE=${1:-outputs/sfm_selective_factors_a100_v1}
LOG_DIR="$PROJECT_DIR/$BASE/logs"
mkdir -p "$LOG_DIR"

sbatch \
  --account=ctbrowngrp \
  --partition=gpu-a100-h \
  --gres=gpu:1 \
  --cpus-per-task=4 \
  --mem=24G \
  --time=04:00:00 \
  --job-name=sfm-selective \
  --output="$LOG_DIR/%x-%j.out" \
  --error="$LOG_DIR/%x-%j.err" \
  --wrap="cd '$PROJECT_DIR' && bash scripts/run_farm_sfm_multiseed.sh '$BASE'"
