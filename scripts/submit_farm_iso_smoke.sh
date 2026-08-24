#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}
LOG_DIR="$PROJECT_DIR/logs"
BASE=${1:-outputs/iso_smoke_gpu}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-ctbrowngrp}

mkdir -p "$LOG_DIR"

sbatch \
  --partition=gpu-a100-h \
  --time=02:00:00 \
  --cpus-per-task=4 \
  --mem=32G \
  --nodes=1 \
  --gres=gpu:1 \
  -A "$SLURM_ACCOUNT" \
  --job-name=iso-smoke \
  --output="$LOG_DIR/%x-%j.out" \
  --error="$LOG_DIR/%x-%j.err" \
  --wrap="cd '$PROJECT_DIR' && bash scripts/run_farm_iso_smoke.sh '$BASE'"
