#!/usr/bin/env bash
#SBATCH --partition=gpu-a100-h
#SBATCH --account=datalabgrp
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00

set -euo pipefail

MODEL=${MODEL:?Set MODEL=spmnet or MODEL=transformer}
PROJECT_DIR=${PROJECT_DIR:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/outputs/icu_cross_mortality_v1}
SEEDS=${SEEDS:-"2026 2027 2028 2029 2030"}

for seed in $SEEDS; do
  MODEL="$MODEL" \
  SEED="$seed" \
  PROJECT_DIR="$PROJECT_DIR" \
  RUN_ROOT="$RUN_ROOT" \
  bash "$PROJECT_DIR/scripts/train_icu_cross_a100.sh"
done
