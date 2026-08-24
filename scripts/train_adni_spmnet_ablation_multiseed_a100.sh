#!/usr/bin/env bash
#SBATCH --account=ctbrowngrp
#SBATCH --partition=gpu-a100-h
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=04:00:00

set -euo pipefail

VARIANT=${VARIANT:?Set VARIANT to an ADNI SPMNet ablation name}
PROJECT_DIR=${PROJECT_DIR:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/outputs/adni_spmnet_ablation_iclr_v1}
SEEDS=${SEEDS:-"2026 2027 2028 2029 2030"}

for seed in $SEEDS; do
  VARIANT="$VARIANT" \
  SEED="$seed" \
  PROJECT_DIR="$PROJECT_DIR" \
  RUN_ROOT="$RUN_ROOT" \
  OUTPUT_DIR="$RUN_ROOT/$VARIANT/seed_$seed" \
  bash "$PROJECT_DIR/scripts/train_adni_spmnet_ablation_a100.sh"
done

