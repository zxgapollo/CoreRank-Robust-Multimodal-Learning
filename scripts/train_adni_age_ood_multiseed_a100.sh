#!/usr/bin/env bash
#SBATCH --partition=gpu-a100-h
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=04:00:00

set -euo pipefail

MODEL=${1:?Usage: train_adni_age_ood_multiseed_a100.sh spmnet|transformer}
PROJECT_DIR=${PROJECT_DIR:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}
DATA_ROOT=${DATA_ROOT:-/group/datalabgrp/xgzhu/datasets/adni-aibl-cross-dataset}
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/outputs/adni_age_ood_full_no_csf_multiseed_v1}
SPLIT_CSV=${SPLIT_CSV:-$DATA_ROOT/manifests/adni/adni_spmnet_age_ood_seed2026.csv}
SPLIT_SUMMARY=${SPLIT_SUMMARY:-$DATA_ROOT/manifests/adni/adni_spmnet_age_ood_seed2026_report.json}
SEEDS=${SEEDS:-"2026 2027 2028 2029 2030"}

for seed in $SEEDS; do
  PROJECT_DIR="$PROJECT_DIR" \
  DATA_ROOT="$DATA_ROOT" \
  RUN_ROOT="$RUN_ROOT" \
  SPLIT_CSV="$SPLIT_CSV" \
  SPLIT_SUMMARY="$SPLIT_SUMMARY" \
  SEED="$seed" \
  OUTPUT_DIR="$RUN_ROOT/$MODEL/seed_$seed" \
  bash "$PROJECT_DIR/scripts/train_adni_model_a100.sh" "$MODEL"
done
