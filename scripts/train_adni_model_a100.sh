#!/usr/bin/env bash
#SBATCH --account=ctbrowngrp
#SBATCH --partition=gpu-a100-h
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=24:00:00

set -euo pipefail

MODEL=${1:?Usage: train_adni_model_a100.sh spmnet|transformer}
PROJECT_DIR=${PROJECT_DIR:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}
DATA_ROOT=${DATA_ROOT:-/group/datalabgrp/xgzhu/datasets/adni-aibl-cross-dataset}
PYTHON_BIN=${PYTHON_BIN:-/group/datalabgrp/xgzhu/env/corerank_synth/bin/python}
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/outputs/adni_spmnet_transformer_a100_v1}
SPLIT_CSV=${SPLIT_CSV:-$DATA_ROOT/manifests/adni/adni_spmnet_split_seed2026.csv}
SPLIT_SUMMARY=${SPLIT_SUMMARY:-$DATA_ROOT/manifests/adni/adni_spmnet_split_seed2026_summary.json}
EXCLUDE_GROUPS=${EXCLUDE_GROUPS:-}
SEED=${SEED:-2026}
OUTPUT_DIR=${OUTPUT_DIR:-$RUN_ROOT/$MODEL}

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_DIR/src"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR/src"

"$PYTHON_BIN" -m adni_real.run_experiment \
  --model "$MODEL" \
  --master-csv "$DATA_ROOT/metadata/adni/processed/adni_t1_baseline_multimodal_master.csv" \
  --cache-root "$DATA_ROOT/processed/brain96" \
  --split-csv "$SPLIT_CSV" \
  --split-summary "$SPLIT_SUMMARY" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --epochs 80 \
  --patience 15 \
  --batch-size 4 \
  --workers 8 \
  --hidden 128 \
  --latent 32 \
  --lr 2e-4 \
  --modality-dropout 0.15 \
  --amp \
  ${EXCLUDE_GROUPS:+--exclude-groups $EXCLUDE_GROUPS}
