#!/usr/bin/env bash
#SBATCH --account=ctbrowngrp
#SBATCH --partition=gpu-a100-h
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}
DATA_ROOT=${DATA_ROOT:-/group/datalabgrp/xgzhu/datasets/adni-aibl-cross-dataset}
PYTHON_BIN=${PYTHON_BIN:-/group/datalabgrp/xgzhu/env/corerank_synth/bin/python}
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/outputs/adni_spmnet_ablation_iclr_v1}
BASELINE_ROOT=${BASELINE_ROOT:-$PROJECT_DIR/outputs/adni_age_ood_full_no_csf_multiseed_v1}

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_DIR/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
cd "$PROJECT_DIR/src"

"$PYTHON_BIN" -m adni_real.evaluate_missingness \
  --master-csv "$DATA_ROOT/metadata/adni/processed/adni_t1_baseline_multimodal_master.csv" \
  --split-csv "$DATA_ROOT/manifests/adni/adni_spmnet_age_ood_seed2026.csv" \
  --baseline-root "$BASELINE_ROOT" \
  --no-dropout-root "$RUN_ROOT/no_modality_dropout" \
  --output-json "$RUN_ROOT/missingness_summary.json" \
  --output-csv "$RUN_ROOT/missingness_per_seed.csv" \
  --batch-size 8 \
  --workers 8 \
  --random-replicates 10

