#!/usr/bin/env bash
#SBATCH --account=ctbrowngrp
#SBATCH --partition=gpu-a100-h
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=04:00:00

set -euo pipefail

VARIANT=${VARIANT:?Set VARIANT to an ADNI SPMNet ablation name}
SEED=${SEED:-2026}
PROJECT_DIR=${PROJECT_DIR:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}
DATA_ROOT=${DATA_ROOT:-/group/datalabgrp/xgzhu/datasets/adni-aibl-cross-dataset}
PYTHON_BIN=${PYTHON_BIN:-/group/datalabgrp/xgzhu/env/corerank_synth/bin/python}
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/outputs/adni_spmnet_ablation_iclr_v1}
SPLIT_CSV=${SPLIT_CSV:-$DATA_ROOT/manifests/adni/adni_spmnet_age_ood_seed2026.csv}
SPLIT_SUMMARY=${SPLIT_SUMMARY:-$DATA_ROOT/manifests/adni/adni_spmnet_age_ood_seed2026_report.json}
EPOCHS=${EPOCHS:-80}
PATIENCE=${PATIENCE:-15}
BATCH_SIZE=${BATCH_SIZE:-4}
WORKERS=${WORKERS:-8}
OUTPUT_DIR=${OUTPUT_DIR:-$RUN_ROOT/$VARIANT/seed_$SEED}

extra_args=()
case "$VARIANT" in
  no_incidence|no_task_mask|no_private|no_reconstruction|no_sparsity|no_witness|no_modality_dropout|mean_fusion|direct_bypass)
    extra_args+=(--ablation "$VARIANT")
    ;;
  no_demographics)
    extra_args+=(--exclude-groups demographics)
    ;;
  no_cognition)
    extra_args+=(--exclude-groups cognition)
    ;;
  no_behavior)
    extra_args+=(--exclude-groups behavior)
    ;;
  no_genetics_history)
    extra_args+=(--exclude-groups genetics_history)
    ;;
  no_mri)
    extra_args+=(--exclude-mri)
    ;;
  latent_16)
    extra_args+=(--latent 16)
    ;;
  latent_64)
    extra_args+=(--latent 64)
    ;;
  private_4)
    extra_args+=(--private 4)
    ;;
  private_16)
    extra_args+=(--private 16)
    ;;
  modality_dropout_030)
    extra_args+=(--modality-dropout 0.30)
    ;;
  *)
    printf 'Unknown VARIANT: %s\n' "$VARIANT" >&2
    exit 2
    ;;
esac

if [[ -e "$OUTPUT_DIR/metrics.json" ]]; then
  printf 'Refusing to overwrite completed run: %s\n' "$OUTPUT_DIR/metrics.json" >&2
  exit 3
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_DIR/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_DIR/src"

"$PYTHON_BIN" -m adni_real.run_experiment \
  --model spmnet \
  --master-csv "$DATA_ROOT/metadata/adni/processed/adni_t1_baseline_multimodal_master.csv" \
  --cache-root "$DATA_ROOT/processed/brain96" \
  --split-csv "$SPLIT_CSV" \
  --split-summary "$SPLIT_SUMMARY" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  --hidden 128 \
  --latent 32 \
  --private 8 \
  --lr 2e-4 \
  --modality-dropout 0.15 \
  --amp \
  "${extra_args[@]}"

