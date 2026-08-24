#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}
RUN_ROOT=${1:-$PROJECT_DIR/outputs/adni_spmnet_ablation_iclr_v1}
BASELINE_ROOT=${BASELINE_ROOT:-$PROJECT_DIR/outputs/adni_age_ood_full_no_csf_multiseed_v1}
PYTHON_BIN=${PYTHON_BIN:-/group/datalabgrp/xgzhu/env/corerank_synth/bin/python}
SBATCH_BIN=${SBATCH_BIN:-$(command -v sbatch)}

variants=(
  no_incidence
  no_task_mask
  no_private
  no_reconstruction
  no_sparsity
  no_witness
  no_modality_dropout
  mean_fusion
  direct_bypass
  no_demographics
  no_cognition
  no_behavior
  no_genetics_history
  no_mri
  latent_16
  latent_64
  private_4
  private_16
  modality_dropout_030
)

if [[ -e "$RUN_ROOT/job_manifest.tsv" ]]; then
  printf 'Refusing to submit into an existing manifest: %s\n' "$RUN_ROOT/job_manifest.tsv" >&2
  exit 3
fi

mkdir -p "$RUN_ROOT/logs/smoke" "$RUN_ROOT/logs/main"
manifest="$RUN_ROOT/job_manifest.tsv"
printf 'stage\tvariant\tjob_id\tdependency\n' > "$manifest"

main_jobs=()
no_dropout_job=""
for variant in "${variants[@]}"; do
  smoke_job=$(
    "$SBATCH_BIN" --parsable \
      --account=ctbrowngrp \
      --partition=gpu-a100-h \
      --gres=gpu:a100:1 \
      --cpus-per-task=10 \
      --mem=80G \
      --time=00:30:00 \
      --job-name="adni-smk-${variant:0:10}" \
      --output="$RUN_ROOT/logs/smoke/${variant}_%j.out" \
      --error="$RUN_ROOT/logs/smoke/${variant}_%j.err" \
      --export="ALL,PROJECT_DIR=$PROJECT_DIR,RUN_ROOT=$RUN_ROOT/smoke,VARIANT=$variant,SEED=2026,EPOCHS=1,PATIENCE=1,WORKERS=4" \
      "$PROJECT_DIR/scripts/train_adni_spmnet_ablation_a100.sh"
  )
  printf 'smoke\t%s\t%s\t\n' "$variant" "$smoke_job" >> "$manifest"

  main_job=$(
    "$SBATCH_BIN" --parsable \
      --dependency="afterok:$smoke_job" \
      --account=ctbrowngrp \
      --partition=gpu-a100-h \
      --gres=gpu:a100:1 \
      --cpus-per-task=10 \
      --mem=80G \
      --time=04:00:00 \
      --job-name="adni-abl-${variant:0:10}" \
      --output="$RUN_ROOT/logs/main/${variant}_%j.out" \
      --error="$RUN_ROOT/logs/main/${variant}_%j.err" \
      --export="ALL,PROJECT_DIR=$PROJECT_DIR,RUN_ROOT=$RUN_ROOT,VARIANT=$variant" \
      "$PROJECT_DIR/scripts/train_adni_spmnet_ablation_multiseed_a100.sh"
  )
  printf 'main\t%s\t%s\tafterok:%s\n' "$variant" "$main_job" "$smoke_job" >> "$manifest"
  main_jobs+=("$main_job")
  if [[ "$variant" == "no_modality_dropout" ]]; then
    no_dropout_job="$main_job"
  fi
done

missingness_job=$(
  "$SBATCH_BIN" --parsable \
    --dependency="afterok:$no_dropout_job" \
    --account=ctbrowngrp \
    --partition=gpu-a100-h \
    --gres=gpu:a100:1 \
    --cpus-per-task=10 \
    --mem=80G \
    --time=04:00:00 \
    --job-name=adni-abl-missing \
    --output="$RUN_ROOT/logs/main/missingness_%j.out" \
    --error="$RUN_ROOT/logs/main/missingness_%j.err" \
    --export="ALL,PROJECT_DIR=$PROJECT_DIR,RUN_ROOT=$RUN_ROOT,BASELINE_ROOT=$BASELINE_ROOT" \
    "$PROJECT_DIR/scripts/evaluate_adni_spmnet_missingness_a100.sh"
)
printf 'evaluation\tmissingness\t%s\tafterok:%s\n' "$missingness_job" "$no_dropout_job" >> "$manifest"

all_dependencies=$(IFS=:; printf '%s' "${main_jobs[*]}:$missingness_job")
summary_job=$(
  "$SBATCH_BIN" --parsable \
    --dependency="afterok:$all_dependencies" \
    --partition=high \
    --account=datalabgrp \
    --cpus-per-task=2 \
    --mem=8G \
    --time=00:30:00 \
    --job-name=adni-abl-summary \
    --output="$RUN_ROOT/logs/main/summary_%j.out" \
    --error="$RUN_ROOT/logs/main/summary_%j.err" \
    --wrap="cd '$PROJECT_DIR' && '$PYTHON_BIN' '$PROJECT_DIR/scripts/summarize_adni_spmnet_ablation.py' --run-root '$RUN_ROOT' --baseline-root '$BASELINE_ROOT' --output-json '$RUN_ROOT/ablation_summary.json' --output-table '$RUN_ROOT/ablation_table.csv' --output-per-seed '$RUN_ROOT/ablation_per_seed.csv'"
)
printf 'summary\tall\t%s\tafterok:%s\n' "$summary_job" "$all_dependencies" >> "$manifest"

printf 'run_root=%s\nmanifest=%s\nsmoke_jobs=%s\nmain_jobs=%s\nmissingness_job=%s\nsummary_job=%s\n' \
  "$RUN_ROOT" "$manifest" "${#variants[@]}" "${#main_jobs[@]}" "$missingness_job" "$summary_job"

