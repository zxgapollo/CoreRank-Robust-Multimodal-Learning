#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/outputs/icu_cross_metre_hourly_multimodal_v1}
mkdir -p "$RUN_ROOT/logs"

cache_job=$(sbatch --parsable \
  --job-name=metre_mm_cache \
  --output="$RUN_ROOT/logs/cache_%j.out" \
  --error="$RUN_ROOT/logs/cache_%j.err" \
  --export="ALL,PROJECT_DIR=$PROJECT_DIR,RUN_ROOT=$RUN_ROOT" \
  "$PROJECT_DIR/scripts/build_icu_cross_metre_hourly_cache.sh")

smoke_jobs=()
for model in spmnet transformer; do
  job=$(sbatch --parsable \
    --dependency="afterok:$cache_job" \
    --job-name="metre_mm_smoke_${model:0:3}" \
    --output="$RUN_ROOT/logs/smoke_${model}_%j.out" \
    --error="$RUN_ROOT/logs/smoke_${model}_%j.err" \
    --export="ALL,MODEL=$model,SEED=99,EPOCHS=2,PATIENCE=2,BATCH_SIZE=256,PROJECT_DIR=$PROJECT_DIR,RUN_ROOT=$RUN_ROOT/smoke,CACHE_ROOT=$RUN_ROOT/cache" \
    "$PROJECT_DIR/scripts/train_icu_cross_a100.sh")
  smoke_jobs+=("$job")
done

smoke_dependency=$(IFS=:; echo "${smoke_jobs[*]}")
training_jobs=()
for model in spmnet transformer; do
  job=$(sbatch --parsable \
    --dependency="afterok:$smoke_dependency" \
    --time=08:00:00 \
    --job-name="metre_mm_${model:0:3}_5seed" \
    --output="$RUN_ROOT/logs/${model}_5seed_%j.out" \
    --error="$RUN_ROOT/logs/${model}_5seed_%j.err" \
    --export="ALL,MODEL=$model,BATCH_SIZE=256,PROJECT_DIR=$PROJECT_DIR,RUN_ROOT=$RUN_ROOT,CACHE_ROOT=$RUN_ROOT/cache" \
    "$PROJECT_DIR/scripts/train_icu_cross_multiseed_a100.sh")
  training_jobs+=("$job")
done

training_dependency=$(IFS=:; echo "${training_jobs[*]}")
summary_job=$(sbatch --parsable \
  --dependency="afterok:$training_dependency" \
  --partition=high \
  --account=datalabgrp \
  --cpus-per-task=1 \
  --mem=4G \
  --time=00:20:00 \
  --job-name=metre_mm_summary \
  --output="$RUN_ROOT/logs/summary_%j.out" \
  --error="$RUN_ROOT/logs/summary_%j.err" \
  --wrap="cd '$PROJECT_DIR' && /group/datalabgrp/xgzhu/env/corerank_synth/bin/python '$PROJECT_DIR/scripts/summarize_icu_cross.py' --run-root '$RUN_ROOT' --output '$RUN_ROOT/summary.json'")

printf 'cache_job=%s\nsmoke_jobs=%s\ntraining_jobs=%s\nsummary_job=%s\n' \
  "$cache_job" "${smoke_jobs[*]}" "${training_jobs[*]}" "$summary_job"
