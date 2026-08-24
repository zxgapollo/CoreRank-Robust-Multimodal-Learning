#!/usr/bin/env bash
#SBATCH --partition=high
#SBATCH --account=datalabgrp
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}
RUN_ROOT=${RUN_ROOT:-$PROJECT_DIR/outputs/icu_cross_metre_hourly_multimodal_v1}
PYTHON_BIN=${PYTHON_BIN:-/group/datalabgrp/xgzhu/env/corerank_synth/bin/python}
MIMIC_ROOT=${MIMIC_ROOT:-/group/datalabgrp/xgzhu/CausalMedical/Dataset/raw_data/MIMIC-IV2/physionet.org/files/mimiciv/2.0}
EICU_ROOT=${EICU_ROOT:-/group/datalabgrp/xgzhu/CausalMedical/Dataset/raw_data/eICU/physionet.org/files/eicu-crd/2.0}

mkdir -p "$RUN_ROOT/cache"
PYTHONPATH="$PROJECT_DIR/src" "$PYTHON_BIN" -m icu_cross.build_metre_hourly_cache \
  --mimic-root "$MIMIC_ROOT" \
  --eicu-root "$EICU_ROOT" \
  --output-root "$RUN_ROOT/cache" \
  --observation-hours "${OBSERVATION_HOURS:-48}" \
  --gap-hours "${GAP_HOURS:-6}" \
  --max-los-hours "${MAX_LOS_HOURS:-240}" \
  --chunksize "${CHUNKSIZE:-1000000}" \
  ${MAX_STAYS:+--max-stays "$MAX_STAYS"}
