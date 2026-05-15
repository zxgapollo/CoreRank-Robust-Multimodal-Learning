#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${1:-/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning}

cd "$PROJECT_DIR"
exec srun \
  --partition=gpu-a100-h \
  --time=48:00:00 \
  --cpus-per-task=8 \
  --mem=120G \
  --nodes=1 \
  --gres=gpu:1 \
  -A datalabgrp \
  --pty /bin/bash -il
