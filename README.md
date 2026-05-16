# CoreRank Synthetic Benchmark

This repository contains a first synthetic benchmark and PyTorch implementation for a constrained-likelihood CoreRank model.

See `DESIGN.md` for the full algorithm and experimental design.

## Scientific scope

The core idea is sound as a synthetic-testable claim if stated locally:

> A modality is useful for robust prediction when it adds nuisance-adjusted Fisher directions for the recoverable disease-core block.

The benchmark is not claiming global causal graph identification. It tests local disease-core recoverability, sparse modality-to-core footprint recovery, missing-modality behavior, robustness under a controlled spurious-bias shift, robustness under a controlled domain/mechanism shift, and a structural innovation parameterization over recovered core coordinates.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m corerank_synth.run_experiment --scenario complementary --n-train 512 --n-val 256 --n-test 256 --epochs 3 --batch-size 128 --output-dir outputs/smoke
```

The current training default uses dimension-normalized reconstruction loss, an SEM innovation prior `E = (I - A)Z - Gamma C`, and a classifier on innovation coordinates. Farm scripts also add context-invariance penalties for known synthetic bias/domain variables and select the best checkpoint by predictive sufficiency first, then lower context leakage. Use `--recon-reduction sum --no-sem-prior` to reproduce the earlier reconstruction-heavy baseline family.

Run tests:

```bash
pytest
```

## Main run

```bash
bash scripts/run_smoke.sh
bash scripts/run_main_synthetic.sh
```

## Useful server commands

Complementary-rank smoke test:

```bash
python -m corerank_synth.run_experiment \
  --scenario complementary \
  --n-train 1024 --n-val 512 --n-test 512 \
  --epochs 5 --batch-size 128 \
  --output-dir outputs/smoke_complementary
```

Biased-modality robustness smoke test:

```bash
python -m corerank_synth.run_experiment \
  --scenario biased \
  --biased-modality 0 \
  --bias-strength 2.0 \
  --train-bias-corr 0.85 \
  --test-bias-corr -0.5 \
  --n-train 1024 --n-val 512 --n-test 512 \
  --epochs 5 --batch-size 128 \
  --output-dir outputs/smoke_biased
```

Domain/mechanism-shift smoke test:

```bash
python -m corerank_synth.run_experiment \
  --scenario domain \
  --domain-shifted-modality 0 \
  --domain-shift-strength 1.5 \
  --n-train 1024 --n-val 512 --n-test 512 \
  --epochs 5 --batch-size 128 \
  --output-dir outputs/smoke_domain
```

## UC Davis Farm A100

Project path:

```bash
/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning
```

Environment path:

```bash
/group/datalabgrp/xgzhu/env/corerank_synth
```

Interactive A100 allocation:

```bash
bash scripts/farm_srun_a100.sh
```

Run Farm smoke tests inside the allocation:

```bash
bash scripts/run_farm_smoke.sh
```

Submit the full synthetic grid as a detached Slurm job:

```bash
bash scripts/submit_farm_main.sh outputs/main_grid_gpu
```
