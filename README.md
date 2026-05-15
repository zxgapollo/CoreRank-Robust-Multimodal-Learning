# CoreRank Synthetic Benchmark

This repository contains a first synthetic benchmark and PyTorch implementation for a constrained-likelihood CoreRank model.

See `DESIGN.md` for the full algorithm and experimental design.

## Scientific scope

The core idea is sound as a synthetic-testable claim if stated locally:

> A modality is useful for robust prediction when it adds nuisance-adjusted Fisher directions for the recoverable disease-core block.

The benchmark is not claiming global causal graph identification. It tests local disease-core recoverability, sparse modality-to-core footprint recovery, missing-modality behavior, and robustness under a controlled spurious-bias shift.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m corerank_synth.run_experiment --scenario complementary --n-train 512 --n-val 256 --n-test 256 --epochs 3 --batch-size 128 --output-dir outputs/smoke
```

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
