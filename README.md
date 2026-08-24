# Intrinsic State Observability Synthetic Experiments

This repo now contains a clean first-round PyTorch synthetic benchmark for the
**Intrinsic State Observability (ISO)** hypothesis:

> Multimodal learning improves over unimodal learning when additional modalities
> reduce label-relevant ambiguity of an intrinsic latent state `S*`, rather than
> merely adding raw information.

The new implementation lives in `src/iso_synth` and follows the structural
generator

```text
Y ~ p(Y | S*)
X_i = g_i(A_i S* + B_i U_i + C_i Q + eps_i)
```

It implements:

- synthetic generator for complementary, redundant, nuisance-only, shortcut,
  noisy-modality, and mediated-context cases;
- source `train`/`val`/`id_test` splits and a target OOD `test` split where the
  latent state graph and label mechanism remain fixed while modality residuals,
  nuisance-label correlations, shortcut correlations, or noise quality shift;
- `UnimodalMLP`, `ConcatMLP`, `LateFusionMLP`, `ISO-PoE`, and
  `OracleStateMLP` baselines;
- oracle label-relevant observability `lambda_Y(M)`;
- state ambiguity proxy `Uhat_Y(M)`;
- AUC / ACC / NLL plus state recovery R2 / CKA / MCC;
- sample-complexity grids and the five first-round figures.

## ISO quick start

```bash
python -m iso_synth.run_experiment \
  --scenarios complementary,redundant,nuisance_only,shortcut,noisy_modality,mediated_context \
  --n-train-grid 128,512 \
  --seeds 0 \
  --epochs 10 \
  --output-dir outputs/iso_first_pass
```

Smoke run:

```bash
bash scripts/run_iso_smoke.sh
```

Expected outputs:

```text
outputs/iso_first_pass/results.csv
outputs/iso_first_pass/figures/fig_observability_vs_auc.png
outputs/iso_first_pass/figures/fig_ambiguity_vs_nll.png
outputs/iso_first_pass/figures/fig_sample_complexity.png
outputs/iso_first_pass/figures/fig_shortcut_ood.png
outputs/iso_first_pass/figures/fig_state_recovery.png
```

The older CoreRank synthetic prototype is still available under
`src/corerank_synth` for reference.

---

# Legacy CoreRank Synthetic Benchmark

This repository contains a first synthetic benchmark and PyTorch implementation for a constrained-likelihood CoreRank model.

See `DESIGN.md` for the full algorithm and experimental design.

## Scientific scope

The core idea is sound as a synthetic-testable claim if stated locally:

> A modality is useful for robust prediction when it adds nuisance-adjusted Fisher directions for the recoverable disease-core block.

The benchmark is not claiming global causal graph identification. It tests local disease-core recoverability, sparse modality-to-core footprint recovery, missing-modality behavior, robustness under controlled shortcut, measurement, and semantic-proxy OOD shifts, and a structural innovation parameterization over recovered core coordinates.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m corerank_synth.run_experiment --scenario complementary --n-train 512 --n-val 256 --n-test 256 --epochs 3 --batch-size 128 --output-dir outputs/smoke
```

The current training default uses dimension-normalized reconstruction loss, an SEM innovation prior `E = (I - A)Z`, and a classifier on innovation coordinates. Demographic/comorbidity/site variables are modeled as ordinary modalities, not privileged context inputs. Synthetic shortcut/domain variables are used only for diagnostics, and checkpoint selection uses ID validation AUROC. Use `--recon-reduction sum --no-sem-prior` to reproduce the earlier reconstruction-heavy baseline family.

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

Covariate-modality shortcut robustness smoke test:

```bash
python -m corerank_synth.run_experiment \
  --scenario shortcut \
  --n-modalities 4 \
  --biased-modality 3 \
  --bias-strength 2.5 \
  --train-bias-corr 0.85 \
  --test-bias-corr -0.5 \
  --n-train 1024 --n-val 512 --n-test 512 \
  --epochs 5 --batch-size 128 \
  --output-dir outputs/smoke_shortcut
```

Domain/mechanism-shift smoke test:

```bash
python -m corerank_synth.run_experiment \
  --scenario measurement \
  --n-modalities 4 \
  --domain-shifted-modality 0 \
  --domain-shift-strength 2.0 \
  --n-train 1024 --n-val 512 --n-test 512 \
  --epochs 5 --batch-size 128 \
  --output-dir outputs/smoke_measurement
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
SLURM_ACCOUNT=ctbrowngrp bash scripts/submit_farm_iso_main.sh outputs/iso_ood_main_grid_gpu
```
