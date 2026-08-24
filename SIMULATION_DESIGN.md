# ISO OOD Simulation Design

## Material Passport

- Artifact: code experiment plan
- Project: Intrinsic State Observability synthetic benchmark
- Purpose: test OOD generalization from graph-induced latent state recovery
- Date: 2026-06-05
- Status: implemented for first A100 run

## Core Formulation

The synthetic benchmark follows the final OOD formulation from the discussion:

```math
Z_j = f_j(Z_{\mathrm{pa}(j)}, \epsilon_j),
\qquad
X_i = g_i(Z_{\Gamma_i}, N_i^\tau, R_i^\tau),
\qquad
Y \sim p(Y \mid S^\star).
```

Here

```math
S^\star = \rho(Z, G_Z^\star)
```

is the graph-induced, label-relevant latent state. Train and test are not IID.
The target test split changes observation-layer nuisance and residual factors:

```math
p^{\mathrm{train}}(N_i, R_i) \ne p^{\mathrm{test}}(N_i, R_i),
```

and may also change

```math
p^\tau(N_i, R_i \mid Z).
```

The latent causal graph, latent mechanisms, and label mechanism remain fixed:

```math
G_Z^{\mathrm{train}} = G_Z^{\mathrm{test}} = G_Z^\star,
\qquad
p^{\mathrm{train}}(Y \mid S^\star)
=
p^{\mathrm{test}}(Y \mid S^\star).
```

This makes the key contrast explicit:

```math
p^{\mathrm{train}}(Y \mid X) \ne p^{\mathrm{test}}(Y \mid X),
\qquad
p^{\mathrm{train}}(Y \mid S^\star)
=
p^{\mathrm{test}}(Y \mid S^\star).
```

## Implemented Generator

The implemented first-round generator is:

```text
Y ~ p(Y | S*)
X_i = tanh(A_i S* + B_i U_i^tau + C_i Q^tau + eps_i^tau)
```

`train`, `val`, and `id_test` are source-domain splits. `test` is the target
OOD split.

The target split changes observation-layer factors through:

- `ood_residual_shift`: target-domain mean shift in modality-private residuals.
- `train_nuisance_corr` and `test_nuisance_corr`: source and target correlations
  between nuisance residuals and labels, allowing `p(U_i | Y)` to flip or weaken.
- `ood_noise_multiplier`: target-domain modality quality/noise shift, applied to
  the noisy modality scenario.
- `train_shortcut_corr` and `test_shortcut_corr`: source/target shortcut
  correlation for shortcut-style scenarios.

The label mechanism and latent state graph are held fixed across all splits.

## Scenarios

- `complementary`: each modality observes different state coordinates; OOD comes
  from nuisance/residual shift.
- `redundant`: one modality duplicates another state footprint; tests whether raw
  extra information without new state directions helps.
- `nuisance_only`: the last modality has no state signal and only residual signal;
  source nuisance-label correlation should not help OOD.
- `shortcut`: a noncausal shortcut is correlated with the label in source and
  flipped or weakened in target.
- `noisy_modality`: target noise increases for a designated modality.
- `mediated_context`: a demographic-like context modality observes an upstream
  latent coordinate that affects downstream disease coordinates through the
  stable graph. Its mediated path is valid, while its direct shortcut residual is
  target-unstable.

## Primary Comparisons

Models:

- unimodal MLPs
- concat MLP
- late-fusion MLP
- ISO-PoE latent-state model
- oracle state MLP

Metrics:

- ID vs OOD AUC, accuracy, NLL
- state recovery R2 / CKA / MCC
- oracle label-relevant observability `lambda_Y(M)`
- ambiguity proxy `Uhat_Y(M)`
- effective nuisance-adjusted information rank

Key figures:

- observability vs OOD AUC
- ambiguity vs OOD NLL
- sample-complexity curve under OOD test
- source ID vs target OOD shortcut comparison
- state recovery comparison

## Farm A100 Run

Smoke:

```bash
bash scripts/submit_farm_iso_smoke.sh outputs/iso_ood_smoke_gpu
```

Main grid:

```bash
bash scripts/submit_farm_iso_main.sh outputs/iso_ood_main_grid_gpu
```

The Farm submit scripts default to `SLURM_ACCOUNT=ctbrowngrp` and can be
overridden without editing files.
