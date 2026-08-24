# AD-SCM Multimodal OOD Simulation Design

This benchmark uses the Alzheimer's disease example as the main simulation setting.
It is independent of the earlier abstract six-factor SCM.

## Latent SCM

Latent variables:

- `D`: demographic/risk latent factor, e.g. age/risk burden.
- `S`: latent AD disease state.
- `Z1`: amyloid burden.
- `Z2`: tau pathology.
- `Z3`: neurodegeneration.
- `Z4`: cognitive reserve.

Stable structural equations:

```text
D  := f_D(U_D)
Z4 := f_4(U_4)
S  := f_S(D, U_S)
Z1 := f_1(S, U_1)
Z2 := f_2(S, Z1, U_2)
Z3 := f_3(S, Z2, D, U_3)
```

The stable graph is:

```text
D -> S -> Z1 -> Z2 -> Z3
D -> Z3
S -> Z2
S -> Z3
Z4 -> Y_obs
Z2 -> Y_obs
Z3 -> Y_obs
```

`D` is a real causal parent of `S` and `Z3`. In this experiment, the shortcut is
the amount of the `D -> Z3 -> Y_obs` path that enters the observed diagnosis. This
captures the fact that the observed AD label can mix AD pathology with age-related
neurodegeneration/cognitive decline.

## Multimodal Observations

Modalities:

```text
X_PET     := g_PET(Z1, Z2, C_PET, U_PET)
X_MRI     := g_MRI(Z3, C_MRI, U_MRI)
X_CogTest := g_Cog(Z3, Z4, C_Cog, U_Cog)
X_Demo    := g_Demo(D, C_Demo, U_Demo)
```

`X_Demo` contains a stable demographic signal from `D`. We do not add a fake
label artifact directly to `X_Demo`; the shortcut operates through the latent path
`D -> Z3 -> Y_obs`.

## Observed Label

The observed AD label is not equal to `S`.

```text
clinical_score = h(Z2, Z3, Z4, U_Y)
Y_obs = 0 if clinical_score < t1
Y_obs = 1 if t1 <= clinical_score < t2
Y_obs = 2 if clinical_score >= t2
```

This captures cases where latent AD state can be abnormal while observed diagnosis
is not yet AD, e.g. high amyloid but low neurodegeneration and high reserve.

## Experiment 1: Demographic Shortcut Sweep

Train split:

```text
Z3 := f_3(S, Z2, beta_train * D, U_3)
```

ID test:

```text
Z3 := f_3(S, Z2, beta_train * D, U_3)
```

OOD test:

```text
Z3 := f_3(S, Z2, beta_test * D, U_3)
```

Sweep:

```text
beta_train in {0, 0.2, 0.4, 0.6, 0.8, 1.0}
beta_test fixed, e.g. 0.1
```

Expected pattern: as `beta_train` grows, observed labels in training contain more
age-related neurodegeneration through `Z3`. If test has a weaker `D -> Z3` path,
a direct multimodal predictor can over-rely on the demographic pathway learned in
training.

## Experiment 2: Single-Modality Noise Sweep

Train and ID test use the base modality noise.

OOD test increases noise only in one modality, e.g. MRI:

```text
X_MRI^ood = g_MRI(Z3, C_MRI, U_MRI) + sigma_ood * noise
```

Sweep:

```text
sigma_ood in {0.2, 0.5, 1.0, 1.5, 2.0, 3.0}
```

The latent SCM, graph, and label mechanism are unchanged.

## Metrics

Prediction:

- multiclass accuracy
- macro F1
- multiclass NLL
- one-vs-rest macro AUC

OOD degradation:

- ID accuracy minus OOD accuracy
- ID macro AUC minus OOD macro AUC

Diagnostics:

- class balance per split
- latent graph invariant check
- shortcut-component/label correlation in train, ID, and OOD
- per-modality noise level

## Important Boundary

The demographic latent `D` is not removed wholesale: it is a real parent of `S` and
can be a real parent of `Z3`. The experiment varies the strength of the `D -> Z3`
contribution across environments while preserving the graph skeleton and the label
definition `Y_obs = h(Z2, Z3, Z4)`.
