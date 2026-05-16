# CoreRank Synthetic Benchmark and Model Design

This document specifies a deployable synthetic benchmark and a first implementation of **CoreRank**, a constrained-likelihood multimodal generative model for recovering a task-relevant disease core from biased, heterogeneous, and partially observed modalities.

The implementation is intentionally scoped to the synthetic benchmark first. It is designed so that Codex can extend it to real medical datasets after the core theory-method alignment is validated.

---

## 1. Research goal

We want to test the following claim:

> A medical modality helps robust prediction only when it contributes new **nuisance-adjusted Fisher-rank directions** for recovering a minimal disease core `Z*`; it may hurt when it mostly contributes redundant directions, domain-specific measurement shifts, or modality-private shortcuts.

The synthetic benchmark must therefore expose ground-truth values for:

1. disease core `Z*` and its latent causal graph `A*`,
2. modality-private nuisance variables `U_m`,
3. non-core context/shortcut variables such as age, site, scanner, visit timing, or gene covariates,
4. modality-to-core footprint graph `G*`,
5. true modality evidence matrices and effective core rank,
6. missing-modality, concept-shift, and domain-shift settings,
7. robust prediction targets.

The first implementation should **not** claim full causal graph recovery. It should test local recoverability of the disease-core block and, optionally, sparse footprint recovery.

---

## 2. Important formulation constraints

### 2.1 Do not claim global identifiability

The Fisher-rank condition is local. The correct claim is:

```text
Full nuisance-adjusted core Fisher rank implies local block recoverability / local identifiability / asymptotic posterior concentration of Z*, up to an invertible transformation.
```

Avoid claims such as:

```text
The disease state is globally recovered.
The full causal graph is identifiable.
The learned factors are automatically causal disease mechanisms.
```

### 2.2 Do not rely on parent/child modality roles

In the synthetic benchmark, all modalities are treated as observations carrying posterior evidence about `Z*`. Whether a modality is upstream or downstream of the disease core is not used by the method.

### 2.3 Distinguish the core causal graph from the footprint graph

The benchmark may expose a latent causal graph among core variables:

```text
A*[j, k] != 0 iff disease-core coordinate Z_k is a direct parent of Z_j.
```

This graph is useful for data generation and for motivating examples such as age affecting recognition. The first CoreRank implementation does **not** learn this graph and should not claim internal causal-DAG recovery.

The graph learned by the first implementation is instead the **modality-to-core footprint graph**:

```text
G[m, j] = 1  iff modality m contains recoverable evidence about disease-core coordinate Z_j.
```

This is different from an internal causal graph among disease factors.

The current implementation also includes an optional **core structural graph** learner:

```text
Z_j <- sum_k A[j, k] Z_k + E_j
```

It uses a NOTEARS-style acyclicity penalty and graph sparsity penalty. This should be presented as learning a soft structural prior over the recovered core coordinates, not as proof of full causal identification from observational data.

### 2.4 Core variables are not all modality-causing variables

The disease core is a minimal task-relevant latent block, not a bucket containing every variable that affects a modality. Variables such as age, sex, gene status, scanner/site, acquisition protocol, and visit timing may affect a modality or even affect an intermediate cognitive measurement, but they should not automatically become part of the disease core.

For example, age can affect recognition scores. If the target is Alzheimer disease status, the path

```text
Age -> recognition-like measurement -> classifier
```

can be a problematic shortcut when train and test age distributions differ. CoreRank should represent this as context/nuisance-adjusted evidence, not as proof that age itself is a disease-core coordinate.

### 2.5 Do not implement the objective as a loss pile

The paper formulation should be:

```math
max_{theta, phi, G} L_ELBO(theta, phi, G)

subject to
E_x log det(eps I + Kbar_O(x)) >= kappa,
Omega(G) <= s,
p_theta(y | z, u_1, ..., u_M) = p_theta(y | z).
```

The code uses an augmented Lagrangian / constrained relaxation:

```math
min -L_ELBO
  + lambda_rank [kappa - c_rank]_+
  + 0.5 rho_rank [kappa - c_rank]_+^2
  + lambda_sparse [Omega(G) - s]_+
  + 0.5 rho_sparse [Omega(G) - s]_+^2.
```

This is not described as an arbitrary regularizer stack. The rank and sparsity terms are relaxations of explicit constraints.

### 2.6 Conditional independence is an approximation

The simple model computes

```math
K_O(x) = sum_{m in O} I_m^core(z_hat_x).
```

This assumes that modalities are conditionally independent given the disease core and nuisance variables, or that we use a block-diagonal approximation to the joint Fisher. In medical data, shared unobserved nuisance can violate this. The code is therefore written so future versions can replace the sum by a joint Fisher estimator.

---

## 3. Mathematical objects

### 3.1 Latent variables

For each sample:

```math
Z* in R^r                    disease core
A*                           latent causal graph among disease-core coordinates
C                            observed or latent non-core context: age, sex, site, gene, scanner, visit timing
U_m in R^q                   modality-private nuisance / bias
Y ~ Bernoulli(sigmoid(beta^T Z* + nonlinear terms))
```

The synthetic code samples `Z*` from a simple acyclic structural equation model:

```math
Z_j = sum_{k in Pa(j)} A^*_{jk} Z_k + E_j.
```

This allows us to talk about causal relations among core coordinates while keeping graph recovery out of scope for the first method.

Each modality is generated from a subset of disease-core coordinates and its private nuisance:

```math
X_m = f_m(G*_m \odot Z*, U_m, C, B_m, D_m, eps_m).
```

`B_m` is an explicit spurious bias variable used for robustness testing. It affects the modality but is not a true cause of `Y`.
`D_m` is a domain/mechanism variable such as site or scanner shift. It changes the observation mechanism but not the disease label mechanism.

### 3.2 Nuisance-adjusted core Fisher

For a Gaussian decoder

```math
p_theta(x_m | z, u_m) = N(mu_m(g_m \odot z, u_m), sigma_m^2 I),
```

the local Fisher blocks are

```math
I_zz^m = J_z^T Sigma^{-1} J_z,
I_zu^m = J_z^T Sigma^{-1} J_u,
I_uu^m = J_u^T Sigma^{-1} J_u.
```

The effective information about `z` after adjusting out nuisance `u_m` is the Schur complement:

```math
I_m^core = I_zz^m - I_zu^m (I_uu^m + damping I)^{-1} I_uz^m.
```

This is the central quantity of the method.

### 3.3 Multimodal core information matrix

For observed modality subset `O`:

```math
K_O(x) = sum_{m in O} I_m^core(z_hat_x, u_hat_m).
```

Normalize it to prevent confidence-scale cheating:

```math
Kbar_O = r K_O / (Tr(K_O) + eps).
```

The core-rank constraint is

```math
c_rank = E_x log det(eps I + Kbar_O(x)) >= kappa.
```

This rewards full-rank, balanced eigenvalue coverage, not simple scale inflation.

### 3.4 Sparse footprint constraint

The learned soft gate matrix is

```math
g_mj = sigmoid(alpha_mj / tau).
```

The expected footprint complexity is

```math
Omega(G) = sum_{m,j} g_mj.
```

The constraint is

```math
Omega(G) <= s.
```

This encourages each modality to explain only the disease-core coordinates it actually observes.

---

## 4. Model architecture

### 4.1 Encoders

Each modality has an encoder:

```text
E_m(x_m) -> z_mu_m, z_logvar_m, u_mu_m, u_logvar_m.
```

The modality-specific `q_m(z | x_m)` distributions are fused by a Gaussian product of experts with a standard normal prior:

```math
Precision_O = I + sum_{m in O} diag(exp(-z_logvar_m)),
Mean_O = Precision_O^{-1} sum_{m in O} exp(-z_logvar_m) * z_mu_m.
```

This supports missing modalities naturally.

### 4.2 Decoders

Each modality has a decoder:

```text
D_m(g_m \odot z, u_m) -> x_m_mean.
```

The decoder likelihood is Gaussian with fixed variance in the first implementation. The decoder-induced Jacobian is used to compute Fisher; therefore the decoder is part of the theory, not merely a reconstruction module.

### 4.3 Prediction head

The classifier only reads the disease core:

```text
p(y | z, u_1, ..., u_M) = p(y | z).
```

This architecture-level restriction implements the conditional independence constraint. The nuisance variables are never given to the classifier.

---

## 5. Training objective

For one sample with observed modalities `O`:

```math
L_ELBO = E_q[log p(y | z) + sum_{m in O} log p(x_m | z, u_m)]
         - KL(q(z | x_O) || p(z))
         - sum_{m in O} KL(q(u_m | x_m) || p(u_m)).
```

The implemented minimization objective is:

```math
J = -L_ELBO
    + lambda_rank * relu(kappa - c_rank)
    + 0.5 * rho_rank * relu(kappa - c_rank)^2
    + lambda_sparse * relu(Omega(G) - s)
    + 0.5 * rho_sparse * relu(Omega(G) - s)^2
    + alpha_struct ||Z - A Z||^2
    + alpha_dag h(A)^2
    + alpha_A ||A||_1
    + alpha_ctx Corr(Z, C)^2.
```

The context-invariance term is optional and is only applied when known non-core context variables are available, such as synthetic bias variables, site/domain indicators, age, scanner, or visit timing. It encodes the claim that these variables may affect modalities but should not become disease-core coordinates.

Dual variables are updated during training:

```text
lambda_rank   <- max(0, lambda_rank + rho_rank * relu(kappa - c_rank))
lambda_sparse <- max(0, lambda_sparse + rho_sparse * relu(Omega(G) - s))
```

---

## 6. Synthetic benchmark design

### 6.1 Default scenario: complementary rank

Purpose: verify that multimodal learning improves when modalities cover complementary directions of `Z*`.

Default:

```text
r = 6 disease-core dimensions
M = 3 modalities
q = 3 private nuisance dimensions per modality
x_dim = 16 per modality
```

Ground-truth footprint:

```text
modality 0 observes Z0, Z1, Z2
modality 1 observes Z2, Z3, Z4
modality 2 observes Z1, Z4, Z5
```

Single modalities are rank deficient. All modalities together are full rank.

Expected behavior:

1. CoreRank full multimodal model should have higher latent recovery than each unimodal model.
2. Learned `K_O` should have larger effective rank for larger complementary modality subsets.
3. Prediction performance should correlate with `logdet(Kbar_O)` and effective rank.
4. Learned gates should approximately recover the footprint pattern.

### 6.2 Redundant modality scenario

Purpose: test that adding a modality with redundant Fisher directions gives little gain.

Modify footprint so modality 2 mostly duplicates modality 0:

```text
modality 2 observes Z0, Z1, Z2.
```

Expected behavior:

1. `logdet(K_{0,2}) - logdet(K_0)` should be small.
2. Prediction gain from adding modality 2 to modality 0 should be small.
3. Rank-gain should be predictive of downstream gain.

### 6.3 Case-I: concept/covariate shortcut shift

Purpose: test robustness when one modality contains a strong non-core shortcut correlated with the label in training but shifted at test time.

This is the synthetic analogue of the PPT case where age affects a recognition-like measurement, but using the age-related path as an Alzheimer classifier shortcut is unstable when the age distribution differs between train and test.

Generation:

```math
B_m = rho_split * (2Y - 1) + sqrt(1 - rho_split^2) * eps.
```

Train uses positive correlation, e.g. `rho_train = 0.9`.
OOD test uses negative or zero correlation, e.g. `rho_test = -0.6` or `0.0`.
The implementation applies this explicit spurious shift to one configurable modality by default, while keeping the other modalities governed by their core and private nuisance terms.

Expected behavior:

1. ERM fusion should overfit the biased modality.
2. CoreRank should be more robust if the nuisance pathway `u_m` absorbs the bias and the classifier is restricted to `z`.
3. Measure bias leakage by predicting the known bias scalar from learned `z_hat`. Lower is better.

### 6.4 Case-II: domain/mechanism shift

Purpose: test robustness when a modality's observation mechanism changes across domains, even though the disease-label mechanism remains stable.

Examples:

```text
site/scanner/protocol -> MRI feature distribution
visit schedule        -> measurement availability / measurement noise
hospital population   -> lab calibration or coding practice
```

Generation:

```math
X_m = f_m(G^*_m \odot Z^*, U_m, eps_m) + gamma_domain D_m v_m.
```

Train and validation have one domain value; OOD test shifts `D_m`. This should be absorbed by nuisance/domain-specific variation rather than becoming disease-core evidence.

Expected behavior:

1. ERM may lose performance if it uses domain-specific measurement artifacts.
2. CoreRank should be less sensitive when the shifted direction is nuisance-adjusted.
3. Oracle Fisher diagnostics should still identify which observed modalities carry recoverable core directions.

### 6.5 Case-III: missing information / incomplete core coverage

Purpose: test that posterior uncertainty and rank score predict degradation under missing modalities.

Evaluation:

```text
For every nonempty subset O of modalities:
  compute AUROC/accuracy
  compute latent R2 and MCC
  compute logdet(Kbar_O)
  compute effective rank
  compute trace posterior covariance
```

Expected behavior:

```text
Higher logdet/effective-rank should correlate with higher latent recovery and better prediction.
```

This case is not just random missingness. It asks whether the observed subset `O` contains enough nuisance-adjusted evidence to recover the disease-core block needed for prediction.

---

## 7. Metrics

### 7.1 Prediction

- Accuracy
- AUROC when sklearn is available
- Binary cross-entropy

### 7.2 Core recovery

Because the theory only guarantees block recovery up to invertible transformation, use two levels of metrics:

1. **Linear R2** from learned `z_hat` to true `Z*` via ridge regression.
2. **MCC**: mean absolute correlation after Hungarian matching. This is stricter and approximates component-wise recovery.

### 7.3 Footprint recovery

Threshold learned gates:

```text
g_hat_mj = 1[sigmoid(alpha_mj) > threshold].
```

Compute precision, recall, F1 against `G*`.

### 7.4 Fisher/rank diagnostics

For each modality subset:

- normalized logdet score,
- effective rank,
- eigenvalue spectrum,
- minimum eigenvalue,
- trace inverse when numerically stable.

The normalized logdet score must not add isotropic jitter before trace normalization. Otherwise a zero-information matrix can be converted into an artificial identity matrix and falsely appear full rank. Numerical stabilization is applied after preserving the zero-information case.

### 7.5 Bias leakage

Fit a ridge/logistic probe from `z_hat` to known nuisance bias variables. Lower R2/AUC means less leakage.

---

## 8. Baselines

### 8.1 ERM early fusion

Concatenate available modalities and train a classifier.

This tests whether ordinary discriminative fusion overfits spurious modalities.
ERM is a discriminative baseline: it directly learns `p(y | x_1, ..., x_M)` or a decision boundary from observed features to labels. It does not model `Z*`, modality-private nuisance, domain variables, or missing-modality uncertainty.

### 8.2 CoreRank without rank constraint

Set `rank_weight = 0` or `kappa <= -inf`. This is a PoE multimodal VAE with the same architecture but no core-rank constraint.

### 8.3 CoreRank without sparse footprint

Set sparse budget very large. This tests whether the gates matter for footprint recovery and component recovery.

### 8.4 Unimodal CoreRank

Evaluate the same model using one modality at a time. This gives the modality subset curve.

---

## 9. What the first code drop implements

The current code implements:

1. synthetic data generation for complementary, redundant, biased, and domain-shift scenarios;
2. CoreRankVAE with PoE posterior, private nuisance variables, soft footprint gates;
3. decoder-induced nuisance-adjusted Fisher via autograd Jacobians;
4. normalized-logdet core-rank constraint;
5. footprint sparsity constraint;
6. augmented Lagrangian training;
7. missing-modality evaluation over all modality subsets;
8. ERM baseline;
9. true-generator Fisher/rank diagnostics for the synthetic benchmark;
10. bias/domain leakage probes from learned `z_hat`;
11. temperature annealing and binary pressure options for soft footprint gates;
12. optional core structural graph learning with acyclicity and sparsity penalties;
13. optional context-invariance penalties for known non-core variables;
14. metrics and CSV/JSON output.

The first code drop does **not** implement:

1. proving global identifiability of the internal disease-core DAG `A`;
2. real medical datasets;
3. joint Fisher for shared nuisance;
4. hard-concrete gates;
5. image/text feature extraction.

These are future extensions after the synthetic benchmark validates the theory.

---

## 10. Suggested synthetic experiment grid

Run at least 5 seeds per cell.

```text
Scenario grid:
  complementary
  redundant
  biased
  domain

Sample sizes:
  1000, 3000, 10000

Noise std:
  0.2, 0.5, 1.0

Bias strength:
  0.0, 1.0, 2.0

Domain shift strength:
  0.0, 1.0, 1.5, 2.0

Training variants:
  corerank_full
  corerank_no_rank
  corerank_no_sparse
  erm_fusion
```

Primary plots:

1. `logdet(Kbar_O)` vs latent R2 across modality subsets.
2. `logdet(Kbar_O)` vs AUROC across modality subsets.
3. test-OOD AUROC under bias shift.
4. test-OOD AUROC under domain/mechanism shift.
5. learned gate heatmap vs true footprint.
6. learned-decoder rank vs true-generator rank.
7. eigenvalue spectra for single modalities and all modalities.

---

## 11. Server execution

Install:

```bash
pip install -r requirements.txt
```

Run one smoke test:

```bash
python -m corerank_synth.run_experiment \
  --scenario complementary \
  --n-train 512 --n-val 256 --n-test 256 \
  --epochs 3 --batch-size 128 \
  --z-dim 6 --u-dim 3 --x-dim 16 \
  --output-dir outputs/smoke
```

Run a serious experiment:

```bash
bash scripts/run_main_synthetic.sh
```

Outputs:

```text
outputs/<run_name>/config.json
outputs/<run_name>/metrics.json
outputs/<run_name>/subset_metrics.csv
outputs/<run_name>/gates.npy
outputs/<run_name>/true_footprint.npy
outputs/<run_name>/model.pt
```

---

## 12. Acceptance criteria for the synthetic benchmark

The implementation is scientifically useful if it demonstrates the following:

1. In the complementary scenario, all modalities produce full effective rank and better latent recovery than unimodal subsets.
2. In the redundant scenario, adding redundant modalities gives small rank gain and small prediction gain.
3. In the biased scenario, CoreRank has lower OOD degradation and lower bias leakage than ERM fusion.
4. In the domain scenario, CoreRank is less sensitive to the shifted measurement mechanism and has lower domain leakage than ERM-like representations.
5. Learned gates recover a sparse approximation of the ground-truth footprint after annealing or a harder gate relaxation.
6. Learned-decoder rank agrees directionally with true-generator rank; if it saturates, report true rank separately and treat learned rank as a model diagnostic, not ground truth.
7. Removing the rank constraint reduces the alignment between `logdet(Kbar)` and latent recovery.
8. Removing the sparse footprint constraint weakens footprint recovery and component-level interpretability.
