# MIMIC-IV → eICU five-modality transfer experiment

Run date: 2026-08-22

## Protocol

- Task used exactly as supplied by the existing CausalMedical/MUSE preprocessing: binary `mortality`.
- Train/model selection: MIMIC-IV train (115,306) and validation (16,472).
- Source test: MIMIC-IV test (32,945).
- Zero-shot target test: eICU test (15,825); no eICU labels were used for training, normalization, epoch selection, or threshold selection.
- Five shared modalities: demographics, diagnosis, procedure/treatment, medication, and labs.
- Dataset-specific modalities excluded: MIMIC-IV discharge notes and eICU APACHE APS.
- Diagnosis/procedure/medication codes were pooled from the existing 768-D Bio_ClinicalBERT embeddings. MIMIC-IV train statistics alone were used for normalization.
- The decision threshold was selected by Youden J on MIMIC-IV validation and transferred unchanged to both test sets.
- Five paired seeds: 2026–2030.

## Results

Values are mean ± sample SD over five paired seeds.

| Model | MIMIC AUROC | MIMIC AUPRC | MIMIC balanced accuracy | eICU AUROC | eICU AUPRC | eICU balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| SPMNet | 0.8603 ± 0.0015 | 0.4456 ± 0.0061 | 0.7764 ± 0.0039 | 0.6342 ± 0.0192 | 0.0637 ± 0.0061 | 0.5971 ± 0.0149 |
| Transformer | 0.8598 ± 0.0017 | 0.4471 ± 0.0030 | 0.7786 ± 0.0030 | 0.6702 ± 0.0161 | 0.0768 ± 0.0081 | 0.5972 ± 0.0128 |

Paired SPMNet − Transformer differences:

- MIMIC AUROC: +0.0004 (paired t-test p=0.765); no evidence of a difference.
- eICU AUROC: −0.0360 (95% t interval half-width 0.0152; paired t-test p=0.0028). Transformer was higher in all five paired seeds; the two-sided exact sign test is p=0.0625 because n=5.
- eICU AUPRC: −0.0131 (paired t-test p=0.0086). Transformer was again higher in all five seeds.

Under this as-processed protocol, SPMNet does not outperform the Transformer on eICU transfer.

## Required interpretation limits

1. The targets are not clinically identical. MIMIC-IV labels death within 90 days after a live hospital discharge and excludes in-hospital deaths; eICU labels death by ICU discharge. This is representation transfer across a target-definition shift, not strict external validation of one endpoint.
2. The supplied MIMIC split is admission-random, not patient-disjoint. Of 23,505 unique MIMIC test patients, 15,402 also occur in train. The internal MIMIC result may therefore be optimistic.
3. Target prevalence differs substantially: MIMIC test 9.28% versus eICU test 3.99%.
4. Source-selected probabilities are poorly calibrated on eICU (ECE about 0.48–0.49); AUROC/AUPRC should be treated as the primary cross-dataset metrics.
5. Labs use a dataset-invariant statistical summary because the stored MIMIC and eICU lab vectors have different feature identities (57 value channels versus 158 channels). The experiment does not claim item-level lab harmonization.

## Artifacts

- Aggregate JSON/CSV: `outputs/icu_cross_mortality_v1/summary.json` and `summary.csv`.
- Cache/data audit: `outputs/icu_cross_mortality_v1/cache/cache_summary.json`.
- Per-seed metrics/configs: `outputs/icu_cross_mortality_v1/{spmnet,transformer}/seed_*/`.
- Full farm artifacts, predictions, checkpoints, and logs: `/group/datalabgrp/xgzhu/CoreRank-Robust-Multimodal-Learning/outputs/icu_cross_mortality_v1/`.
