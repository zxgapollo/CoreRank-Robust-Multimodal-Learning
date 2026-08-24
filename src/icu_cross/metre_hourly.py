from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


MODALITIES = ("static", "bedside", "laboratory", "medications", "procedures")
TEMPORAL_MODALITIES = ("bedside", "laboratory", "medications", "procedures")

# The 85 numeric concepts in the public METRE mimic_col_order.json.  Each is
# represented by an hourly value and an observation indicator.  Fourteen
# culture-site indicators and 16 intervention channels bring the temporal
# representation to the paper's 200 channels.
NUMERIC_CONCEPTS = (
    "so2", "po2", "pco2", "fio2", "ph", "baseexcess", "bicarbonate", "totalco2",
    "hematocrit", "hemoglobin", "chloride", "calcium", "temperature", "potassium",
    "sodium", "lactate", "glucose", "heart_rate", "sbp", "dbp", "mbp", "sbp_ni",
    "dbp_ni", "mbp_ni", "resp_rate", "wbc", "basophils", "eosinophils",
    "lymphocytes", "monocytes", "neutrophils", "atypical_lymphocytes", "bands",
    "immature_granulocytes", "metamyelocytes", "nrbc", "troponin_t", "ck_mb",
    "ntprobnp", "albumin", "total_protein", "aniongap", "bun", "calcium_chem",
    "creatinine", "fibrinogen", "inr", "pt", "ptt", "mch", "mchc", "mcv",
    "platelet", "rbc", "rdw", "screen", "positive_culture", "has_sensitivity",
    "alt", "alp", "ast", "amylase", "bilirubin_total", "bilirubin_direct",
    "bilirubin_indirect", "ck_cpk", "ggt", "ld_ldh", "gcs", "crp", "weight",
    "uo", "Central Venous Pressure", "Creatinine urine", "Magnesium",
    "Peak inspiratory pressure", "Phosphate", "Plateau Pressure",
    "Positive end-expiratory pressure", "Positive end-expiratory pressure Set",
    "Red blood cell count urine", "Tidal Volume Observed", "Total Protein Urine",
    "White blood cell count urine", "pH urine",
)

BEDSIDES = (
    "so2", "fio2", "temperature", "heart_rate", "sbp", "dbp", "mbp", "sbp_ni",
    "dbp_ni", "mbp_ni", "resp_rate", "gcs", "weight", "uo",
    "Central Venous Pressure", "Peak inspiratory pressure", "Plateau Pressure",
    "Positive end-expiratory pressure", "Positive end-expiratory pressure Set",
    "Tidal Volume Observed",
)
LABS = tuple(name for name in NUMERIC_CONCEPTS if name not in BEDSIDES)
CULTURE_SITES = tuple(f"culture_site_{index}" for index in range(14))
MEDICATIONS = (
    "antib", "dopamine", "epinephrine", "norepinephrine", "phenylephrine",
    "vasopressin", "dobutamine", "milrinone", "heparin",
)
PROCEDURES = ("vent", "crrt", "rbc", "platelets", "ffp", "colloid", "crystalloid")
INTERVENTIONS = MEDICATIONS + PROCEDURES

COMORBIDITIES = (
    "myocardial_infarct", "congestive_heart_failure", "peripheral_vascular_disease",
    "cerebrovascular_disease", "dementia", "chronic_pulmonary_disease",
    "rheumatic_disease", "peptic_ulcer_disease", "mild_liver_disease",
    "diabetes_without_cc", "diabetes_with_cc", "paraplegia", "renal_disease",
    "malignant_cancer", "severe_liver_disease", "metastatic_solid_tumor", "aids",
)
STATIC_FEATURES = (
    "age_scaled", "female", "male", "gender_unknown", "white", "black", "asian",
    "hispanic", "native", "ethnicity_other", "ethnicity_unknown", *COMORBIDITIES,
    "admission_emergency", "admission_elective", "admission_transfer", "admission_other",
)

# Official METRE outlier limits (MIMIC definitions).  Variables absent from the
# files had no published bound and are only checked for finiteness.
LOW: Mapping[str, float] = {
    "ph": 6.3, "baseexcess": -100.0, "temperature": 14.2, "lactate": 0.01,
    "uo": 0.0, "pH urine": 3.0, "basophils": 0.0, "eosinophils": 0.0,
    "lymphocytes": 0.0, "monocytes": 0.0, "neutrophils": 0.0, "bands": 0.0,
}
HIGH: Mapping[str, float] = {
    "so2": 100, "po2": 770, "pco2": 220, "ph": 10, "baseexcess": 100,
    "bicarbonate": 66, "totalco2": 80, "chloride": 200, "hemoglobin": 30,
    "hematocrit": 100, "calcium": 1.87, "temperature": 47, "potassium": 15,
    "sodium": 250, "lactate": 33, "glucose": 2200, "heart_rate": 390,
    "sbp": 375, "sbp_ni": 375, "mbp": 375, "mbp_ni": 375, "dbp": 375,
    "dbp_ni": 375, "resp_rate": 330, "wbc": 1100, "basophils": 8,
    "atypical_lymphocytes": 17, "nrbc": 143, "troponin_t": 24, "ck_mb": 700,
    "albumin": 60, "total_protein": 20, "aniongap": 55, "bun": 300,
    "calcium_chem": 28, "creatinine": 66, "fibrinogen": 1700, "inr": 15,
    "pt": 150, "ptt": 500, "mch": 46, "mchc": 43, "mcv": 140,
    "platelet": 2200, "rbc": 8, "rdw": 38, "alt": 11000, "ast": 22000,
    "alp": 4000, "amylase": 2800, "bilirubin_total": 66,
    "bilirubin_indirect": 66, "bilirubin_direct": 66, "ck_cpk": 10000,
    "ggt": 10000, "ld_ldh": 35000, "crp": 4000, "weight": 550, "uo": 2445,
    "Central Venous Pressure": 400, "Creatinine urine": 650, "Magnesium": 22,
    "Peak inspiratory pressure": 40, "Phosphate": 22, "Plateau Pressure": 61,
    "Positive end-expiratory pressure": 30, "Tidal Volume Observed": 2000,
    "Total Protein Urine": 7500, "White blood cell count urine": 750, "pH urine": 10,
}


@dataclass
class HourlyAccumulator:
    n_stays: int
    concepts: Sequence[str]
    hours: int = 48

    def __post_init__(self) -> None:
        self.index = {name: index for index, name in enumerate(self.concepts)}
        shape = (self.n_stays, self.hours, len(self.concepts))
        self.total = np.zeros(shape, dtype=np.float32)
        self.count = np.zeros(shape, dtype=np.uint16)

    def update(
        self,
        stays: np.ndarray,
        concepts: Iterable[str],
        times: np.ndarray,
        values: np.ndarray,
    ) -> None:
        names = np.asarray(list(concepts), dtype=object)
        concept_ids = np.fromiter((self.index.get(str(name), -1) for name in names), dtype=np.int64)
        hour_ids = np.floor(np.asarray(times, dtype=np.float64)).astype(np.int64, casting="unsafe")
        values = np.asarray(values, dtype=np.float64)
        stays = np.asarray(stays, dtype=np.int64)
        valid = (
            (stays >= 0) & (stays < self.n_stays) & (concept_ids >= 0)
            & np.isfinite(times) & np.isfinite(values) & (hour_ids >= 0) & (hour_ids < self.hours)
        )
        for name, lower in LOW.items():
            selected = names == name
            valid[selected] &= values[selected] >= lower
        for name, upper in HIGH.items():
            selected = names == name
            valid[selected] &= values[selected] <= upper
        if not valid.any():
            return
        stays, hour_ids, concept_ids, values = (
            stays[valid], hour_ids[valid], concept_ids[valid], values[valid]
        )
        flat = (stays * self.hours + hour_ids) * len(self.concepts) + concept_ids
        np.add.at(self.total.ravel(), flat, values.astype(np.float32))
        np.add.at(self.count.ravel(), flat, 1)

    def means_and_observed(self) -> tuple[np.ndarray, np.ndarray]:
        observed = self.count > 0
        means = np.divide(
            self.total, self.count, out=np.full_like(self.total, np.nan), where=observed,
        )
        return means, observed


@dataclass
class BinaryHourlyAccumulator:
    n_stays: int
    concepts: Sequence[str]
    hours: int = 48

    def __post_init__(self) -> None:
        self.index = {name: index for index, name in enumerate(self.concepts)}
        self.values = np.zeros((self.n_stays, self.hours, len(self.concepts)), dtype=np.uint8)

    def update(self, stays: np.ndarray, concepts: Iterable[str], times: np.ndarray) -> None:
        names = np.asarray(list(concepts), dtype=object)
        concept_ids = np.fromiter((self.index.get(str(name), -1) for name in names), dtype=np.int64)
        hour_ids = np.floor(np.asarray(times, dtype=np.float64)).astype(np.int64, casting="unsafe")
        stays = np.asarray(stays, dtype=np.int64)
        valid = (
            (stays >= 0) & (stays < self.n_stays) & (concept_ids >= 0)
            & np.isfinite(times) & (hour_ids >= 0) & (hour_ids < self.hours)
        )
        self.values[stays[valid], hour_ids[valid], concept_ids[valid]] = 1

    def update_intervals(
        self,
        stays: np.ndarray,
        concepts: Iterable[str],
        starts: np.ndarray,
        stops: np.ndarray,
    ) -> None:
        for stay, concept, start, stop in zip(stays, concepts, starts, stops):
            index = self.index.get(str(concept), -1)
            if stay < 0 or index < 0 or not np.isfinite(start):
                continue
            first = max(0, int(np.floor(start)))
            last = self.hours - 1 if not np.isfinite(stop) else min(self.hours - 1, int(np.floor(stop)))
            if first <= last and first < self.hours and last >= 0:
                self.values[int(stay), first:last + 1, index] = 1


def impute_and_normalize(
    means: np.ndarray,
    observed: np.ndarray,
    train_rows: np.ndarray,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Fit source-train statistics, z-score, forward fill, stay-mean fill, then zero."""
    feature_count = means.shape[-1]
    center = np.zeros(feature_count, dtype=np.float32)
    scale = np.ones(feature_count, dtype=np.float32)
    for feature in range(feature_count):
        selected = means[train_rows, :, feature][observed[train_rows, :, feature]]
        if selected.size:
            center[feature] = float(np.mean(selected, dtype=np.float64))
            std = float(np.std(selected, dtype=np.float64))
            scale[feature] = std if std > 1e-6 else 1.0
    normalized = (means - center[None, None, :]) / scale[None, None, :]
    # METRE-style causal forward fill; no information is propagated backwards.
    for hour in range(1, normalized.shape[1]):
        missing = ~observed[:, hour, :]
        normalized[:, hour, :][missing] = normalized[:, hour - 1, :][missing]
    # Initial gaps use the within-window patient mean, then the source mean (=0).
    finite = np.isfinite(normalized)
    stay_mean = np.divide(
        np.nansum(normalized, axis=1), finite.sum(axis=1),
        out=np.full((normalized.shape[0], normalized.shape[2]), np.nan, dtype=np.float32),
        where=finite.sum(axis=1) > 0,
    )
    missing = ~np.isfinite(normalized)
    replacement = np.broadcast_to(stay_mean[:, None, :], normalized.shape)
    normalized[missing] = replacement[missing]
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return normalized, {"mean": center.tolist(), "std": scale.tolist()}


def paired_channels(values: np.ndarray, observed: np.ndarray) -> np.ndarray:
    return np.stack((values, observed.astype(np.float32)), axis=-1).reshape(
        values.shape[0], values.shape[1], -1
    )


def validate_schema() -> None:
    assert len(NUMERIC_CONCEPTS) == 85
    assert len(BEDSIDES) == 20
    assert len(LABS) == 65
    assert len(CULTURE_SITES) == 14
    assert len(MEDICATIONS) == 9
    assert len(PROCEDURES) == 7
    assert 2 * len(BEDSIDES) + 2 * len(LABS) + len(CULTURE_SITES) + len(INTERVENTIONS) == 200


validate_schema()
