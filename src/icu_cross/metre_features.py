from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


METRE_MODALITIES = ("demographics", "vitals", "labs", "medications", "interventions")
SUMMARY_NAMES = ("mean", "std", "min", "max", "last", "log_count", "slope")
EVENT_NAMES = ("present", "log_count", "first_hour", "last_hour")

VITAL_CONCEPTS = ("heart_rate", "resp_rate", "spo2", "temperature", "sbp", "dbp", "map")
LAB_CONCEPTS = (
    "albumin", "alkaline_phosphatase", "alt", "ast", "base_excess", "bicarbonate",
    "bilirubin_total", "bilirubin_direct", "bun", "calcium", "calcium_ionized",
    "chloride", "creatinine", "glucose", "hematocrit", "hemoglobin", "inr",
    "lactate", "magnesium", "phosphate", "platelets", "potassium", "ptt", "sodium",
    "wbc", "ph", "pao2", "paco2",
)
MEDICATION_CONCEPTS = (
    "antibiotic", "anticoagulant", "vasopressor", "sedative",
    "opioid", "insulin", "diuretic", "corticosteroid",
)
INTERVENTION_CONCEPTS = (
    "mechanical_ventilation", "dialysis_crrt", "rbc_transfusion", "platelet_transfusion",
    "plasma_transfusion", "crystalloid_bolus", "colloid_bolus", "other_procedure",
)

MIMIC_VITAL_ITEM_MAP = {
    220045: "heart_rate",
    220210: "resp_rate", 224690: "resp_rate",
    220277: "spo2",
    223761: "temperature", 223762: "temperature",
    220050: "sbp", 220179: "sbp",
    220051: "dbp", 220180: "dbp",
    220052: "map", 220181: "map",
}
MIMIC_LAB_ITEM_MAP = {
    50862: "albumin", 50863: "alkaline_phosphatase", 50861: "alt", 50878: "ast",
    50802: "base_excess", 50882: "bicarbonate", 50885: "bilirubin_total",
    50883: "bilirubin_direct", 51006: "bun", 50893: "calcium", 50808: "calcium_ionized",
    50902: "chloride", 50806: "chloride", 50912: "creatinine", 50931: "glucose",
    50809: "glucose", 51221: "hematocrit", 50810: "hematocrit", 51222: "hemoglobin",
    50811: "hemoglobin", 51237: "inr", 50813: "lactate", 50960: "magnesium",
    50970: "phosphate", 51265: "platelets", 50971: "potassium", 50822: "potassium",
    51275: "ptt", 50983: "sodium", 50824: "sodium", 51300: "wbc", 51301: "wbc",
    50820: "ph", 50821: "pao2", 50818: "paco2",
}
EICU_LAB_NAME_MAP = {
    "albumin": "albumin", "alkaline phos.": "alkaline_phosphatase",
    "alt (sgpt)": "alt", "ast (sgot)": "ast", "base excess": "base_excess",
    "bicarbonate": "bicarbonate", "hco3": "bicarbonate", "total bilirubin": "bilirubin_total",
    "direct bilirubin": "bilirubin_direct", "bun": "bun", "calcium": "calcium",
    "ionized calcium": "calcium_ionized", "chloride": "chloride", "creatinine": "creatinine",
    "glucose": "glucose", "bedside glucose": "glucose", "hct": "hematocrit",
    "hgb": "hemoglobin", "pt - inr": "inr", "lactate": "lactate",
    "magnesium": "magnesium", "phosphate": "phosphate", "platelets x 1000": "platelets",
    "potassium": "potassium", "ptt": "ptt", "sodium": "sodium", "wbc x 1000": "wbc",
    "ph": "ph", "pao2": "pao2", "paco2": "paco2",
}

MEDICATION_PATTERNS: Mapping[str, tuple[str, ...]] = {
    "antibiotic": ("cillin", "cef", "mycin", "floxacin", "meropenem", "vancomycin", "aztreonam", "doxycycline"),
    "anticoagulant": ("heparin", "enoxaparin", "warfarin", "apixaban", "rivaroxaban"),
    "vasopressor": ("norepinephrine", "epinephrine", "dopamine", "vasopressin", "phenylephrine"),
    "sedative": ("propofol", "midazolam", "dexmedetomidine", "lorazepam"),
    "opioid": ("fentanyl", "morphine", "hydromorphone", "oxycodone"),
    "insulin": ("insulin",),
    "diuretic": ("furosemide", "bumetanide", "torsemide"),
    "corticosteroid": ("hydrocortisone", "methylprednisolone", "dexamethasone", "prednisone"),
}
INTERVENTION_PATTERNS: Mapping[str, tuple[str, ...]] = {
    "mechanical_ventilation": ("ventilat", "intubat", "endotracheal"),
    "dialysis_crrt": ("dialysis", "crrt", "renal replacement"),
    "rbc_transfusion": ("red blood", "packed cell", "rbc"),
    "platelet_transfusion": ("platelet",),
    "plasma_transfusion": ("fresh frozen", "plasma", "ffp"),
    "crystalloid_bolus": ("crystalloid", "normal saline", "lactated ringer", "fluid bolus"),
    "colloid_bolus": ("albumin", "colloid"),
}

# Broad physiologic plausibility bounds, applied identically in both databases.
# These remove unit/coding errors without encoding source-specific information.
VALUE_RANGES: Mapping[str, tuple[float, float]] = {
    "heart_rate": (0.0, 390.0), "resp_rate": (0.0, 330.0), "spo2": (0.0, 100.0),
    "temperature": (20.0, 47.0), "sbp": (0.0, 375.0), "dbp": (0.0, 375.0),
    "map": (0.0, 375.0), "albumin": (0.0, 10.0), "alkaline_phosphatase": (0.0, 2500.0),
    "alt": (0.0, 10000.0), "ast": (0.0, 10000.0), "base_excess": (-50.0, 50.0),
    "bicarbonate": (0.0, 80.0), "bilirubin_total": (0.0, 80.0),
    "bilirubin_direct": (0.0, 80.0), "bun": (0.0, 300.0), "calcium": (0.0, 30.0),
    "calcium_ionized": (0.0, 10.0), "chloride": (0.0, 200.0), "creatinine": (0.0, 80.0),
    "glucose": (0.0, 3000.0), "hematocrit": (0.0, 100.0), "hemoglobin": (0.0, 30.0),
    "inr": (0.0, 30.0), "lactate": (0.0, 50.0), "magnesium": (0.0, 20.0),
    "phosphate": (0.0, 30.0), "platelets": (0.0, 3000.0), "potassium": (0.0, 20.0),
    "ptt": (0.0, 300.0), "sodium": (0.0, 250.0), "wbc": (0.0, 1000.0),
    "ph": (6.0, 8.0), "pao2": (0.0, 800.0), "paco2": (0.0, 300.0),
}


@dataclass
class ContinuousAccumulator:
    n_stays: int
    concepts: Sequence[str]

    def __post_init__(self) -> None:
        shape = (self.n_stays, len(self.concepts))
        self.index = {name: i for i, name in enumerate(self.concepts)}
        self.count = np.zeros(shape, dtype=np.int32)
        self.total = np.zeros(shape, dtype=np.float64)
        self.total_sq = np.zeros(shape, dtype=np.float64)
        self.total_t = np.zeros(shape, dtype=np.float64)
        self.total_t2 = np.zeros(shape, dtype=np.float64)
        self.total_tx = np.zeros(shape, dtype=np.float64)
        self.minimum = np.full(shape, np.inf, dtype=np.float64)
        self.maximum = np.full(shape, -np.inf, dtype=np.float64)
        self.last = np.zeros(shape, dtype=np.float64)
        self.last_time = np.full(shape, -np.inf, dtype=np.float64)

    def update(self, stays: np.ndarray, concepts: Iterable[str], times: np.ndarray, values: np.ndarray) -> None:
        concept_array = np.asarray(list(concepts), dtype=object)
        concept_ids = np.fromiter((self.index.get(str(value), -1) for value in concept_array), dtype=np.int64)
        valid = (stays >= 0) & (concept_ids >= 0) & np.isfinite(times) & np.isfinite(values)
        for name, (lower, upper) in VALUE_RANGES.items():
            selected = concept_array == name
            valid[selected] &= (values[selected] >= lower) & (values[selected] <= upper)
        if not valid.any():
            return
        stays, concept_ids, times, values = stays[valid], concept_ids[valid], times[valid], values[valid]
        flat = stays * len(self.concepts) + concept_ids
        np.add.at(self.count.ravel(), flat, 1)
        np.add.at(self.total.ravel(), flat, values)
        np.add.at(self.total_sq.ravel(), flat, values * values)
        np.add.at(self.total_t.ravel(), flat, times)
        np.add.at(self.total_t2.ravel(), flat, times * times)
        np.add.at(self.total_tx.ravel(), flat, times * values)
        np.minimum.at(self.minimum.ravel(), flat, values)
        np.maximum.at(self.maximum.ravel(), flat, values)
        order = np.lexsort((times, flat))
        ordered_flat = flat[order]
        take = np.r_[ordered_flat[1:] != ordered_flat[:-1], True]
        chosen = order[take]
        for position in chosen:
            row, column = int(stays[position]), int(concept_ids[position])
            if times[position] >= self.last_time[row, column]:
                self.last_time[row, column] = times[position]
                self.last[row, column] = values[position]

    def matrix(self) -> tuple[np.ndarray, np.ndarray]:
        n = self.count.astype(np.float64)
        observed = n > 0
        safe = np.maximum(n, 1.0)
        mean = self.total / safe
        variance = np.maximum(self.total_sq / safe - mean * mean, 0.0)
        denominator = n * self.total_t2 - self.total_t * self.total_t
        slope = np.divide(
            n * self.total_tx - self.total_t * self.total,
            denominator,
            out=np.zeros_like(denominator),
            where=np.abs(denominator) > 1e-8,
        )
        minimum = np.where(observed, self.minimum, 0.0)
        maximum = np.where(observed, self.maximum, 0.0)
        last = np.where(observed, self.last, 0.0)
        values = np.stack((mean, np.sqrt(variance), minimum, maximum, last, np.log1p(n), slope), axis=2)
        return values.reshape(self.n_stays, -1).astype(np.float32), observed.any(axis=1).astype(np.uint8)


@dataclass
class EventAccumulator:
    n_stays: int
    concepts: Sequence[str]

    def __post_init__(self) -> None:
        shape = (self.n_stays, len(self.concepts))
        self.index = {name: i for i, name in enumerate(self.concepts)}
        self.count = np.zeros(shape, dtype=np.int32)
        self.first = np.full(shape, np.inf, dtype=np.float64)
        self.last = np.full(shape, -np.inf, dtype=np.float64)

    def update(self, stays: np.ndarray, concepts: Iterable[str], times: np.ndarray) -> None:
        concept_ids = np.fromiter((self.index.get(str(value), -1) for value in concepts), dtype=np.int64)
        valid = (stays >= 0) & (concept_ids >= 0) & np.isfinite(times)
        if not valid.any():
            return
        stays, concept_ids, times = stays[valid], concept_ids[valid], times[valid]
        flat = stays * len(self.concepts) + concept_ids
        np.add.at(self.count.ravel(), flat, 1)
        np.minimum.at(self.first.ravel(), flat, times)
        np.maximum.at(self.last.ravel(), flat, times)

    def matrix(self, observation_hours: float) -> tuple[np.ndarray, np.ndarray]:
        observed = self.count > 0
        values = np.stack(
            (
                observed.astype(np.float64),
                np.log1p(self.count),
                np.where(observed, self.first / observation_hours, 0.0),
                np.where(observed, self.last / observation_hours, 0.0),
            ),
            axis=2,
        )
        return values.reshape(self.n_stays, -1).astype(np.float32), observed.any(axis=1).astype(np.uint8)


def match_categories(texts: Sequence[object], patterns: Mapping[str, Sequence[str]]) -> list[str | None]:
    output: list[str | None] = []
    for raw in texts:
        text = str(raw or "").lower()
        match = next((name for name, needles in patterns.items() if any(needle in text for needle in needles)), None)
        output.append(match)
    return output
