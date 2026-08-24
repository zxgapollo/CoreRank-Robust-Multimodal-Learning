from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .build_metre_cache import _limit_cohort, _patient_disjoint_split
from .features import broad_ethnicity
from .metre_hourly import (
    BEDSIDES,
    COMORBIDITIES,
    CULTURE_SITES,
    INTERVENTIONS,
    LABS,
    MEDICATIONS,
    MODALITIES,
    NUMERIC_CONCEPTS,
    PROCEDURES,
    STATIC_FEATURES,
    TEMPORAL_MODALITIES,
    BinaryHourlyAccumulator,
    HourlyAccumulator,
    impute_and_normalize,
    paired_channels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a five-modality hourly METRE-compatible cache.")
    parser.add_argument("--mimic-root", required=True)
    parser.add_argument("--eicu-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--observation-hours", type=int, default=48)
    parser.add_argument("--gap-hours", type=float, default=6.0)
    parser.add_argument("--max-los-hours", type=float, default=240.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--max-stays", type=int)
    return parser.parse_args()


def _chunks(path: Path, columns: Sequence[str], size: int) -> Iterable[pd.DataFrame]:
    yield from pd.read_csv(path, usecols=list(columns), chunksize=size, low_memory=False)


def _mimic_cohort(root: Path, observation: float, gap: float, max_los: float) -> pd.DataFrame:
    icu = pd.read_csv(
        root / "icu/icustays.csv",
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
    )
    admissions = pd.read_csv(
        root / "hosp/admissions.csv",
        usecols=["subject_id", "hadm_id", "admittime", "admission_type", "admission_location", "race", "hospital_expire_flag"],
    )
    patients = pd.read_csv(
        root / "hosp/patients.csv", usecols=["subject_id", "gender", "anchor_age", "anchor_year"]
    )
    icu["intime"] = pd.to_datetime(icu.intime)
    icu["outtime"] = pd.to_datetime(icu.outtime)
    icu = icu.sort_values(["subject_id", "intime", "stay_id"])
    # mimic_derived.icustay_detail calculates hospstay_seq after joining ICU
    # stays to admissions.  Therefore this is the first ICU-associated hospital
    # admission, not necessarily the patient's first admission of any kind.
    first_icu = icu.groupby("subject_id", sort=False).head(1)
    cohort = first_icu.merge(admissions, on=["subject_id", "hadm_id"], validate="one_to_one").merge(
        patients, on="subject_id", validate="many_to_one"
    )
    cohort["age"] = cohort.anchor_age + cohort.intime.dt.year - cohort.anchor_year
    cohort["los_hours"] = cohort.los * 24.0
    cohort = cohort[
        (cohort.age >= 18)
        & cohort.los_hours.between(observation + gap, max_los, inclusive="both")
        & cohort.hospital_expire_flag.isin([0, 1])
    ].copy()
    cohort["dataset"] = "mimic4"
    cohort["patient_id"] = cohort.subject_id.astype(str)
    cohort["stay_key"] = cohort.stay_id.astype(str)
    cohort["label"] = cohort.hospital_expire_flag.astype(np.uint8)
    cohort["ethnicity"] = cohort.race
    cohort["admission_source"] = cohort.admission_type.fillna("").astype(str) + " " + cohort.admission_location.fillna("").astype(str)
    return cohort.reset_index(drop=True)


def _eicu_cohort(root: Path, observation: float, gap: float, max_los: float) -> pd.DataFrame:
    columns = [
        "patientunitstayid", "uniquepid", "gender", "age", "ethnicity", "admissionweight",
        "hospitaladmitsource", "unitadmitsource", "unitdischargeoffset", "hospitaldischargestatus",
    ]
    cohort = pd.read_csv(root / "patient.csv", usecols=columns)
    cohort["age_num"] = pd.to_numeric(cohort.age.replace({"> 89": "90"}), errors="coerce")
    cohort["los_hours"] = pd.to_numeric(cohort.unitdischargeoffset, errors="coerce") / 60.0
    cohort = cohort[
        (cohort.age_num >= 18)
        & cohort.los_hours.between(observation + gap, max_los, inclusive="both")
        & cohort.hospitaldischargestatus.isin(["Alive", "Expired"])
    ].copy()
    cohort["dataset"] = "eicu"
    cohort["patient_id"] = cohort.uniquepid.astype(str)
    cohort["stay_key"] = cohort.patientunitstayid.astype(str)
    cohort["label"] = (cohort.hospitaldischargestatus == "Expired").astype(np.uint8)
    cohort["age"] = cohort.age_num
    cohort["admission_source"] = cohort.hospitaladmitsource.fillna("").astype(str) + " " + cohort.unitadmitsource.fillna("").astype(str)
    return cohort.reset_index(drop=True)


def _admission_group(value: object) -> str:
    text = str(value or "").lower()
    if any(word in text for word in ("emerg", "accident", "ed", "floor")):
        return "emergency"
    if any(word in text for word in ("elective", "scheduled", "operating room")):
        return "elective"
    if any(word in text for word in ("transfer", "other hospital", "other icu", "recovery room")):
        return "transfer"
    return "other"


def _code_flags(raw_codes: Iterable[object]) -> np.ndarray:
    flags = np.zeros(len(COMORBIDITIES), dtype=np.float32)
    for raw in raw_codes:
        code = str(raw or "").upper().replace(".", "").strip()
        if not code:
            continue
        def mark(name: str) -> None:
            flags[COMORBIDITIES.index(name)] = 1.0
        numeric = code[:3]
        if numeric in {"410", "412"} or code.startswith(("I21", "I22", "I252")): mark("myocardial_infarct")
        if numeric == "428" or code.startswith("I50"): mark("congestive_heart_failure")
        if numeric in {"440", "441", "443"} or code.startswith(("I70", "I71", "I73")): mark("peripheral_vascular_disease")
        if (numeric.isdigit() and 430 <= int(numeric) <= 438) or code.startswith(tuple(f"I{i}" for i in range(60, 70))): mark("cerebrovascular_disease")
        if numeric in {"290", "294", "331"} or code.startswith(("F00", "F01", "F02", "F03", "G30")): mark("dementia")
        if (numeric.isdigit() and 490 <= int(numeric) <= 505) or code.startswith(tuple(f"J{i}" for i in range(40, 48))): mark("chronic_pulmonary_disease")
        if numeric in {"710", "714", "725"} or code.startswith(("M05", "M06", "M32", "M34", "M35")): mark("rheumatic_disease")
        if numeric in {"531", "532", "533", "534"} or code.startswith(("K25", "K26", "K27", "K28")): mark("peptic_ulcer_disease")
        if numeric in {"570", "571"} or code.startswith(("K70", "K71", "K73", "K74", "K76")): mark("mild_liver_disease")
        if code.startswith(("2504", "2505", "2506", "2507", "E102", "E103", "E104", "E105", "E106", "E107", "E112", "E113", "E114", "E115", "E116", "E117")): mark("diabetes_with_cc")
        elif code.startswith(("250", "E10", "E11", "E12", "E13", "E14")): mark("diabetes_without_cc")
        if numeric in {"342", "343", "344"} or code.startswith(("G81", "G82")): mark("paraplegia")
        if numeric in {"582", "585", "586"} or code.startswith(("N18", "N19")): mark("renal_disease")
        if ((numeric.isdigit() and (140 <= int(numeric) <= 195 or 200 <= int(numeric) <= 208)) or code.startswith("C")): mark("malignant_cancer")
        if numeric in {"456", "572"} or code.startswith(("K72", "I85", "K766")): mark("severe_liver_disease")
        if numeric in {"196", "197", "198", "199"} or code.startswith(("C77", "C78", "C79", "C80")): mark("metastatic_solid_tumor")
        if numeric in {"042", "043", "044"} or code.startswith(("B20", "B21", "B22", "B23", "B24")): mark("aids")
    return flags


def _static_features(dataset: str, root: Path, cohort: pd.DataFrame, chunksize: int) -> np.ndarray:
    flags = np.zeros((len(cohort), len(COMORBIDITIES)), dtype=np.float32)
    if dataset == "mimic4":
        lookup: Mapping[int, list[int]] = {int(hadm): [index] for index, hadm in enumerate(cohort.hadm_id)}
        path, id_column, code_column = root / "hosp/diagnoses_icd.csv", "hadm_id", "icd_code"
    else:
        lookup = {int(stay): [index] for index, stay in enumerate(cohort.patientunitstayid)}
        path, id_column, code_column = root / "diagnosis.csv", "patientunitstayid", "icd9code"
    collected: dict[int, list[str]] = {}
    for chunk in _chunks(path, [id_column, code_column], chunksize):
        chunk = chunk[chunk[id_column].isin(lookup)]
        for identifier, group in chunk.groupby(id_column):
            collected.setdefault(int(identifier), []).extend(group[code_column].dropna().astype(str).tolist())
    for identifier, rows in lookup.items():
        for row in rows:
            flags[row] = _code_flags(collected.get(identifier, ()))
    output = np.zeros((len(cohort), len(STATIC_FEATURES)), dtype=np.float32)
    output[:, 0] = cohort.age.to_numpy(np.float32) / 100.0
    for row, record in enumerate(cohort.itertuples(index=False)):
        gender = str(getattr(record, "gender", "") or "").upper()
        gender_group = "female" if gender in {"F", "FEMALE"} else "male" if gender in {"M", "MALE"} else "unknown"
        ethnicity = broad_ethnicity(getattr(record, "ethnicity", ""))
        output[row, 1:4] = [gender_group == name for name in ("female", "male", "unknown")]
        output[row, 4:11] = [ethnicity == name for name in ("white", "black", "asian", "hispanic", "native", "other", "unknown")]
        output[row, 11:28] = flags[row]
        admission = _admission_group(getattr(record, "admission_source", ""))
        output[row, 28:32] = [admission == name for name in ("emergency", "elective", "transfer", "other")]
    return output


MIMIC_LAB_ALIASES: Mapping[str, str] = {
    "oxygen saturation": "so2", "po2": "po2", "pco2": "pco2", "fio2": "fio2", "ph": "ph",
    "base excess": "baseexcess", "calculated bicarbonate, whole blood": "bicarbonate", "bicarbonate": "bicarbonate",
    "total co2": "totalco2", "hematocrit": "hematocrit", "hemoglobin": "hemoglobin",
    "chloride": "chloride", "chloride, whole blood": "chloride", "free calcium": "calcium",
    "potassium": "potassium", "potassium, whole blood": "potassium", "sodium": "sodium",
    "sodium, whole blood": "sodium", "lactate": "lactate", "glucose": "glucose",
    "glucose, whole blood": "glucose", "white blood cells": "wbc", "basophils": "basophils",
    "eosinophils": "eosinophils", "lymphocytes": "lymphocytes", "monocytes": "monocytes",
    "neutrophils": "neutrophils", "atypical lymphocytes": "atypical_lymphocytes", "bands": "bands",
    "immature granulocytes": "immature_granulocytes", "metamyelocytes": "metamyelocytes",
    "nucleated red cells": "nrbc", "troponin t": "troponin_t", "ck-mb index": "ck_mb",
    "ntprobnp": "ntprobnp", "albumin": "albumin", "total protein": "total_protein",
    "anion gap": "aniongap", "urea nitrogen": "bun", "calcium, total": "calcium_chem",
    "creatinine": "creatinine", "fibrinogen, functional": "fibrinogen", "inr(pt)": "inr",
    "pt": "pt", "ptt": "ptt", "mch": "mch", "mchc": "mchc", "mcv": "mcv",
    "platelet count": "platelet", "red blood cells": "rbc", "rdw": "rdw",
    "alanine aminotransferase (alt)": "alt", "alkaline phosphatase": "alp",
    "asparate aminotransferase (ast)": "ast", "amylase": "amylase",
    "bilirubin, total": "bilirubin_total", "bilirubin, direct": "bilirubin_direct",
    "bilirubin, indirect": "bilirubin_indirect", "creatine kinase (ck)": "ck_cpk",
    "gamma glutamyltransferase": "ggt", "ldh": "ld_ldh", "c-reactive protein": "crp",
    "creatinine, urine": "Creatinine urine", "magnesium": "Magnesium", "phosphate": "Phosphate",
    "red blood cells, urine": "Red blood cell count urine", "protein, total, urine": "Total Protein Urine",
    "white blood cells, urine": "White blood cell count urine", "ph, urine": "pH urine",
}

MIMIC_CHART_ITEMS: Mapping[int, str] = {
    220227: "so2", 220277: "so2", 223835: "fio2", 223761: "temperature", 223762: "temperature",
    220045: "heart_rate", 220050: "sbp", 220051: "dbp", 220052: "mbp", 225312: "mbp",
    220179: "sbp_ni",
    220180: "dbp_ni", 220181: "mbp_ni", 220210: "resp_rate", 224690: "resp_rate",
    226512: "weight", 224639: "weight", 220074: "Central Venous Pressure",
    224695: "Peak inspiratory pressure", 224696: "Plateau Pressure",
    220339: "Positive end-expiratory pressure", 224700: "Positive end-expiratory pressure Set",
    224685: "Tidal Volume Observed", 226755: "gcs", 227013: "gcs",
    225664: "glucose", 220621: "glucose", 226537: "glucose",
}

MIMIC_GCS_COMPONENTS: Mapping[int, str] = {
    220739: "eyes", 223900: "verbal", 223901: "motor",
}

EICU_LAB_ALIASES: Mapping[str, str] = {
    "spo2": "so2", "pao2": "po2", "paco2": "pco2", "fio2": "fio2", "ph": "ph",
    "base excess": "baseexcess", "peep": "Positive end-expiratory pressure",
    "hco3": "bicarbonate", "bicarbonate": "bicarbonate",
    "total co2": "totalco2", "hct": "hematocrit", "hgb": "hemoglobin", "chloride": "chloride",
    "ionized calcium": "calcium", "potassium": "potassium", "sodium": "sodium", "lactate": "lactate",
    "glucose": "glucose", "bedside glucose": "glucose", "wbc x 1000": "wbc", "-basos": "basophils",
    "-eos": "eosinophils", "-lymphs": "lymphocytes", "-monos": "monocytes", "-polys": "neutrophils",
    "-bands": "bands", "nrbc": "nrbc", "troponin - t": "troponin_t", "cpk-mb": "ck_mb",
    "bnp": "ntprobnp", "albumin": "albumin", "total protein": "total_protein", "anion gap": "aniongap",
    "bun": "bun", "calcium": "calcium_chem", "creatinine": "creatinine", "fibrinogen": "fibrinogen",
    "pt - inr": "inr", "pt": "pt", "ptt": "ptt", "mch": "mch", "mchc": "mchc", "mcv": "mcv",
    "platelets x 1000": "platelet", "rbc": "rbc", "rdw": "rdw", "alt (sgpt)": "alt",
    "alkaline phos.": "alp", "ast (sgot)": "ast", "amylase": "amylase", "total bilirubin": "bilirubin_total",
    "direct bilirubin": "bilirubin_direct", "indirect bilirubin": "bilirubin_indirect", "cpk": "ck_cpk",
    "ggt": "ggt", "ldh": "ld_ldh", "crp": "crp", "urinary creatinine": "Creatinine urine",
    "magnesium": "Magnesium", "phosphate": "Phosphate", "urine rbc": "Red blood cell count urine",
    "24 h urine protein": "Total Protein Urine", "wbc's in urine": "White blood cell count urine", "urine ph": "pH urine",
}

# These are deliberately zero-filled by the published METRE eICU pipeline.
# They must not be treated as mapping failures in the cross-database audit.
EXPECTED_EICU_EMPTY = frozenset({
    "calcium", "atypical_lymphocytes", "immature_granulocytes", "metamyelocytes",
    "nrbc", "ntprobnp", "bilirubin_direct", "bilirubin_indirect", "ggt", "ld_ldh",
    "Peak inspiratory pressure", "Plateau Pressure",
    "Positive end-expiratory pressure Set", "Red blood cell count urine",
    "pH urine", "Total Protein Urine",
})


def _eicu_lab_concept(value: object) -> str | None:
    concept = EICU_LAB_ALIASES.get(str(value or "").strip().lower())
    return None if concept in EXPECTED_EICU_EMPTY else concept

EICU_NURSE_CHART_PAIRS: Mapping[tuple[str, str], str] = {
    ("heart rate", "heart rate"): "heart_rate",
    ("respiratory rate", "respiratory rate"): "resp_rate",
    ("o2 saturation", "o2 saturation"): "so2",
    ("non-invasive bp", "non-invasive bp systolic"): "sbp_ni",
    ("non-invasive bp", "non-invasive bp diastolic"): "dbp_ni",
    ("non-invasive bp", "non-invasive bp mean"): "mbp_ni",
    ("temperature", "temperature (c)"): "temperature",
    ("invasive bp", "invasive bp systolic"): "sbp",
    ("invasive bp", "invasive bp diastolic"): "dbp",
    ("invasive bp", "invasive bp mean"): "mbp",
    ("map (mmhg)", "value"): "mbp",
    ("arterial line map (mmhg)", "value"): "mbp",
    ("glasgow coma score", "gcs total"): "gcs",
}

EICU_HICL_EVENTS: Mapping[int, str] = {
    37410: "norepinephrine", 36346: "norepinephrine", 2051: "norepinephrine",
    37407: "epinephrine", 39089: "epinephrine", 36437: "epinephrine",
    34361: "epinephrine", 2050: "epinephrine", 8777: "dobutamine", 40: "dobutamine",
    2060: "dopamine", 2059: "dopamine", 37028: "phenylephrine",
    35517: "phenylephrine", 35587: "phenylephrine", 2087: "phenylephrine",
    38884: "vasopressin", 38883: "vasopressin", 2839: "vasopressin",
    9744: "milrinone", 39654: "heparin", 9545: "heparin", 2807: "heparin",
    33442: "heparin", 8643: "heparin", 33314: "heparin", 2808: "heparin",
    2810: "heparin",
}


def _culture_index(text: object) -> str | None:
    value = str(text or "").lower()
    if any(x in value for x in ("blood", "venipuncture", "serology")): return "culture_site_0"
    if any(x in value for x in ("urine", "kidney")): return "culture_site_1"
    if any(x in value for x in ("nasopharynx", "throat", "swab", "respiratory viral")): return "culture_site_2"
    if "stool" in value: return "culture_site_3"
    if "sputum" in value: return "culture_site_4"
    if "immunology" in value: return "culture_site_5"
    if "tissue" in value: return "culture_site_6"
    if "mrsa" in value or "staph aureus screen" in value: return "culture_site_7"
    if "csf" in value or "spinal" in value: return "culture_site_8"
    if "peritoneal" in value: return "culture_site_9"
    if "viral" in value or "antigen" in value: return "culture_site_10"
    if any(x in value for x in ("bronch", "tracheal")): return "culture_site_11"
    if "rectal" in value or "anorectal" in value or "vaginal" in value: return "culture_site_12"
    return "culture_site_13" if value else None


EVENT_PATTERNS: Mapping[str, tuple[str, ...]] = {
    "antib": ("cillin", "cef", "mycin", "floxacin", "meropenem", "vancomycin", "aztreonam", "doxycycline", "antibiotic"),
    "dopamine": ("dopamine",),
    # Longest/most specific names precede epinephrine because the latter is a
    # literal substring of norepinephrine.
    "norepinephrine": ("norepinephrine", "levophed"),
    "epinephrine": ("epinephrine", "adrenaline"),
    "phenylephrine": ("phenylephrine", "neosynephrine"),
    "vasopressin": ("vasopressin",), "dobutamine": ("dobutamine",), "milrinone": ("milrinone",),
    "heparin": ("heparin",), "vent": ("ventilat", "intubat", "endotracheal"),
    "crrt": ("crrt", "dialysis", "renal replacement"), "rbc": ("red blood", "packed cell", " rbc"),
    "platelets": ("platelet",), "ffp": ("fresh frozen", "plasma", "ffp"),
    "colloid": ("albumin", "colloid"), "crystalloid": ("crystalloid", "normal saline", "lactated ringer", "fluid bolus"),
}


def _match_event(value: object) -> str | None:
    text = " " + str(value or "").lower()
    return next((name for name, patterns in EVENT_PATTERNS.items() if any(pattern in text for pattern in patterns)), None)


def _mimic_gcs_score(
    motor: float,
    verbal: float,
    eyes: float,
    previous: tuple[float, float, float] = (np.nan, np.nan, np.nan),
) -> float:
    """Reproduce the key rules in mimic_derived.gcs for one chart time."""
    previous_motor, previous_verbal, previous_eyes = previous
    if verbal == 0 or (not np.isfinite(verbal) and previous_verbal == 0):
        return 15.0
    if previous_verbal == 0:
        return float(
            (motor if np.isfinite(motor) else 6.0)
            + (verbal if np.isfinite(verbal) else 5.0)
            + (eyes if np.isfinite(eyes) else 4.0)
        )
    return float(
        (motor if np.isfinite(motor) else previous_motor if np.isfinite(previous_motor) else 6.0)
        + (verbal if np.isfinite(verbal) else previous_verbal if np.isfinite(previous_verbal) else 5.0)
        + (eyes if np.isfinite(eyes) else previous_eyes if np.isfinite(previous_eyes) else 4.0)
    )


def _mimic_hourly(root: Path, cohort: pd.DataFrame, hours: int, chunksize: int) -> tuple[HourlyAccumulator, BinaryHourlyAccumulator, BinaryHourlyAccumulator]:
    numeric = HourlyAccumulator(len(cohort), NUMERIC_CONCEPTS, hours)
    cultures = BinaryHourlyAccumulator(len(cohort), CULTURE_SITES, hours)
    events = BinaryHourlyAccumulator(len(cohort), INTERVENTIONS, hours)
    stay_lookup = {int(stay): index for index, stay in enumerate(cohort.stay_id)}
    hadm_lookup = {int(hadm): index for index, hadm in enumerate(cohort.hadm_id)}
    starts_by_stay = pd.Series(cohort.intime.to_numpy(), index=cohort.stay_id.astype(np.int64)).to_dict()
    starts_by_hadm = pd.Series(cohort.intime.to_numpy(), index=cohort.hadm_id.astype(np.int64)).to_dict()
    gcs_previous: dict[int, tuple[float, float, float]] = {}
    chart_item_ids = set(MIMIC_CHART_ITEMS) | set(MIMIC_GCS_COMPONENTS)

    for number, chunk in enumerate(
        _chunks(root / "icu/chartevents.csv", ["stay_id", "charttime", "itemid", "value", "valuenum"], chunksize), 1
    ):
        chunk = chunk[chunk.stay_id.isin(stay_lookup) & chunk.itemid.isin(chart_item_ids)].copy()
        components = chunk[chunk.itemid.isin(MIMIC_GCS_COMPONENTS)].copy()
        if not components.empty:
            components["component"] = components.itemid.map(MIMIC_GCS_COMPONENTS)
            components["component_value"] = pd.to_numeric(components.valuenum, errors="coerce")
            ett = (components.itemid == 223900) & components.value.fillna("").eq("No Response-ETT")
            components.loc[ett, "component_value"] = 0.0
            pivot = components.pivot_table(
                index=["stay_id", "charttime"], columns="component", values="component_value", aggfunc="max"
            ).reset_index()
            for name in ("motor", "verbal", "eyes"):
                if name not in pivot:
                    pivot[name] = np.nan
            pivot = pivot.sort_values(["stay_id", "charttime"])
            gcs_rows: list[int] = []
            gcs_times: list[float] = []
            gcs_values: list[float] = []
            for record in pivot.itertuples(index=False):
                stay_id = int(record.stay_id)
                current = (float(record.motor), float(record.verbal), float(record.eyes))
                gcs_rows.append(stay_lookup[stay_id])
                gcs_times.append(
                    (pd.Timestamp(record.charttime) - pd.Timestamp(starts_by_stay[stay_id])).total_seconds() / 3600
                )
                gcs_values.append(_mimic_gcs_score(*current, gcs_previous.get(stay_id, (np.nan, np.nan, np.nan))))
                gcs_previous[stay_id] = current
            numeric.update(
                np.asarray(gcs_rows), ["gcs"] * len(gcs_rows), np.asarray(gcs_times), np.asarray(gcs_values)
            )
        regular = chunk[chunk.itemid.isin(MIMIC_CHART_ITEMS) & chunk.valuenum.notna()].copy()
        if not regular.empty:
            times = (pd.to_datetime(regular.charttime, errors="coerce") - pd.to_datetime(regular.stay_id.map(starts_by_stay))).dt.total_seconds().to_numpy() / 3600
            values = regular.valuenum.to_numpy(np.float64)
            values[regular.itemid.to_numpy() == 223761] = (values[regular.itemid.to_numpy() == 223761] - 32) * 5 / 9
            numeric.update(regular.stay_id.map(stay_lookup).to_numpy(np.int64), regular.itemid.map(MIMIC_CHART_ITEMS), times, values)
        if number % 20 == 0: print(f"mimic chartevents chunks={number}", flush=True)

    lab_dictionary = pd.read_csv(root / "hosp/d_labitems.csv", usecols=["itemid", "label"])
    lab_dictionary["concept"] = lab_dictionary.label.astype(str).str.lower().map(MIMIC_LAB_ALIASES)
    lab_map = lab_dictionary.dropna(subset=["concept"]).set_index("itemid").concept.to_dict()
    for number, chunk in enumerate(_chunks(root / "hosp/labevents.csv", ["hadm_id", "charttime", "itemid", "valuenum"], chunksize), 1):
        chunk = chunk[chunk.hadm_id.isin(hadm_lookup) & chunk.itemid.isin(lab_map) & chunk.valuenum.notna()].copy()
        if not chunk.empty:
            times = (pd.to_datetime(chunk.charttime, errors="coerce") - pd.to_datetime(chunk.hadm_id.map(starts_by_hadm))).dt.total_seconds().to_numpy() / 3600
            numeric.update(chunk.hadm_id.map(hadm_lookup).to_numpy(np.int64), chunk.itemid.map(lab_map), times, chunk.valuenum.to_numpy(np.float64))
        if number % 20 == 0: print(f"mimic labevents chunks={number}", flush=True)

    for chunk in _chunks(root / "icu/outputevents.csv", ["stay_id", "charttime", "value"], chunksize):
        chunk = chunk[chunk.stay_id.isin(stay_lookup)].copy()
        if not chunk.empty:
            times = (pd.to_datetime(chunk.charttime, errors="coerce") - pd.to_datetime(chunk.stay_id.map(starts_by_stay))).dt.total_seconds().to_numpy() / 3600
            numeric.update(chunk.stay_id.map(stay_lookup).to_numpy(np.int64), ["uo"] * len(chunk), times, pd.to_numeric(chunk.value, errors="coerce").to_numpy())

    for chunk in _chunks(root / "hosp/microbiologyevents.csv", ["hadm_id", "charttime", "spec_type_desc", "org_name", "ab_name", "interpretation"], chunksize):
        chunk = chunk[chunk.hadm_id.isin(hadm_lookup)].copy()
        if chunk.empty: continue
        times = (pd.to_datetime(chunk.charttime, errors="coerce") - pd.to_datetime(chunk.hadm_id.map(starts_by_hadm))).dt.total_seconds().to_numpy() / 3600
        rows = chunk.hadm_id.map(hadm_lookup).to_numpy(np.int64)
        sites = [_culture_index(value) for value in chunk.spec_type_desc]
        cultures.update(rows, sites, times)
        organism = chunk.org_name.fillna("").astype(str).str.lower()
        positive = ((organism != "") & ~organism.str.contains("no growth")).astype(float).to_numpy()
        screen = chunk.ab_name.notna().astype(float).to_numpy()
        interpretation = chunk.interpretation.fillna("").astype(str).str.upper()
        sensitive = interpretation.map({"S": 1.0, "R": 0.0}).to_numpy()
        numeric.update(np.repeat(rows, 3), np.tile(["positive_culture", "screen", "has_sensitivity"], len(rows)), np.repeat(times, 3), np.stack([positive, screen, sensitive], axis=1).reshape(-1))

    for chunk in _chunks(root / "hosp/prescriptions.csv", ["hadm_id", "starttime", "stoptime", "drug"], chunksize):
        chunk = chunk[chunk.hadm_id.isin(hadm_lookup)].copy()
        if chunk.empty: continue
        starts = (pd.to_datetime(chunk.starttime, errors="coerce") - pd.to_datetime(chunk.hadm_id.map(starts_by_hadm))).dt.total_seconds().to_numpy() / 3600
        stops = (pd.to_datetime(chunk.stoptime, errors="coerce") - pd.to_datetime(chunk.hadm_id.map(starts_by_hadm))).dt.total_seconds().to_numpy() / 3600
        events.update_intervals(chunk.hadm_id.map(hadm_lookup).to_numpy(np.int64), [_match_event(v) for v in chunk.drug], starts, stops)

    item_labels = pd.read_csv(root / "icu/d_items.csv", usecols=["itemid", "label"]).set_index("itemid").label.to_dict()
    for filename in ("inputevents.csv", "procedureevents.csv"):
        for chunk in _chunks(root / f"icu/{filename}", ["stay_id", "starttime", "endtime", "itemid"], chunksize):
            chunk = chunk[chunk.stay_id.isin(stay_lookup)].copy()
            if chunk.empty: continue
            starts = (pd.to_datetime(chunk.starttime, errors="coerce") - pd.to_datetime(chunk.stay_id.map(starts_by_stay))).dt.total_seconds().to_numpy() / 3600
            stops = (pd.to_datetime(chunk.endtime, errors="coerce") - pd.to_datetime(chunk.stay_id.map(starts_by_stay))).dt.total_seconds().to_numpy() / 3600
            labels = chunk.itemid.map(item_labels).fillna("")
            events.update_intervals(chunk.stay_id.map(stay_lookup).to_numpy(np.int64), [_match_event(v) for v in labels], starts, stops)
    return numeric, cultures, events


def _eicu_hourly(root: Path, cohort: pd.DataFrame, hours: int, chunksize: int) -> tuple[HourlyAccumulator, BinaryHourlyAccumulator, BinaryHourlyAccumulator]:
    numeric = HourlyAccumulator(len(cohort), NUMERIC_CONCEPTS, hours)
    cultures = BinaryHourlyAccumulator(len(cohort), CULTURE_SITES, hours)
    events = BinaryHourlyAccumulator(len(cohort), INTERVENTIONS, hours)
    lookup = {int(stay): index for index, stay in enumerate(cohort.patientunitstayid)}

    # METRE extracts routine vital signs from nurseCharting and uses
    # vitalPeriodic only for CVP (including the published 0.736 conversion).
    periodic = {"cvp": "Central Venous Pressure"}
    for chunk in _chunks(root / "vitalPeriodic.csv", ["patientunitstayid", "observationoffset", *periodic], chunksize):
        chunk = chunk[chunk.patientunitstayid.isin(lookup)]
        long = chunk.melt(id_vars=["patientunitstayid", "observationoffset"], value_vars=list(periodic), var_name="source", value_name="value").dropna(subset=["value"])
        numeric.update(long.patientunitstayid.map(lookup).to_numpy(np.int64), long.source.map(periodic), long.observationoffset.to_numpy(np.float64) / 60, long.value.to_numpy(np.float64) * 0.736)

    for chunk in _chunks(root / "lab.csv", ["patientunitstayid", "labresultoffset", "labname", "labresult"], chunksize):
        chunk = chunk[chunk.patientunitstayid.isin(lookup)].copy()
        names = chunk.labname.map(_eicu_lab_concept)
        keep = names.notna() & chunk.labresult.notna()
        kept_names = names[keep]
        kept_values = chunk.loc[keep, "labresult"].to_numpy(np.float64)
        fio2 = kept_names.to_numpy(dtype=object) == "fio2"
        kept_values[fio2 & (kept_values <= 1.0)] *= 100.0
        numeric.update(chunk.loc[keep, "patientunitstayid"].map(lookup).to_numpy(np.int64), kept_names, chunk.loc[keep, "labresultoffset"].to_numpy(np.float64) / 60, kept_values)

    for chunk in _chunks(root / "nurseCharting.csv", ["patientunitstayid", "nursingchartoffset", "nursingchartcelltypevallabel", "nursingchartcelltypevalname", "nursingchartvalue"], chunksize):
        chunk = chunk[chunk.patientunitstayid.isin(lookup)].copy()
        labels = chunk.nursingchartcelltypevallabel.fillna("").astype(str).str.strip().str.lower()
        names = chunk.nursingchartcelltypevalname.fillna("").astype(str).str.strip().str.lower()
        concepts = pd.Series(
            [EICU_NURSE_CHART_PAIRS.get(pair) for pair in zip(labels, names)], index=chunk.index, dtype=object
        )
        keep = concepts.notna()
        numeric.update(chunk.loc[keep, "patientunitstayid"].map(lookup).to_numpy(np.int64), concepts[keep], chunk.loc[keep, "nursingchartoffset"].to_numpy(np.float64) / 60, pd.to_numeric(chunk.loc[keep, "nursingchartvalue"], errors="coerce").to_numpy())

    # The published eICU construction only takes observed tidal volume from
    # respiratoryCharting.  PEEP and FiO2 are taken from the lab table.
    for chunk in _chunks(root / "respiratoryCharting.csv", ["patientunitstayid", "respchartoffset", "respchartvaluelabel", "respchartvalue"], chunksize):
        chunk = chunk[chunk.patientunitstayid.isin(lookup)].copy()
        labels = chunk.respchartvaluelabel.fillna("").astype(str).str.lower()
        keep = labels.eq("tidal volume observed (vt)")
        numeric.update(chunk.loc[keep, "patientunitstayid"].map(lookup).to_numpy(np.int64), ["Tidal Volume Observed"] * int(keep.sum()), chunk.loc[keep, "respchartoffset"].to_numpy(np.float64) / 60, pd.to_numeric(chunk.loc[keep, "respchartvalue"], errors="coerce").to_numpy())

    procedure_ranges: dict[tuple[int, str], tuple[float, float]] = {}
    for chunk in _chunks(root / "intakeOutput.csv", ["patientunitstayid", "intakeoutputoffset", "cellpath", "celllabel", "cellvaluenumeric"], chunksize):
        chunk = chunk[chunk.patientunitstayid.isin(lookup)].copy()
        text = (chunk.cellpath.fillna("").astype(str) + " " + chunk.celllabel.fillna("").astype(str)).str.lower()
        urine = text.str.contains("urine") & chunk.cellvaluenumeric.notna()
        numeric.update(chunk.loc[urine, "patientunitstayid"].map(lookup).to_numpy(np.int64), ["uo"] * int(urine.sum()), chunk.loc[urine, "intakeoutputoffset"].to_numpy(np.float64) / 60, chunk.loc[urine, "cellvaluenumeric"].to_numpy(np.float64))
        concepts = text.map(_match_event)
        keep = concepts.isin(PROCEDURES)
        procedure_frame = pd.DataFrame({
            "row": chunk.loc[keep, "patientunitstayid"].map(lookup).to_numpy(np.int64),
            "concept": concepts[keep].to_numpy(dtype=object),
            "time": pd.to_numeric(chunk.loc[keep, "intakeoutputoffset"], errors="coerce").to_numpy() / 60,
        }).dropna(subset=["time"])
        for (row, concept), group in procedure_frame.groupby(["row", "concept"]):
            key = (int(row), str(concept))
            start, stop = float(group.time.min()), float(group.time.max())
            if key in procedure_ranges:
                old_start, old_stop = procedure_ranges[key]
                start, stop = min(start, old_start), max(stop, old_stop)
            procedure_ranges[key] = (start, stop)
    if procedure_ranges:
        events.update_intervals(
            np.asarray([key[0] for key in procedure_ranges]),
            [key[1] for key in procedure_ranges],
            np.asarray([value[0] for value in procedure_ranges.values()]),
            np.asarray([value[1] for value in procedure_ranges.values()]),
        )

    for chunk in _chunks(root / "microLab.csv", ["patientunitstayid", "culturetakenoffset", "culturesite", "organism", "antibiotic", "sensitivitylevel"], chunksize):
        chunk = chunk[chunk.patientunitstayid.isin(lookup)].copy()
        rows = chunk.patientunitstayid.map(lookup).to_numpy(np.int64)
        times = chunk.culturetakenoffset.to_numpy(np.float64) / 60
        cultures.update(rows, [_culture_index(value) for value in chunk.culturesite], times)
        organism = chunk.organism.fillna("").astype(str).str.lower()
        positive = ((organism != "") & (organism != "no growth")).astype(float).to_numpy()
        screen = chunk.antibiotic.notna().astype(float).to_numpy()
        sensitive = chunk.sensitivitylevel.map({"Sensitive": 1.0, "Resistant": 0.0}).to_numpy()
        numeric.update(np.repeat(rows, 3), np.tile(["positive_culture", "screen", "has_sensitivity"], len(rows)), np.repeat(times, 3), np.stack([positive, screen, sensitive], axis=1).reshape(-1))

    medication_columns = [
        "patientunitstayid", "drugorderoffset", "drugstartoffset", "drugstopoffset",
        "drugordercancelled", "drugname", "drughiclseqno", "dosage",
    ]
    for chunk in _chunks(root / "medication.csv", medication_columns, chunksize):
        chunk = chunk[
            chunk.patientunitstayid.isin(lookup)
            & chunk.drugordercancelled.eq("No")
            & chunk.dosage.notna()
        ].copy()
        if chunk.empty:
            continue
        name_events = [_match_event(value) for value in chunk.drugname]
        hicl_events = pd.to_numeric(chunk.drughiclseqno, errors="coerce").map(EICU_HICL_EVENTS)
        concepts = [hicl if pd.notna(hicl) else name for hicl, name in zip(hicl_events, name_events)]
        starts = pd.to_numeric(chunk.drugorderoffset, errors="coerce").to_numpy(np.float64) / 60
        antibiotic = np.asarray(concepts, dtype=object) == "antib"
        starts[antibiotic] = pd.to_numeric(chunk.loc[antibiotic, "drugstartoffset"], errors="coerce").to_numpy(np.float64) / 60
        stops = pd.to_numeric(chunk.drugstopoffset, errors="coerce").to_numpy(np.float64) / 60
        events.update_intervals(chunk.patientunitstayid.map(lookup).to_numpy(np.int64), concepts, starts, stops)
    for chunk in _chunks(root / "respiratoryCare.csv", ["patientunitstayid", "priorventstartoffset", "priorventendoffset"], chunksize):
        chunk = chunk[chunk.patientunitstayid.isin(lookup)].copy()
        events.update_intervals(chunk.patientunitstayid.map(lookup).to_numpy(np.int64), ["vent"] * len(chunk), pd.to_numeric(chunk.priorventstartoffset, errors="coerce").to_numpy() / 60, pd.to_numeric(chunk.priorventendoffset, errors="coerce").to_numpy() / 60)

    # Admission weight is a valid baseline observation at hour zero.
    weights = pd.to_numeric(cohort.admissionweight, errors="coerce").to_numpy()
    valid = np.isfinite(weights)
    numeric.update(np.flatnonzero(valid), ["weight"] * int(valid.sum()), np.zeros(int(valid.sum())), weights[valid])
    return numeric, cultures, events


def _write(
    root: Path,
    dataset: str,
    cohort: pd.DataFrame,
    features: Mapping[str, np.ndarray],
    availability: Mapping[str, np.ndarray],
    splits: Mapping[str, np.ndarray],
) -> dict[str, object]:
    audit: dict[str, object] = {"dataset": dataset, "splits": {}}
    for split, indices in splits.items():
        directory = root / dataset / split
        directory.mkdir(parents=True, exist_ok=True)
        for name in MODALITIES:
            np.save(directory / f"{name}.npy", features[name][indices].astype(np.float16 if name != "static" else np.float32))
        mask = np.stack([availability[name] for name in MODALITIES], axis=1)[indices].astype(np.uint8)
        labels = cohort.label.to_numpy(np.uint8)[indices]
        np.save(directory / "mask.npy", mask)
        np.save(directory / "labels.npy", labels)
        (directory / "ids.txt").write_text("\n".join(cohort.stay_key.iloc[indices].astype(str)) + "\n", encoding="utf-8")
        (directory / "patient_ids.txt").write_text("\n".join(cohort.patient_id.iloc[indices].astype(str)) + "\n", encoding="utf-8")
        audit["splits"][split] = {
            "n": int(len(indices)), "positives": int(labels.sum()), "prevalence": float(labels.mean()),
            "unique_patients": int(cohort.patient_id.iloc[indices].nunique()),
            "modality_availability": {name: float(mask[:, index].mean()) for index, name in enumerate(MODALITIES)},
        }
    (root / dataset / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def _features(
    numeric: HourlyAccumulator,
    cultures: BinaryHourlyAccumulator,
    events: BinaryHourlyAccumulator,
    static: np.ndarray,
    train_rows: np.ndarray,
    normalization: Mapping[str, Sequence[float]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    means, observed = numeric.means_and_observed()
    if normalization is None:
        normalized, statistics = impute_and_normalize(means, observed, train_rows)
    else:
        center = np.asarray(normalization["mean"], dtype=np.float32)
        scale = np.asarray(normalization["std"], dtype=np.float32)
        normalized = (means - center[None, None, :]) / scale[None, None, :]
        for hour in range(1, normalized.shape[1]):
            missing = ~observed[:, hour, :]
            normalized[:, hour, :][missing] = normalized[:, hour - 1, :][missing]
        finite = np.isfinite(normalized)
        stay_mean = np.divide(
            np.nansum(normalized, axis=1), finite.sum(axis=1),
            out=np.full((normalized.shape[0], normalized.shape[2]), np.nan, dtype=np.float32),
            where=finite.sum(axis=1) > 0,
        )
        missing = ~np.isfinite(normalized)
        replacement = np.broadcast_to(stay_mean[:, None, :], normalized.shape)
        normalized[missing] = replacement[missing]
        normalized = np.nan_to_num(normalized).astype(np.float32)
        statistics = {"mean": center.tolist(), "std": scale.tolist()}
    concept_index = {name: index for index, name in enumerate(NUMERIC_CONCEPTS)}
    bedside_indices = [concept_index[name] for name in BEDSIDES]
    lab_indices = [concept_index[name] for name in LABS]
    event_index = {name: index for index, name in enumerate(INTERVENTIONS)}
    features = {
        "static": static,
        "bedside": paired_channels(normalized[:, :, bedside_indices], observed[:, :, bedside_indices]),
        "laboratory": np.concatenate((
            paired_channels(normalized[:, :, lab_indices], observed[:, :, lab_indices]),
            cultures.values.astype(np.float32),
        ), axis=2),
        "medications": events.values[:, :, [event_index[name] for name in MEDICATIONS]].astype(np.float32),
        "procedures": events.values[:, :, [event_index[name] for name in PROCEDURES]].astype(np.float32),
    }
    availability = {
        "static": np.ones(len(static), dtype=np.uint8),
        "bedside": observed[:, :, bedside_indices].any(axis=(1, 2)).astype(np.uint8),
        "laboratory": (observed[:, :, lab_indices].any(axis=(1, 2)) | cultures.values.any(axis=(1, 2))).astype(np.uint8),
        # A zero intervention record is an observed absence, not a missing modality.
        "medications": np.ones(len(static), dtype=np.uint8),
        "procedures": np.ones(len(static), dtype=np.uint8),
    }
    return features, availability, statistics


def _channel_audit(
    numeric: HourlyAccumulator,
    cultures: BinaryHourlyAccumulator,
    events: BinaryHourlyAccumulator,
) -> dict[str, object]:
    numeric_rates = (numeric.count > 0).mean(axis=(0, 1))
    culture_rates = cultures.values.mean(axis=(0, 1))
    event_rates = events.values.mean(axis=(0, 1))
    return {
        "numeric_observation_rate": {
            name: float(numeric_rates[index]) for index, name in enumerate(NUMERIC_CONCEPTS)
        },
        "culture_hourly_rate": {
            name: float(culture_rates[index]) for index, name in enumerate(CULTURE_SITES)
        },
        "intervention_hourly_rate": {
            name: float(event_rates[index]) for index, name in enumerate(INTERVENTIONS)
        },
    }


def _alignment_audit(
    mimic: Mapping[str, object],
    eicu: Mapping[str, object],
) -> dict[str, object]:
    source = mimic["numeric_observation_rate"]
    target = eicu["numeric_observation_rate"]
    target_absent = sorted(name for name in NUMERIC_CONCEPTS if float(target[name]) == 0.0)
    unexpected_target_absent = sorted(set(target_absent) - EXPECTED_EICU_EMPTY)
    expected_but_present = sorted(
        name for name in EXPECTED_EICU_EMPTY if float(target.get(name, 0.0)) > 0.0
    )
    source_absent = sorted(name for name in NUMERIC_CONCEPTS if float(source[name]) == 0.0)
    return {
        "official_expected_eicu_empty": sorted(EXPECTED_EICU_EMPTY),
        "eicu_absent": target_absent,
        "unexpected_eicu_absent": unexpected_target_absent,
        "official_empty_but_present": expected_but_present,
        "mimic_absent": source_absent,
        "passes_expected_empty_check": not unexpected_target_absent and not expected_but_present,
    }


def _validate(root: Path, feature_shapes: Mapping[str, Sequence[int]]) -> dict[str, object]:
    mimic_patients: dict[str, set[str]] = {}
    audit: dict[str, object] = {"finite": True, "split_checks": {}}
    for dataset, splits in (("mimic4", ("train", "val", "test")), ("eicu", ("test",))):
        for split in splits:
            directory = root / dataset / split
            labels = np.load(directory / "labels.npy", mmap_mode="r")
            mask = np.load(directory / "mask.npy", mmap_mode="r")
            patients = (directory / "patient_ids.txt").read_text().splitlines()
            if mask.shape != (len(labels), len(MODALITIES)): raise ValueError(f"bad mask {dataset}/{split}: {mask.shape}")
            for name, trailing in feature_shapes.items():
                values = np.load(directory / f"{name}.npy", mmap_mode="r")
                if values.shape != (len(labels), *trailing): raise ValueError(f"bad {name} shape {dataset}/{split}: {values.shape}")
                if not np.isfinite(values).all(): raise ValueError(f"nonfinite {name} in {dataset}/{split}")
            if dataset == "mimic4": mimic_patients[split] = set(patients)
            audit["split_checks"][f"{dataset}/{split}"] = {"n": len(labels), "positives": int(labels.sum())}
    overlap = {
        "train_val": len(mimic_patients["train"] & mimic_patients["val"]),
        "train_test": len(mimic_patients["train"] & mimic_patients["test"]),
        "val_test": len(mimic_patients["val"] & mimic_patients["test"]),
    }
    if any(overlap.values()): raise ValueError(f"patient leakage: {overlap}")
    audit["mimic_patient_overlap"] = overlap
    return audit


def main() -> None:
    args = parse_args()
    mimic_root, eicu_root, output_root = Path(args.mimic_root), Path(args.eicu_root), Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    mimic = _limit_cohort(_mimic_cohort(mimic_root, args.observation_hours, args.gap_hours, args.max_los_hours), args.max_stays, args.seed)
    eicu = _limit_cohort(_eicu_cohort(eicu_root, args.observation_hours, args.gap_hours, args.max_los_hours), args.max_stays, args.seed)
    mimic_splits = _patient_disjoint_split(mimic, args.seed)
    print(json.dumps({"mimic4": {"n": len(mimic), "prevalence": float(mimic.label.mean())}, "eicu": {"n": len(eicu), "prevalence": float(eicu.label.mean())}}), flush=True)

    mimic_numeric, mimic_cultures, mimic_events = _mimic_hourly(mimic_root, mimic, args.observation_hours, args.chunksize)
    mimic_static = _static_features("mimic4", mimic_root, mimic, args.chunksize)
    mimic_channel_audit = _channel_audit(mimic_numeric, mimic_cultures, mimic_events)
    mimic_features, mimic_availability, normalization = _features(
        mimic_numeric, mimic_cultures, mimic_events, mimic_static, mimic_splits["train"]
    )
    del mimic_numeric, mimic_cultures, mimic_events

    eicu_numeric, eicu_cultures, eicu_events = _eicu_hourly(eicu_root, eicu, args.observation_hours, args.chunksize)
    eicu_static = _static_features("eicu", eicu_root, eicu, args.chunksize)
    eicu_channel_audit = _channel_audit(eicu_numeric, eicu_cultures, eicu_events)
    eicu_features, eicu_availability, _ = _features(
        eicu_numeric, eicu_cultures, eicu_events, eicu_static, np.arange(len(eicu)), normalization
    )
    audits = {
        "mimic4": _write(output_root, "mimic4", mimic, mimic_features, mimic_availability, mimic_splits),
        "eicu": _write(output_root, "eicu", eicu, eicu_features, eicu_availability, {"test": np.arange(len(eicu))}),
    }
    feature_shapes = {name: list(values.shape[1:]) for name, values in mimic_features.items()}
    validation = _validate(output_root, feature_shapes)
    channel_audits = {"mimic4": mimic_channel_audit, "eicu": eicu_channel_audit}
    alignment = _alignment_audit(mimic_channel_audit, eicu_channel_audit)
    manifest = {
        "protocol": "METRE-compatible local raw-CSV reconstruction with explicit clinical modalities",
        "reference": "Liao and Voldman, Journal of Biomedical Informatics 141 (2023) 104356",
        "fidelity_note": "The public METRE output schema and task window are reproduced; local raw CSV mappings replace unavailable BigQuery-derived intermediate tables.",
        "observation_hours": args.observation_hours,
        "prediction_gap_hours": args.gap_hours,
        "minimum_icu_los_hours": args.observation_hours + args.gap_hours,
        "maximum_icu_los_hours": args.max_los_hours,
        "label_definition": "death during the same hospital admission in both MIMIC-IV and eICU",
        "modalities": list(MODALITIES),
        "temporal_modalities": list(TEMPORAL_MODALITIES),
        "feature_shapes": feature_shapes,
        "pre_normalized": True,
        "normalization": {"fit_on": "MIMIC-IV train observed values only", **normalization},
        "static_features": list(STATIC_FEATURES),
        "bedside_concepts": list(BEDSIDES),
        "laboratory_concepts": list(LABS),
        "culture_sites": list(CULTURE_SITES),
        "medication_concepts": list(MEDICATIONS),
        "procedure_concepts": list(PROCEDURES),
        "temporal_channel_audit": {"bedside": 40, "laboratory": 144, "medications": 9, "procedures": 7, "total": 200},
        "imputation": "hourly mean; source-train z-score; causal forward fill; within-stay mean for leading gaps; source mean zero fallback; original observation indicators retained",
        "split_policy": "MIMIC first ICU-associated hospital stay and first ICU stay, patient-disjoint 70/10/20; eICU held out in full; source-validation-only model selection",
        "seed": args.seed,
        "audits": audits,
        "channel_audits": channel_audits,
        "cross_dataset_alignment": alignment,
        "validation": validation,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
