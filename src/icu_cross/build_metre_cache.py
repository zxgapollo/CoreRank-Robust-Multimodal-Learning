from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .features import broad_ethnicity
from .metre_features import (
    EICU_LAB_NAME_MAP,
    EVENT_NAMES,
    INTERVENTION_CONCEPTS,
    INTERVENTION_PATTERNS,
    LAB_CONCEPTS,
    MEDICATION_CONCEPTS,
    MEDICATION_PATTERNS,
    METRE_MODALITIES,
    MIMIC_LAB_ITEM_MAP,
    MIMIC_VITAL_ITEM_MAP,
    SUMMARY_NAMES,
    VITAL_CONCEPTS,
    ContinuousAccumulator,
    EventAccumulator,
    match_categories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local-CSV METRE-style MIMIC-IV/eICU caches.")
    parser.add_argument("--mimic-root", required=True)
    parser.add_argument("--eicu-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--observation-hours", type=float, default=48.0)
    parser.add_argument("--gap-hours", type=float, default=6.0)
    parser.add_argument("--max-los-hours", type=float, default=240.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--max-stays", type=int)
    return parser.parse_args()


def _read_chunks(path: Path, usecols: Iterable[str], chunksize: int, **kwargs: object) -> Iterable[pd.DataFrame]:
    yield from pd.read_csv(path, usecols=list(usecols), chunksize=chunksize, low_memory=False, **kwargs)


def _mimic_cohort(root: Path, observation: float, gap: float, max_los: float) -> pd.DataFrame:
    icu = pd.read_csv(root / "icu/icustays.csv", usecols=["subject_id", "hadm_id", "stay_id", "intime", "los"])
    adm = pd.read_csv(root / "hosp/admissions.csv", usecols=["subject_id", "hadm_id", "race", "hospital_expire_flag"])
    patients = pd.read_csv(root / "hosp/patients.csv", usecols=["subject_id", "gender", "anchor_age", "anchor_year"])
    cohort = icu.merge(adm, on=["subject_id", "hadm_id"], validate="many_to_one").merge(
        patients, on="subject_id", validate="many_to_one"
    )
    cohort["intime"] = pd.to_datetime(cohort["intime"])
    cohort["age"] = cohort["anchor_age"] + cohort["intime"].dt.year - cohort["anchor_year"]
    cohort["los_hours"] = cohort["los"] * 24.0
    cohort = cohort[
        (cohort["age"] >= 18)
        & (cohort["los_hours"] >= observation + gap)
        & (cohort["los_hours"] <= max_los)
        & cohort["hospital_expire_flag"].isin([0, 1])
    ].copy()
    cohort["dataset"] = "mimic4"
    cohort["patient_id"] = cohort["subject_id"].astype(str)
    cohort["stay_key"] = cohort["stay_id"].astype(str)
    cohort["label"] = cohort["hospital_expire_flag"].astype(np.uint8)
    cohort["ethnicity"] = cohort["race"]
    return cohort.reset_index(drop=True)


def _eicu_cohort(root: Path, observation: float, gap: float, max_los: float) -> pd.DataFrame:
    columns = [
        "patientunitstayid", "uniquepid", "gender", "age", "ethnicity", "admissionheight",
        "admissionweight", "hospitalid", "unitdischargeoffset", "hospitaldischargestatus",
    ]
    cohort = pd.read_csv(root / "patient.csv", usecols=columns)
    cohort["age_num"] = pd.to_numeric(cohort["age"].replace({"> 89": "90"}), errors="coerce")
    cohort["los_hours"] = cohort["unitdischargeoffset"] / 60.0
    cohort = cohort[
        (cohort["age_num"] >= 18)
        & (cohort["los_hours"] >= observation + gap)
        & (cohort["los_hours"] <= max_los)
        & cohort["hospitaldischargestatus"].isin(["Alive", "Expired"])
    ].copy()
    cohort["dataset"] = "eicu"
    cohort["patient_id"] = cohort["uniquepid"].astype(str)
    cohort["stay_key"] = cohort["patientunitstayid"].astype(str)
    cohort["label"] = (cohort["hospitaldischargestatus"] == "Expired").astype(np.uint8)
    cohort["age"] = cohort["age_num"]
    return cohort.reset_index(drop=True)


def _limit_cohort(cohort: pd.DataFrame, max_stays: int | None, seed: int) -> pd.DataFrame:
    if max_stays is None or len(cohort) <= max_stays:
        return cohort.reset_index(drop=True)
    positive = cohort[cohort.label == 1]
    negative = cohort[cohort.label == 0]
    positive_n = max(1, round(max_stays * len(positive) / len(cohort)))
    result = pd.concat(
        [positive.sample(min(len(positive), positive_n), random_state=seed),
         negative.sample(min(len(negative), max_stays - positive_n), random_state=seed)]
    )
    return result.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _demographics(cohort: pd.DataFrame) -> np.ndarray:
    gender_names = ("female", "male", "unknown")
    ethnicity_names = ("white", "black", "asian", "hispanic", "native", "other", "unknown")
    result = np.zeros((len(cohort), 11), dtype=np.float32)
    result[:, 0] = cohort["age"].to_numpy(np.float32) / 100.0
    for row, record in enumerate(cohort.itertuples(index=False)):
        raw_gender = str(getattr(record, "gender", "") or "").strip().upper()
        gender = "female" if raw_gender in {"F", "FEMALE"} else "male" if raw_gender in {"M", "MALE"} else "unknown"
        ethnicity = broad_ethnicity(getattr(record, "ethnicity", ""))
        result[row, 1:4] = [float(gender == name) for name in gender_names]
        result[row, 4:] = [float(ethnicity == name) for name in ethnicity_names]
    return result


def _continuous_frame(
    chunk: pd.DataFrame,
    stay_lookup: Mapping[int, int],
    concept_map: Mapping[object, str],
    id_col: str,
    concept_col: str,
    time_values: np.ndarray,
    value_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stay = chunk[id_col].map(stay_lookup).fillna(-1).to_numpy(np.int64)
    concepts = chunk[concept_col].map(concept_map).to_numpy(object)
    return stay, concepts, np.asarray(time_values, dtype=np.float64), np.asarray(value_values, dtype=np.float64)


def _mimic_continuous(
    root: Path,
    cohort: pd.DataFrame,
    observation: float,
    chunksize: int,
    vital_acc: ContinuousAccumulator,
    lab_acc: ContinuousAccumulator,
) -> None:
    stay_lookup = {int(value): index for index, value in enumerate(cohort.stay_id)}
    intime = pd.Series(cohort.intime.to_numpy(), index=cohort.stay_id.astype(np.int64)).to_dict()
    vital_ids = set(MIMIC_VITAL_ITEM_MAP)
    for number, chunk in enumerate(
        _read_chunks(root / "icu/chartevents.csv", ["stay_id", "charttime", "itemid", "valuenum"], chunksize), start=1
    ):
        chunk = chunk[chunk.itemid.isin(vital_ids) & chunk.stay_id.isin(stay_lookup) & chunk.valuenum.notna()].copy()
        if chunk.empty:
            continue
        charttime = pd.to_datetime(chunk.charttime, errors="coerce")
        starts = pd.to_datetime(chunk.stay_id.map(intime), errors="coerce")
        hours = (charttime - starts).dt.total_seconds().to_numpy() / 3600.0
        values = chunk.valuenum.to_numpy(np.float64)
        fahrenheit = chunk.itemid.to_numpy() == 223761
        values[fahrenheit] = (values[fahrenheit] - 32.0) * 5.0 / 9.0
        valid = (hours >= 0) & (hours < observation)
        stay, concept, times, values = _continuous_frame(
            chunk, stay_lookup, MIMIC_VITAL_ITEM_MAP, "stay_id", "itemid", hours, values
        )
        vital_acc.update(stay[valid], concept[valid], times[valid], values[valid])
        if number % 10 == 0:
            print(f"mimic chartevents chunks={number}", flush=True)

    hadm_to_rows: dict[int, list[int]] = {}
    for index, hadm in enumerate(cohort.hadm_id.astype(np.int64)):
        hadm_to_rows.setdefault(int(hadm), []).append(index)
    lab_ids = set(MIMIC_LAB_ITEM_MAP)
    for number, chunk in enumerate(
        _read_chunks(root / "hosp/labevents.csv", ["hadm_id", "charttime", "itemid", "valuenum"], chunksize), start=1
    ):
        chunk = chunk[chunk.itemid.isin(lab_ids) & chunk.hadm_id.isin(hadm_to_rows) & chunk.valuenum.notna()].copy()
        if chunk.empty:
            continue
        chunk["_rows"] = chunk.hadm_id.map(hadm_to_rows)
        chunk = chunk.explode("_rows", ignore_index=True)
        rows = chunk.pop("_rows").to_numpy(np.int64)
        starts = cohort.intime.to_numpy()[rows]
        hours = (pd.to_datetime(chunk.charttime, errors="coerce") - pd.to_datetime(starts)).dt.total_seconds().to_numpy() / 3600.0
        valid = (hours >= 0) & (hours < observation)
        concepts = chunk.itemid.map(MIMIC_LAB_ITEM_MAP).to_numpy(object)
        lab_acc.update(rows[valid], concepts[valid], hours[valid], chunk.valuenum.to_numpy(np.float64)[valid])
        if number % 10 == 0:
            print(f"mimic labevents chunks={number}", flush=True)


def _eicu_continuous(
    root: Path,
    cohort: pd.DataFrame,
    observation: float,
    chunksize: int,
    vital_acc: ContinuousAccumulator,
    lab_acc: ContinuousAccumulator,
) -> None:
    lookup = {int(value): index for index, value in enumerate(cohort.patientunitstayid)}
    vital_columns = {
        "heartrate": "heart_rate", "respiration": "resp_rate", "sao2": "spo2",
        "temperature": "temperature", "systemicsystolic": "sbp",
        "systemicdiastolic": "dbp", "systemicmean": "map",
    }
    usecols = ["patientunitstayid", "observationoffset", *vital_columns]
    for number, chunk in enumerate(_read_chunks(root / "vitalPeriodic.csv", usecols, chunksize), start=1):
        chunk = chunk[chunk.patientunitstayid.isin(lookup) & chunk.observationoffset.between(0, observation * 60, inclusive="left")]
        if chunk.empty:
            continue
        long = chunk.melt(
            id_vars=["patientunitstayid", "observationoffset"], value_vars=list(vital_columns),
            var_name="source", value_name="value",
        ).dropna(subset=["value"])
        stays = long.patientunitstayid.map(lookup).to_numpy(np.int64)
        vital_acc.update(stays, long.source.map(vital_columns), long.observationoffset.to_numpy() / 60.0, long.value.to_numpy(np.float64))
        if number % 10 == 0:
            print(f"eicu vitalPeriodic chunks={number}", flush=True)

    aperiodic_columns = {
        "noninvasivesystolic": "sbp", "noninvasivediastolic": "dbp", "noninvasivemean": "map",
    }
    aperiodic_usecols = ["patientunitstayid", "observationoffset", *aperiodic_columns]
    for number, chunk in enumerate(_read_chunks(root / "vitalAperiodic.csv", aperiodic_usecols, chunksize), start=1):
        chunk = chunk[
            chunk.patientunitstayid.isin(lookup)
            & chunk.observationoffset.between(0, observation * 60, inclusive="left")
        ]
        if chunk.empty:
            continue
        long = chunk.melt(
            id_vars=["patientunitstayid", "observationoffset"], value_vars=list(aperiodic_columns),
            var_name="source", value_name="value",
        ).dropna(subset=["value"])
        stays = long.patientunitstayid.map(lookup).to_numpy(np.int64)
        vital_acc.update(
            stays, long.source.map(aperiodic_columns), long.observationoffset.to_numpy() / 60.0,
            long.value.to_numpy(np.float64),
        )
        if number % 10 == 0:
            print(f"eicu vitalAperiodic chunks={number}", flush=True)

    for number, chunk in enumerate(
        _read_chunks(root / "lab.csv", ["patientunitstayid", "labresultoffset", "labname", "labresult"], chunksize), start=1
    ):
        names = chunk.labname.astype(str).str.lower()
        chunk = chunk[
            chunk.patientunitstayid.isin(lookup)
            & chunk.labresultoffset.between(0, observation * 60, inclusive="left")
            & names.isin(EICU_LAB_NAME_MAP)
            & chunk.labresult.notna()
        ].copy()
        if chunk.empty:
            continue
        stays = chunk.patientunitstayid.map(lookup).to_numpy(np.int64)
        concepts = chunk.labname.astype(str).str.lower().map(EICU_LAB_NAME_MAP)
        lab_acc.update(stays, concepts, chunk.labresultoffset.to_numpy() / 60.0, chunk.labresult.to_numpy(np.float64))
        if number % 10 == 0:
            print(f"eicu lab chunks={number}", flush=True)


def _event_rows(
    chunk: pd.DataFrame,
    lookup: Mapping[int, int],
    id_col: str,
    text_col: str,
    times: np.ndarray,
    patterns: Mapping[str, tuple[str, ...]],
    accumulator: EventAccumulator,
    observation: float,
) -> None:
    chunk = chunk.copy()
    chunk["_event_time"] = np.asarray(times, dtype=np.float64)
    chunk = chunk[chunk[id_col].isin(lookup)].copy()
    if chunk.empty:
        return
    concepts = np.asarray(match_categories(chunk[text_col].fillna("").astype(str), patterns), dtype=object)
    stays = chunk[id_col].map(lookup).fillna(-1).to_numpy(np.int64)
    times = chunk.pop("_event_time").to_numpy(np.float64)
    valid = (concepts != None) & (times >= 0) & (times < observation)  # noqa: E711
    accumulator.update(stays[valid], concepts[valid], times[valid])


def _mimic_events(
    root: Path,
    cohort: pd.DataFrame,
    observation: float,
    chunksize: int,
    medications: EventAccumulator,
    interventions: EventAccumulator,
) -> None:
    stay_lookup = {int(value): index for index, value in enumerate(cohort.stay_id)}
    intime_by_stay = pd.Series(cohort.intime.to_numpy(), index=cohort.stay_id.astype(np.int64)).to_dict()
    hadm_to_rows: dict[int, list[int]] = {}
    for index, hadm in enumerate(cohort.hadm_id.astype(np.int64)):
        hadm_to_rows.setdefault(int(hadm), []).append(index)

    for chunk in _read_chunks(root / "hosp/prescriptions.csv", ["hadm_id", "starttime", "stoptime", "drug"], chunksize):
        chunk = chunk[chunk.hadm_id.isin(hadm_to_rows)].copy()
        if chunk.empty:
            continue
        chunk["_rows"] = chunk.hadm_id.map(hadm_to_rows)
        chunk = chunk.explode("_rows", ignore_index=True)
        rows = chunk.pop("_rows").to_numpy(np.int64)
        starts = pd.to_datetime(chunk.starttime, errors="coerce")
        stops = pd.to_datetime(chunk.stoptime, errors="coerce")
        icu_starts = pd.to_datetime(cohort.intime.to_numpy()[rows])
        raw_hours = (starts - icu_starts).dt.total_seconds().to_numpy() / 3600.0
        stop_hours = (stops - icu_starts).dt.total_seconds().to_numpy() / 3600.0
        hours = np.maximum(raw_hours, 0.0)
        concepts = np.asarray(match_categories(chunk.drug, MEDICATION_PATTERNS), dtype=object)
        overlaps_window = (raw_hours < observation) & (np.isnan(stop_hours) | (stop_hours >= 0))
        valid = (concepts != None) & overlaps_window  # noqa: E711
        medications.update(rows[valid], concepts[valid], hours[valid])

    item_labels = pd.read_csv(root / "icu/d_items.csv", usecols=["itemid", "label"]).set_index("itemid").label.to_dict()
    for filename in ("inputevents.csv", "procedureevents.csv"):
        for chunk in _read_chunks(root / f"icu/{filename}", ["stay_id", "starttime", "itemid"], chunksize):
            chunk = chunk[chunk.stay_id.isin(stay_lookup)].copy()
            if chunk.empty:
                continue
            starts = pd.to_datetime(chunk.stay_id.map(intime_by_stay), errors="coerce")
            hours = (pd.to_datetime(chunk.starttime, errors="coerce") - starts).dt.total_seconds().to_numpy() / 3600.0
            chunk["label"] = chunk.itemid.map(item_labels).fillna("")
            _event_rows(chunk, stay_lookup, "stay_id", "label", hours, INTERVENTION_PATTERNS, interventions, observation)
            _event_rows(chunk, stay_lookup, "stay_id", "label", hours, MEDICATION_PATTERNS, medications, observation)


def _eicu_events(
    root: Path,
    cohort: pd.DataFrame,
    observation: float,
    chunksize: int,
    medications: EventAccumulator,
    interventions: EventAccumulator,
) -> None:
    lookup = {int(value): index for index, value in enumerate(cohort.patientunitstayid)}
    for filename, offset_col, text_col in (
        ("medication.csv", "drugstartoffset", "drugname"),
        ("infusionDrug.csv", "infusionoffset", "drugname"),
        ("admissionDrug.csv", "drugoffset", "drugname"),
    ):
        for chunk in _read_chunks(root / filename, ["patientunitstayid", offset_col, text_col], chunksize):
            times = pd.to_numeric(chunk[offset_col], errors="coerce").to_numpy() / 60.0
            _event_rows(chunk, lookup, "patientunitstayid", text_col, times, MEDICATION_PATTERNS, medications, observation)
    for chunk in _read_chunks(root / "treatment.csv", ["patientunitstayid", "treatmentoffset", "treatmentstring"], chunksize):
        _event_rows(
            chunk, lookup, "patientunitstayid", "treatmentstring",
            pd.to_numeric(chunk.treatmentoffset, errors="coerce").to_numpy() / 60.0,
            INTERVENTION_PATTERNS, interventions, observation,
        )
    for chunk in _read_chunks(root / "respiratoryCare.csv", ["patientunitstayid", "ventstartoffset", "airwaytype"], chunksize):
        chunk = chunk[chunk.patientunitstayid.isin(lookup)].copy()
        if chunk.empty:
            continue
        times = pd.to_numeric(chunk.ventstartoffset, errors="coerce").to_numpy() / 60.0
        concepts = np.full(len(chunk), "mechanical_ventilation", dtype=object)
        valid = np.isfinite(times) & (times >= 0) & (times < observation)
        interventions.update(chunk.patientunitstayid.map(lookup).to_numpy(np.int64)[valid], concepts[valid], times[valid])


def _patient_disjoint_split(cohort: pd.DataFrame, seed: int) -> dict[str, np.ndarray]:
    patients = cohort.groupby("patient_id", as_index=False).label.max()
    train_val, test = train_test_split(
        patients, test_size=0.20, random_state=seed, stratify=patients.label,
    )
    train, val = train_test_split(
        train_val, test_size=0.125, random_state=seed, stratify=train_val.label,
    )
    result = {}
    for name, frame in (("train", train), ("val", val), ("test", test)):
        selected = set(frame.patient_id.astype(str))
        result[name] = np.flatnonzero(cohort.patient_id.astype(str).isin(selected).to_numpy())
    return result


def _write_dataset(
    output_root: Path,
    dataset: str,
    cohort: pd.DataFrame,
    features: Mapping[str, np.ndarray],
    availability: Mapping[str, np.ndarray],
    splits: Mapping[str, np.ndarray],
) -> dict[str, object]:
    audit: dict[str, object] = {"dataset": dataset, "splits": {}}
    for split, indices in splits.items():
        destination = output_root / dataset / split
        destination.mkdir(parents=True, exist_ok=True)
        for modality, values in features.items():
            np.save(destination / f"{modality}.npy", values[indices].astype(np.float32))
        mask = np.stack([availability[name] for name in METRE_MODALITIES], axis=1)[indices].astype(np.uint8)
        labels = cohort.label.to_numpy(np.uint8)[indices]
        np.save(destination / "mask.npy", mask)
        np.save(destination / "labels.npy", labels)
        (destination / "ids.txt").write_text("\n".join(cohort.stay_key.iloc[indices].astype(str)) + "\n", encoding="utf-8")
        (destination / "patient_ids.txt").write_text("\n".join(cohort.patient_id.iloc[indices].astype(str)) + "\n", encoding="utf-8")
        audit["splits"][split] = {
            "n": int(len(indices)), "positives": int(labels.sum()), "prevalence": float(labels.mean()),
            "unique_patients": int(cohort.patient_id.iloc[indices].nunique()),
            "modality_availability": {name: float(mask[:, i].mean()) for i, name in enumerate(METRE_MODALITIES)},
        }
    (output_root / dataset / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def _validate_written_cache(output_root: Path, feature_dimensions: Mapping[str, int]) -> dict[str, object]:
    validation: dict[str, object] = {"finite_features": True, "split_checks": {}}
    mimic_patients: dict[str, set[str]] = {}
    for dataset, splits in (("mimic4", ("train", "val", "test")), ("eicu", ("test",))):
        for split in splits:
            directory = output_root / dataset / split
            labels = np.load(directory / "labels.npy", mmap_mode="r")
            mask = np.load(directory / "mask.npy", mmap_mode="r")
            ids = (directory / "ids.txt").read_text(encoding="utf-8").splitlines()
            patients = (directory / "patient_ids.txt").read_text(encoding="utf-8").splitlines()
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate stay IDs in {dataset}/{split}")
            if len(ids) != len(labels) or len(patients) != len(labels):
                raise ValueError(f"Identifier/label length mismatch in {dataset}/{split}")
            if mask.shape != (len(labels), len(METRE_MODALITIES)):
                raise ValueError(f"Invalid mask shape in {dataset}/{split}: {mask.shape}")
            if set(np.unique(labels).tolist()) != {0, 1}:
                raise ValueError(f"Both outcome classes are required in {dataset}/{split}")
            for modality, width in feature_dimensions.items():
                values = np.load(directory / f"{modality}.npy", mmap_mode="r")
                if values.shape != (len(labels), width):
                    raise ValueError(f"Invalid {modality} shape in {dataset}/{split}: {values.shape}")
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite {modality} values in {dataset}/{split}")
            validation["split_checks"][f"{dataset}/{split}"] = {
                "n": len(labels), "positives": int(labels.sum()), "unique_patients": len(set(patients)),
            }
            if dataset == "mimic4":
                mimic_patients[split] = set(patients)
    overlaps = {
        "train_val": len(mimic_patients["train"] & mimic_patients["val"]),
        "train_test": len(mimic_patients["train"] & mimic_patients["test"]),
        "val_test": len(mimic_patients["val"] & mimic_patients["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"MIMIC patient leakage across splits: {overlaps}")
    validation["mimic_patient_overlap"] = overlaps
    return validation


def _build_one(
    dataset: str,
    root: Path,
    cohort: pd.DataFrame,
    observation: float,
    chunksize: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    vitals = ContinuousAccumulator(len(cohort), VITAL_CONCEPTS)
    labs = ContinuousAccumulator(len(cohort), LAB_CONCEPTS)
    medications = EventAccumulator(len(cohort), MEDICATION_CONCEPTS)
    interventions = EventAccumulator(len(cohort), INTERVENTION_CONCEPTS)
    if dataset == "mimic4":
        _mimic_continuous(root, cohort, observation, chunksize, vitals, labs)
        _mimic_events(root, cohort, observation, chunksize, medications, interventions)
    else:
        _eicu_continuous(root, cohort, observation, chunksize, vitals, labs)
        _eicu_events(root, cohort, observation, chunksize, medications, interventions)
    vital_values, vital_mask = vitals.matrix()
    lab_values, lab_mask = labs.matrix()
    medication_values, medication_mask = medications.matrix(observation)
    intervention_values, intervention_mask = interventions.matrix(observation)
    features = {
        "demographics": _demographics(cohort), "vitals": vital_values, "labs": lab_values,
        "medications": medication_values, "interventions": intervention_values,
    }
    availability = {
        "demographics": np.ones(len(cohort), dtype=np.uint8), "vitals": vital_mask, "labs": lab_mask,
        "medications": medication_mask, "interventions": intervention_mask,
    }
    return features, availability


def main() -> None:
    args = parse_args()
    mimic_root, eicu_root, output_root = Path(args.mimic_root), Path(args.eicu_root), Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    mimic = _limit_cohort(_mimic_cohort(mimic_root, args.observation_hours, args.gap_hours, args.max_los_hours), args.max_stays, args.seed)
    eicu = _limit_cohort(_eicu_cohort(eicu_root, args.observation_hours, args.gap_hours, args.max_los_hours), args.max_stays, args.seed)
    print(json.dumps({
        "mimic4": {"n": len(mimic), "prevalence": float(mimic.label.mean())},
        "eicu": {"n": len(eicu), "prevalence": float(eicu.label.mean())},
    }), flush=True)
    mimic_features, mimic_availability = _build_one("mimic4", mimic_root, mimic, args.observation_hours, args.chunksize)
    eicu_features, eicu_availability = _build_one("eicu", eicu_root, eicu, args.observation_hours, args.chunksize)
    mimic_splits = _patient_disjoint_split(mimic, args.seed)
    eicu_splits = {"test": np.arange(len(eicu), dtype=np.int64)}
    audits = {
        "mimic4": _write_dataset(output_root, "mimic4", mimic, mimic_features, mimic_availability, mimic_splits),
        "eicu": _write_dataset(output_root, "eicu", eicu, eicu_features, eicu_availability, eicu_splits),
    }
    feature_dimensions = {name: int(values.shape[1]) for name, values in mimic_features.items()}
    validation = _validate_written_cache(output_root, feature_dimensions)
    manifest = {
        "protocol": "METRE-style local CSV extraction",
        "reference": "Liao and Voldman, Journal of Biomedical Informatics 141 (2023) 104356",
        "observation_hours": args.observation_hours,
        "prediction_gap_hours": args.gap_hours,
        "minimum_icu_los_hours": args.observation_hours + args.gap_hours,
        "maximum_icu_los_hours": args.max_los_hours,
        "label_definition": "death during the same hospital admission in both MIMIC-IV and eICU",
        "mimic_label_field": "hospital_expire_flag",
        "eicu_label_field": "hospitaldischargestatus == Expired",
        "modalities": list(METRE_MODALITIES),
        "feature_dimensions": feature_dimensions,
        "continuous_summary_features": list(SUMMARY_NAMES),
        "event_summary_features": list(EVENT_NAMES),
        "vital_concepts": list(VITAL_CONCEPTS),
        "lab_concepts": list(LAB_CONCEPTS),
        "medication_concepts": list(MEDICATION_CONCEPTS),
        "intervention_concepts": list(INTERVENTION_CONCEPTS),
        "split_policy": "MIMIC patient-disjoint 70/10/20; eICU held out in full; source-only normalization and selection",
        "seed": args.seed,
        "audits": audits,
        "validation": validation,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
