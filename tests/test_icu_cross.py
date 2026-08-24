import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from icu_cross.build_metre_cache import _event_rows
from icu_cross.build_metre_hourly_cache import (
    EICU_LAB_ALIASES,
    EXPECTED_EICU_EMPTY,
    _eicu_lab_concept,
    _alignment_audit,
    _match_event,
    _mimic_cohort,
    _mimic_gcs_score,
)
from icu_cross.data import cache_modalities, fit_source_statistics
from icu_cross.features import broad_ethnicity, code_vector, demographic_vector, lab_vector, split_codes
from icu_cross.metre_features import ContinuousAccumulator, EventAccumulator
from icu_cross.metre_hourly import (
    BEDSIDES,
    CULTURE_SITES,
    LABS,
    MEDICATIONS,
    PROCEDURES,
    BinaryHourlyAccumulator,
    HourlyAccumulator,
    impute_and_normalize,
    paired_channels,
)
from icu_cross.metre_tcn import METRETemporalConv
from icu_cross.models import MultimodalTransformer, SPMNet, SharedMETREModalityStems


class FakeVocab:
    def __call__(self, token):
        return {"a": 0, "b": 1, "c": 2}.get(token, 0)


def test_shared_semantic_mapping_and_vectors():
    mimic = SimpleNamespace(
        age=65,
        gender="F",
        ethnicity="BLACK/AFRICAN AMERICAN",
        trajectory=(("diagnoses_icd", "procedures_icd", "prescriptions"), ("a", "b", "c")),
    )
    eicu = SimpleNamespace(
        trajectory=(("diagnosis", "treatment", "medication"), ("a", "b", "c")),
    )
    assert split_codes(mimic, "mimic4") == split_codes(eicu, "eicu")
    assert demographic_vector(mimic).shape == (11,)
    assert broad_ethnicity(mimic.ethnicity) == "black"
    embeddings = np.arange(3 * 768, dtype=np.float32).reshape(3, 768)
    vector, available = code_vector(["a", "b"], FakeVocab(), embeddings)
    assert vector.shape == (769,)
    assert available == 1.0
    np.testing.assert_allclose(vector[:-1], embeddings[:2].mean(axis=0))


def test_lab_summary_is_dataset_width_invariant():
    mimic_values = np.zeros((3, 114), dtype=np.float32)
    mimic_values[:, 0] = (0.2, 0.4, 0.8)
    mimic_values[:, 1] = (0, 1, 1)
    eicu_values = np.zeros((3, 158), dtype=np.float32)
    eicu_values[:, 0] = (0.2, 0.4, 0.8)
    mimic_summary, mimic_available = lab_vector(mimic_values, "mimic4")
    eicu_summary, eicu_available = lab_vector(eicu_values, "eicu")
    assert mimic_summary.shape == eicu_summary.shape == (14,)
    assert mimic_available == eicu_available == 1.0
    assert np.isfinite(mimic_summary).all()
    assert np.isfinite(eicu_summary).all()


def test_models_forward_five_modalities():
    dims = (11, 769, 769, 769, 14)
    modalities = [torch.randn(4, dim) for dim in dims]
    mask = torch.ones(4, 5)
    spmnet = SPMNet(dims, hidden=16, latent=6, private=3)
    output = spmnet(modalities, mask, sample=False)
    assert output["logits"].shape == (4,)
    assert len(output["reconstructions"]) == 5
    transformer = MultimodalTransformer(dims, hidden=16, layers=1)
    assert transformer(modalities, mask).shape == (4,)


def test_metre_accumulators_summarize_and_filter_outliers():
    continuous = ContinuousAccumulator(2, ("heart_rate",))
    continuous.update(
        np.array([0, 0, 0, 1]),
        np.array(["heart_rate"] * 4, dtype=object),
        np.array([0.0, 1.0, 2.0, 1.0]),
        np.array([60.0, 70.0, 80.0, 500.0]),
    )
    values, available = continuous.matrix()
    assert values.shape == (2, 7)
    np.testing.assert_allclose(values[0, [0, 2, 3, 4, 6]], [70.0, 60.0, 80.0, 80.0, 10.0])
    np.testing.assert_array_equal(available, [1, 0])

    events = EventAccumulator(2, ("vasopressor",))
    events.update(np.array([0, 0]), np.array(["vasopressor", "vasopressor"]), np.array([6.0, 12.0]))
    event_values, event_available = events.matrix(48.0)
    np.testing.assert_allclose(event_values[0], [1.0, np.log1p(2), 0.125, 0.25])
    np.testing.assert_array_equal(event_available, [1, 0])


def test_dynamic_cache_modalities_and_source_statistics(tmp_path):
    modalities = ("demographics", "vitals")
    (tmp_path / "manifest.json").write_text(json.dumps({"modalities": modalities}), encoding="utf-8")
    split = tmp_path / "mimic4" / "train"
    split.mkdir(parents=True)
    np.save(split / "demographics.npy", np.array([[1.0], [3.0]], dtype=np.float32))
    np.save(split / "vitals.npy", np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32))
    np.save(split / "mask.npy", np.ones((2, 2), dtype=np.uint8))
    np.save(split / "labels.npy", np.array([0, 1], dtype=np.uint8))
    (split / "ids.txt").write_text("a\nb\n", encoding="utf-8")
    assert cache_modalities(tmp_path) == modalities
    statistics = fit_source_statistics(tmp_path)
    assert tuple(statistics["modalities"]) == modalities
    np.testing.assert_allclose(statistics["modalities"]["demographics"]["mean"], [2.0])
    np.testing.assert_allclose(statistics["modalities"]["vitals"]["mean"], [4.0, 6.0])


def test_event_filter_keeps_times_aligned_after_stay_filtering():
    accumulator = EventAccumulator(1, ("vasopressor",))
    frame = pd.DataFrame({"stay": [999, 7], "drug": ["none", "norepinephrine"]})
    _event_rows(
        frame, {7: 0}, "stay", "drug", np.array([100.0, 3.0]),
        {"vasopressor": ("norepinephrine",)}, accumulator, 48.0,
    )
    values, available = accumulator.matrix(48.0)
    np.testing.assert_allclose(values[0], [1.0, np.log1p(1), 3.0 / 48.0, 3.0 / 48.0])
    np.testing.assert_array_equal(available, [1])


def test_metre_hourly_schema_and_causal_imputation():
    accumulator = HourlyAccumulator(2, ("heart_rate",), hours=4)
    accumulator.update(
        np.array([0, 0, 0, 1]),
        np.array(["heart_rate"] * 4),
        np.array([0.1, 0.8, 2.2, 1.0]),
        np.array([60.0, 80.0, 100.0, 500.0]),
    )
    means, observed = accumulator.means_and_observed()
    np.testing.assert_allclose(means[0, [0, 2], 0], [70.0, 100.0])
    assert not observed[1].any()  # official outlier bound removes 500 bpm
    values, statistics = impute_and_normalize(means, observed, np.array([0]))
    assert statistics["mean"] == [85.0]
    np.testing.assert_allclose(values[0, :, 0], [-1.0, -1.0, 1.0, 1.0], atol=1e-6)
    assert paired_channels(values, observed).shape == (2, 4, 2)

    events = BinaryHourlyAccumulator(1, ("vent",), hours=4)
    events.update_intervals(np.array([0]), ["vent"], np.array([0.5]), np.array([2.2]))
    np.testing.assert_array_equal(events.values[0, :, 0], [1, 1, 1, 0])
    assert 2 * len(BEDSIDES) + 2 * len(LABS) + len(CULTURE_SITES) + len(MEDICATIONS) + len(PROCEDURES) == 200


def test_metre_event_names_and_eicu_lab_aliases_are_unambiguous():
    assert _match_event("Norepinephrine 8 mg") == "norepinephrine"
    assert _match_event("LEVOPHED infusion") == "norepinephrine"
    assert _match_event("Epinephrine injection") == "epinephrine"
    assert EICU_LAB_ALIASES["-basos"] == "basophils"
    assert EICU_LAB_ALIASES["-polys"] == "neutrophils"
    assert EICU_LAB_ALIASES["urinary creatinine"] == "Creatinine urine"
    assert EICU_LAB_ALIASES["wbc's in urine"] == "White blood cell count urine"
    assert _eicu_lab_concept("PEEP") == "Positive end-expiratory pressure"
    assert _eicu_lab_concept("BNP") is None
    assert _eicu_lab_concept("direct bilirubin") is None


def test_mimic_gcs_reconstruction_rules():
    assert _mimic_gcs_score(6.0, 5.0, 4.0) == 15.0
    assert _mimic_gcs_score(1.0, 0.0, 1.0) == 15.0
    assert _mimic_gcs_score(np.nan, np.nan, 3.0, (5.0, 4.0, 4.0)) == 12.0


def test_mimic_cohort_uses_first_icu_associated_admission(tmp_path):
    (tmp_path / "icu").mkdir()
    (tmp_path / "hosp").mkdir()
    pd.DataFrame({
        "subject_id": [1], "hadm_id": [20], "stay_id": [200],
        "intime": ["2020-01-01"], "outtime": ["2020-01-04"], "los": [3.0],
    }).to_csv(tmp_path / "icu/icustays.csv", index=False)
    pd.DataFrame({
        "subject_id": [1, 1], "hadm_id": [10, 20],
        "admittime": ["2019-01-01", "2020-01-01"],
        "admission_type": ["ELECTIVE", "EMERGENCY"],
        "admission_location": ["CLINIC", "ED"], "race": ["WHITE", "WHITE"],
        "hospital_expire_flag": [0, 1],
    }).to_csv(tmp_path / "hosp/admissions.csv", index=False)
    pd.DataFrame({
        "subject_id": [1], "gender": ["F"], "anchor_age": [60], "anchor_year": [2020],
    }).to_csv(tmp_path / "hosp/patients.csv", index=False)
    cohort = _mimic_cohort(tmp_path, 48, 6, 240)
    assert cohort.hadm_id.tolist() == [20]
    assert cohort.label.tolist() == [1]


def test_alignment_audit_accepts_only_official_eicu_empty_channels():
    source = {"numeric_observation_rate": {name: 1.0 for name in BEDSIDES + LABS}}
    target = {"numeric_observation_rate": {name: 1.0 for name in BEDSIDES + LABS}}
    for name in EXPECTED_EICU_EMPTY:
        target["numeric_observation_rate"][name] = 0.0
    audit = _alignment_audit(source, target)
    assert audit["passes_expected_empty_check"]
    target["numeric_observation_rate"]["heart_rate"] = 0.0
    audit = _alignment_audit(source, target)
    assert audit["unexpected_eicu_absent"] == ["heart_rate"]


def test_temporal_encoders_are_matched_across_fusion_models():
    dims = (32, 40, 144, 9, 7)
    temporal = (False, True, True, True, True)
    modalities = [torch.randn(3, dims[0])] + [torch.randn(3, 48, dim) for dim in dims[1:]]
    availability = torch.ones(3, 5)
    spmnet = SPMNet(dims, hidden=16, latent=6, private=3, temporal_modalities=temporal)
    transformer = MultimodalTransformer(dims, hidden=16, layers=1, temporal_modalities=temporal)
    assert spmnet(modalities, availability, sample=False)["logits"].shape == (3,)
    assert transformer(modalities, availability).shape == (3,)


def test_pre_normalized_hourly_cache_uses_identity_loader_stats(tmp_path):
    modalities = ("static", "bedside")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"modalities": modalities, "pre_normalized": True}), encoding="utf-8"
    )
    split = tmp_path / "mimic4" / "train"
    split.mkdir(parents=True)
    np.save(split / "static.npy", np.array([[1.0], [3.0]], dtype=np.float32))
    np.save(split / "bedside.npy", np.ones((2, 4, 2), dtype=np.float32) * 7)
    np.save(split / "mask.npy", np.ones((2, 2), dtype=np.uint8))
    np.save(split / "labels.npy", np.array([0, 1], dtype=np.uint8))
    (split / "ids.txt").write_text("a\nb\n", encoding="utf-8")
    statistics = fit_source_statistics(tmp_path)
    np.testing.assert_allclose(statistics["modalities"]["static"]["mean"], [0.0])
    np.testing.assert_allclose(statistics["modalities"]["bedside"]["std"], [1.0, 1.0])


def test_official_metre_tcn_shape_and_causality_length():
    model = METRETemporalConv(inputs=200, channels=(16, 16), kernel_size=3, dropout=0.0)
    values = torch.randn(2, 200, 48)
    logits = model(values)
    assert logits.shape == (2, 48, 2)


def test_shared_metre_encoder_preserves_modality_tokens_with_one_tcn():
    dims = (32, 40, 144, 9, 7)
    temporal = (False, True, True, True, True)
    modalities = [torch.randn(2, dims[0])] + [torch.randn(2, 48, dim) for dim in dims[1:]]
    availability = torch.ones(2, 5)
    stems = SharedMETREModalityStems(
        dims, hidden=8, temporal_modalities=temporal, channels=(16, 16)
    )
    embeddings = stems(modalities)
    assert len(embeddings) == 5
    assert all(value.shape == (2, 8) for value in embeddings)

    spmnet = SPMNet(
        dims, hidden=16, latent=6, private=3, temporal_modalities=temporal,
        encoder_kind="metre_shared", metre_channels=(16, 16),
    )
    transformer = MultimodalTransformer(
        dims, hidden=16, layers=1, temporal_modalities=temporal,
        encoder_kind="metre_shared", metre_channels=(16, 16),
    )
    assert spmnet(modalities, availability, sample=False)["logits"].shape == (2,)
    assert transformer(modalities, availability).shape == (2,)
