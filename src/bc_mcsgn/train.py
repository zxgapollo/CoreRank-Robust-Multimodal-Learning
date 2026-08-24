from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import BCDataset, BCSplit, SCMConfig, SCMParams, SPLIT_NAMES, collate_bc_batch
from .metrics import (
    binary_metrics,
    mask_recovery_metrics,
    permutation_invariant_structure_metrics,
    ridge_r2,
    shortcut_sensitivity,
    state_recovery_metrics,
)
from .models import (
    ConcatMLP,
    LateFusionMLP,
    ModelConfig,
    MultimodalTransformer,
    SFMNet,
    WarmupNet,
    diag_gaussian_kl,
    kl_standard_normal,
)


@dataclass
class TrainConfig:
    warmup_epochs: int = 40
    correction_epochs: int = 80
    baseline_epochs: int = 40
    batch_size: int = 256
    lr: float = 1e-3
    joint_lr_scale: float = 0.35
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    layers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    modality_dropout: float = 0.20
    lambda_cm: float = 0.10
    warmup_proto_weight: float = 1.0
    label_weight: float = 1.0
    full_label_weight: float = 0.50
    recon_weight: float = 1.0
    beta_z: float = 1e-3
    beta_u: float = 1e-3
    incidence_l1_weight: float = 5e-2
    task_l1_weight: float = 1e-2
    witness_weight: float = 10.0
    faithfulness_weight: float = 0.10
    gate_weight: float = 0.02
    sufficiency_weight: float = 0.25
    residual_intervention_weight: float = 0.10
    paired_intervention_weight: float = 1.0
    witness_margin: float = 0.80
    faithfulness_margin: float = 0.20
    structure_fraction: float = 0.50
    task_fraction: float = 0.20
    state_anchor_weight: float = 0.0
    fixed_masks: bool = False
    # Deprecated compatibility fields: the SFM method has no learned DAG and
    # does not consume a biased prototype.
    proto_weight: float = 0.0
    proto_source: str = "warmup"
    fixed_graph: bool = False
    graph_l1_weight: float = 0.0
    dag_weight: float = 0.0
    mask_l1_weight: float = 0.0
    delta_anchor_weight: float = 0.0
    graph_threshold: float = 0.15
    seed: int = 0
    verbose: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.structure_fraction <= 1.0 or not 0.0 <= self.task_fraction <= 1.0:
            raise ValueError("stage fractions must lie in [0, 1]")
        if self.structure_fraction + self.task_fraction > 0.95:
            raise ValueError("leave at least 5% of correction epochs for joint fine-tuning")

    def to_dict(self) -> Dict:
        return asdict(self)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(split: BCSplit, cfg: TrainConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        BCDataset(split),
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=collate_bc_batch,
    )


def _to_device(batch: Mapping, device: str) -> Dict:
    return {
        "x": [x.to(device) for x in batch["x"]],
        "x_intervened": [x.to(device) for x in batch["x_intervened"]],
        "y": batch["y"].to(device),
        "z": batch["z"].to(device),
        "s": batch["s"].to(device),
        "u": [u.to(device) for u in batch["u"]],
        "delta": batch["delta"].to(device),
        "s_tilde": batch["s_tilde"].to(device),
        "obs_mask": batch["obs_mask"].to(device),
    }


def _dropout_obs_mask(obs_mask: torch.Tensor, drop_prob: float) -> torch.Tensor:
    if drop_prob <= 0:
        return obs_mask
    keep = (torch.rand_like(obs_mask) > drop_prob).float()
    out = obs_mask * keep
    empty = out.sum(dim=1) == 0
    if empty.any():
        rows = torch.where(empty)[0]
        for row in rows:
            candidates = torch.where(obs_mask[row] > 0.5)[0]
            if candidates.numel() == 0:
                col = torch.randint(0, obs_mask.shape[1], (1,), device=obs_mask.device)
            else:
                col = candidates[torch.randint(0, candidates.numel(), (1,), device=obs_mask.device)]
            out[row, col] = 1.0
    return out


def _model_config(scfg: SCMConfig, tcfg: TrainConfig, fixed_structure: bool = False) -> ModelConfig:
    return ModelConfig(
        n_modalities=scfg.n_modalities,
        x_dim=scfg.x_dim,
        k=scfg.k,
        u_dim=scfg.u_dim,
        hidden_dim=tcfg.hidden_dim,
        layers=tcfg.layers,
        fixed_structure=fixed_structure,
    )


def _sym_kl(out_a: Dict[str, torch.Tensor], out_b: Dict[str, torch.Tensor]) -> torch.Tensor:
    kl_ab = diag_gaussian_kl(out_a["proto_mu"], out_a["proto_logvar"], out_b["proto_mu"], out_b["proto_logvar"])
    kl_ba = diag_gaussian_kl(out_b["proto_mu"], out_b["proto_logvar"], out_a["proto_mu"], out_a["proto_logvar"])
    return 0.5 * (kl_ab + kl_ba).mean()


@torch.no_grad()
def _warmup_val_loss(model: WarmupNet, split: BCSplit, tcfg: TrainConfig) -> float:
    model.eval()
    losses: List[float] = []
    for batch0 in _loader(split, tcfg, shuffle=False):
        batch = _to_device(batch0, tcfg.device)
        out = model(batch["x"], batch["obs_mask"])
        bce = F.binary_cross_entropy_with_logits(out["logits"], batch["y"])
        proto = F.mse_loss(out["proto_mu"], batch["s_tilde"])
        losses.append(float((bce + tcfg.warmup_proto_weight * proto).detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def train_warmup(train: BCSplit, val: BCSplit, scfg: SCMConfig, tcfg: TrainConfig) -> WarmupNet:
    model = WarmupNet(_model_config(scfg, tcfg)).to(tcfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    best = None
    best_val = float("inf")
    for _ in tqdm(range(tcfg.warmup_epochs), desc="warmup", leave=False, disable=not tcfg.verbose):
        model.train()
        for batch0 in _loader(train, tcfg, shuffle=True):
            batch = _to_device(batch0, tcfg.device)
            mask_a = _dropout_obs_mask(batch["obs_mask"], tcfg.modality_dropout)
            mask_b = _dropout_obs_mask(batch["obs_mask"], tcfg.modality_dropout)
            out_a = model(batch["x"], mask_a)
            out_b = model(batch["x"], mask_b)
            out_full = model(batch["x"], batch["obs_mask"])
            bce = (
                F.binary_cross_entropy_with_logits(out_a["logits"], batch["y"])
                + F.binary_cross_entropy_with_logits(out_b["logits"], batch["y"])
                + F.binary_cross_entropy_with_logits(out_full["logits"], batch["y"])
            ) / 3.0
            loss = bce + tcfg.lambda_cm * _sym_kl(out_a, out_b) + tcfg.warmup_proto_weight * F.mse_loss(
                out_full["proto_mu"], batch["s_tilde"]
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val_loss = _warmup_val_loss(model, val, tcfg)
        if val_loss < best_val:
            best_val = val_loss
            best = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    if best is not None:
        model.load_state_dict(best)
    return model


@torch.no_grad()
def _classifier_val_loss(model: torch.nn.Module, split: BCSplit, tcfg: TrainConfig) -> float:
    model.eval()
    losses: List[float] = []
    for batch0 in _loader(split, tcfg, shuffle=False):
        batch = _to_device(batch0, tcfg.device)
        logits = model(batch["x"], batch["obs_mask"])
        losses.append(float(F.binary_cross_entropy_with_logits(logits, batch["y"]).detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def train_classifier(
    model: torch.nn.Module,
    train: BCSplit,
    val: BCSplit,
    tcfg: TrainConfig,
    name: str,
    paired: bool = False,
) -> torch.nn.Module:
    model.to(tcfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    best = None
    best_val = float("inf")
    for _ in tqdm(range(tcfg.baseline_epochs), desc=name, leave=False, disable=not tcfg.verbose):
        model.train()
        for batch0 in _loader(train, tcfg, shuffle=True):
            batch = _to_device(batch0, tcfg.device)
            mask = _dropout_obs_mask(batch["obs_mask"], tcfg.modality_dropout)
            logits = model(batch["x"], mask)
            loss = F.binary_cross_entropy_with_logits(logits, batch["y"])
            if paired:
                logits_intervened = model(batch["x_intervened"], mask)
                loss = 0.5 * (loss + F.binary_cross_entropy_with_logits(logits_intervened, batch["y"]))
                loss = loss + tcfg.lambda_cm * F.mse_loss(torch.sigmoid(logits), torch.sigmoid(logits_intervened))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val_loss = _classifier_val_loss(model, val, tcfg)
        if val_loss < best_val:
            best_val = val_loss
            best = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    if best is not None:
        model.load_state_dict(best)
    return model


def _masked_reconstruction(out: Dict, targets: List[torch.Tensor], obs_mask: torch.Tensor) -> torch.Tensor:
    recon = targets[0].new_zeros(())
    denom = obs_mask.sum().clamp_min(1.0)
    for m, rec in enumerate(out["recon"]):
        mse = F.mse_loss(rec, targets[m], reduction="none").mean(dim=-1)
        recon = recon + (mse * obs_mask[:, m]).sum() / denom
    return recon


def _cross_mask_consistency(model: SFMNet, batch: Dict, out: Dict, obs_mask: torch.Tensor, tcfg: TrainConfig) -> torch.Tensor:
    mask_b = _dropout_obs_mask(batch["obs_mask"], tcfg.modality_dropout)
    out_b = model(batch["x"], mask_b, sample=False)
    r = model.task_mask()[None, :]
    return (((out_b["z_mu"] - out["z_mu"]).pow(2)) * r).sum() / (r.sum() * obs_mask.shape[0]).clamp_min(1e-8)


def _sfm_loss(model: SFMNet, batch: Dict, obs_mask: torch.Tensor, tcfg: TrainConfig, stage: str) -> torch.Tensor:
    out = model(batch["x"], obs_mask, sample=True)
    recon = _masked_reconstruction(out, batch["x"], obs_mask)
    out_intervened = model(batch["x_intervened"], obs_mask, sample=False)
    recon_intervened = _masked_reconstruction(out_intervened, batch["x_intervened"], obs_mask)
    paired_intervention = F.mse_loss(out["z_mu"], out_intervened["z_mu"])
    kl_z = kl_standard_normal(out["z_mu"], out["z_logvar"]).mean()
    kl_u = torch.stack([kl_standard_normal(mu, lv).mean() for mu, lv in zip(out["u_mu"], out["u_logvar"])]).mean()
    incidence_sparse = model.incidence().mean()
    task_sparse = model.task_mask().mean()
    gate = model.gate_regularizer()
    faithfulness = model.faithfulness_loss(margin=tcfg.faithfulness_margin)

    if stage == "structure":
        return (
            tcfg.recon_weight * (recon + 0.5 * recon_intervened)
            + tcfg.beta_z * kl_z
            + tcfg.beta_u * kl_u
            + tcfg.incidence_l1_weight * incidence_sparse
            + tcfg.faithfulness_weight * faithfulness
            + tcfg.gate_weight * gate
            + tcfg.paired_intervention_weight * paired_intervention
        )

    label = F.binary_cross_entropy_with_logits(out["logits"], batch["y"])
    full_label = F.binary_cross_entropy_with_logits(out["full_logits"], batch["y"])
    teacher = torch.sigmoid(out["full_logits"]).detach()
    sufficiency = F.binary_cross_entropy_with_logits(out["logits"], teacher)
    state_anchor = (
        F.mse_loss(out["s"], batch["s"])
        if tcfg.state_anchor_weight > 0
        else batch["y"].new_zeros(())
    )

    if stage == "task":
        return (
            tcfg.label_weight * label
            + tcfg.full_label_weight * full_label
            + tcfg.sufficiency_weight * sufficiency
            + tcfg.task_l1_weight * task_sparse
            + tcfg.gate_weight * gate
            + tcfg.state_anchor_weight * state_anchor
        )

    consistency = _cross_mask_consistency(model, batch, out, obs_mask, tcfg)
    residual = model.residual_intervention_loss(out["z_mu"], out["u_mu"], obs_mask)
    return (
        tcfg.label_weight * label
        + tcfg.full_label_weight * full_label
        + tcfg.recon_weight * recon
        + tcfg.beta_z * kl_z
        + tcfg.beta_u * kl_u
        + tcfg.incidence_l1_weight * incidence_sparse
        + tcfg.task_l1_weight * task_sparse
        + tcfg.witness_weight * model.witness_loss(margin=tcfg.witness_margin)
        + tcfg.faithfulness_weight * faithfulness
        + tcfg.gate_weight * gate
        + tcfg.sufficiency_weight * sufficiency
        + tcfg.residual_intervention_weight * residual
        + tcfg.paired_intervention_weight * paired_intervention
        + tcfg.lambda_cm * consistency
        + tcfg.state_anchor_weight * state_anchor
    )


def _set_stage_trainable(model: SFMNet, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = stage == "joint"
    if stage == "structure":
        for module in (model.content_encoders, model.private_encoders, model.decoders):
            for parameter in module.parameters():
                parameter.requires_grad = True
        if hasattr(model, "incidence_logits"):
            model.incidence_logits.requires_grad = True
    elif stage == "task":
        for module in (model.selected_classifier, model.full_classifier):
            for parameter in module.parameters():
                parameter.requires_grad = True
        if hasattr(model, "task_mask_logits"):
            model.task_mask_logits.requires_grad = True


def _stage_schedule(tcfg: TrainConfig) -> List[tuple[str, int]]:
    n = tcfg.correction_epochs
    if n <= 0:
        return []
    if n == 1:
        return [("joint", 1)]
    if n == 2:
        return [("structure", 1), ("joint", 1)]
    structure = max(1, int(round(n * tcfg.structure_fraction)))
    task = max(1, int(round(n * tcfg.task_fraction)))
    if structure + task >= n:
        structure = max(1, n - task - 1)
    joint = n - structure - task
    return [("structure", structure), ("task", task), ("joint", joint)]


@torch.no_grad()
def _sfm_val_loss(model: SFMNet, split: BCSplit, tcfg: TrainConfig, stage: str) -> float:
    model.eval()
    losses: List[float] = []
    for batch0 in _loader(split, tcfg, shuffle=False):
        batch = _to_device(batch0, tcfg.device)
        out = model(batch["x"], batch["obs_mask"], sample=False)
        if stage == "structure":
            loss = _masked_reconstruction(out, batch["x"], batch["obs_mask"])
        else:
            loss = F.binary_cross_entropy_with_logits(out["logits"], batch["y"])
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def train_sfm(
    train: BCSplit,
    val: BCSplit,
    params: SCMParams,
    scfg: SCMConfig,
    tcfg: TrainConfig,
    fixed_structure: bool = False,
) -> SFMNet:
    model = SFMNet(
        _model_config(scfg, tcfg, fixed_structure=fixed_structure),
        true_task_mask=torch.tensor(params.state_mask, device=tcfg.device),
        true_incidence=torch.tensor(params.modality_mask, device=tcfg.device),
    ).to(tcfg.device)
    best = None
    best_val = float("inf")
    for stage, epochs in _stage_schedule(tcfg):
        _set_stage_trainable(model, stage)
        trainable = [p for p in model.parameters() if p.requires_grad]
        lr = tcfg.lr * (tcfg.joint_lr_scale if stage == "joint" else 1.0)
        opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=tcfg.weight_decay)
        for _ in tqdm(range(epochs), desc=f"sfm-{stage}", leave=False, disable=not tcfg.verbose):
            model.train()
            for batch0 in _loader(train, tcfg, shuffle=True):
                batch = _to_device(batch0, tcfg.device)
                obs_mask = _dropout_obs_mask(batch["obs_mask"], tcfg.modality_dropout)
                loss = _sfm_loss(model, batch, obs_mask, tcfg, stage)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 5.0)
                opt.step()
            val_loss = _sfm_val_loss(model, val, tcfg, stage)
            if stage != "structure" and val_loss < best_val:
                best_val = val_loss
                best = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    if best is not None:
        model.load_state_dict(best)
    _set_stage_trainable(model, "joint")
    return model


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    split: BCSplit,
    params: SCMParams,
    tcfg: TrainConfig,
) -> Dict[str, np.ndarray]:
    model.eval()
    probs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    states: List[np.ndarray] = []
    zs: List[np.ndarray] = []
    obs_values: List[np.ndarray] = []
    for batch0 in _loader(split, tcfg, shuffle=False):
        batch = _to_device(batch0, tcfg.device)
        logits = model(batch["x"], batch["obs_mask"])
        probs.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(batch["y"].cpu().numpy())
        states.append(batch["s"].cpu().numpy())
        zs.append(batch["z"].cpu().numpy())
        obs_values.append(batch["obs_mask"].cpu().numpy())
    obs = np.concatenate(obs_values)
    return {
        "prob": np.concatenate(probs),
        "y": np.concatenate(ys),
        "s_true": np.concatenate(states),
        "z_true": np.concatenate(zs),
        "rep": None,
        "true_certificate": _numpy_certificate(obs, params.modality_mask, params.state_mask),
    }


@torch.no_grad()
def evaluate_warmup(model: WarmupNet, split: BCSplit, params: SCMParams, tcfg: TrainConfig) -> Dict[str, np.ndarray]:
    model.eval()
    probs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    states: List[np.ndarray] = []
    zs: List[np.ndarray] = []
    reps: List[np.ndarray] = []
    obs_values: List[np.ndarray] = []
    for batch0 in _loader(split, tcfg, shuffle=False):
        batch = _to_device(batch0, tcfg.device)
        out = model(batch["x"], batch["obs_mask"])
        probs.append(torch.sigmoid(out["logits"]).cpu().numpy())
        ys.append(batch["y"].cpu().numpy())
        states.append(batch["s"].cpu().numpy())
        zs.append(batch["z"].cpu().numpy())
        reps.append(out["proto_mu"].cpu().numpy())
        obs_values.append(batch["obs_mask"].cpu().numpy())
    obs = np.concatenate(obs_values)
    return {
        "prob": np.concatenate(probs),
        "y": np.concatenate(ys),
        "s_true": np.concatenate(states),
        "z_true": np.concatenate(zs),
        "rep": np.concatenate(reps),
        "true_certificate": _numpy_certificate(obs, params.modality_mask, params.state_mask),
    }


def _numpy_certificate(obs: np.ndarray, incidence: np.ndarray, task: np.ndarray) -> np.ndarray:
    b = (incidence >= 0.5).astype(np.float32)
    r = task >= 0.5
    if not r.any():
        return np.zeros(obs.shape[0], dtype=np.float32)
    pair = b[:, :, None] * (1.0 - b[:, None, :])
    available = obs[:, :, None, None] * pair[None, :, :, :]
    pair_cert = available.max(axis=1)
    eye = np.eye(b.shape[1], dtype=bool)
    factor = np.where(eye[None, :, :], 1.0, pair_cert).min(axis=-1)
    return factor[:, r].min(axis=-1).astype(np.float32)


@torch.no_grad()
def evaluate_sfm(model: SFMNet, split: BCSplit, params: SCMParams, tcfg: TrainConfig) -> Dict[str, np.ndarray | float]:
    model.eval()
    probs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    states: List[np.ndarray] = []
    zs: List[np.ndarray] = []
    reps: List[np.ndarray] = []
    z_hats: List[np.ndarray] = []
    certs: List[np.ndarray] = []
    obs_values: List[np.ndarray] = []
    residual_losses: List[float] = []
    intervened_probs: List[np.ndarray] = []
    paired_state_mse: List[float] = []
    for batch0 in _loader(split, tcfg, shuffle=False):
        batch = _to_device(batch0, tcfg.device)
        out = model(batch["x"], batch["obs_mask"], sample=False)
        out_intervened = model(batch["x_intervened"], batch["obs_mask"], sample=False)
        probs.append(torch.sigmoid(out["logits"]).cpu().numpy())
        ys.append(batch["y"].cpu().numpy())
        states.append(batch["s"].cpu().numpy())
        zs.append(batch["z"].cpu().numpy())
        reps.append(out["s"].cpu().numpy())
        z_hats.append(out["z_mu"].cpu().numpy())
        certs.append(out["certificate"].cpu().numpy())
        obs_values.append(batch["obs_mask"].cpu().numpy())
        residual_losses.append(float(model.residual_intervention_loss(out["z_mu"], out["u_mu"], batch["obs_mask"]).cpu()))
        intervened_probs.append(torch.sigmoid(out_intervened["logits"]).cpu().numpy())
        paired_state_mse.append(float(F.mse_loss(out["s"], out_intervened["s"]).cpu()))
    obs = np.concatenate(obs_values)
    return {
        "prob": np.concatenate(probs),
        "y": np.concatenate(ys),
        "s_true": np.concatenate(states),
        "z_true": np.concatenate(zs),
        "rep": np.concatenate(reps),
        "z_hat": np.concatenate(z_hats),
        "certificate": np.concatenate(certs),
        "true_certificate": _numpy_certificate(obs, params.modality_mask, params.state_mask),
        "residual_cf_mse": float(np.mean(residual_losses)),
        "intervened_prob": np.concatenate(intervened_probs),
        "paired_state_mse": float(np.mean(paired_state_mse)),
    }


def metrics_from_eval(method: str, split_name: str, out: Dict) -> Dict[str, float | str]:
    row: Dict[str, float | str] = {"method": method, "split": split_name}
    row.update(binary_metrics(out["y"], out["prob"]))
    row.update(state_recovery_metrics(out.get("rep"), out["s_true"]))
    relevant = np.flatnonzero(np.asarray(out["s_true"]).std(axis=0) > 1e-8)
    row.update(state_recovery_metrics(out.get("z_hat"), out["z_true"][:, relevant], prefix="relevant_factor"))
    row["shortcut_sensitivity"] = shortcut_sensitivity(out["prob"], out["s_true"])
    row["residual_cf_mse"] = float(out.get("residual_cf_mse", float("nan")))
    row["paired_state_mse"] = float(out.get("paired_state_mse", float("nan")))
    if "intervened_prob" in out:
        paired_metrics = binary_metrics(out["y"], out["intervened_prob"])
        row["intervened_auc"] = paired_metrics["auc"]
        row["prediction_intervention_mse"] = float(
            np.mean((np.asarray(out["prob"]) - np.asarray(out["intervened_prob"])) ** 2)
        )
    else:
        row["intervened_auc"] = float("nan")
        row["prediction_intervention_mse"] = float("nan")
    if "true_certificate" in out:
        true_cert = np.asarray(out["true_certificate"]).reshape(-1)
        y = np.asarray(out["y"])
        prob = np.asarray(out["prob"])
        true_certified = true_cert > 0.5
        row["true_certificate_rate"] = float(true_certified.mean())
        if true_certified.sum() >= 20 and np.unique(y[true_certified]).size > 1:
            row["true_certified_auc"] = binary_metrics(y[true_certified], prob[true_certified])["auc"]
        else:
            row["true_certified_auc"] = float("nan")
        true_uncertified = ~true_certified
        if true_uncertified.sum() >= 20 and np.unique(y[true_uncertified]).size > 1:
            row["true_uncertified_auc"] = binary_metrics(y[true_uncertified], prob[true_uncertified])["auc"]
        else:
            row["true_uncertified_auc"] = float("nan")
    else:
        row["true_certificate_rate"] = float("nan")
        row["true_certified_auc"] = float("nan")
        row["true_uncertified_auc"] = float("nan")

    if "certificate" in out:
        cert = np.asarray(out["certificate"]).reshape(-1)
        row["certificate_mean"] = float(cert.mean())
        row["certificate_rate"] = float((cert > 0.5).mean())
        certified = cert > 0.5
        if certified.sum() >= 20 and np.unique(np.asarray(out["y"])[certified]).size > 1:
            row["certified_auc"] = binary_metrics(np.asarray(out["y"])[certified], np.asarray(out["prob"])[certified])["auc"]
        else:
            row["certified_auc"] = float("nan")
    else:
        row["certificate_mean"] = float("nan")
        row["certificate_rate"] = float("nan")
        row["certified_auc"] = float("nan")
    return row


def evaluate_all_splits(method: str, outputs: Mapping[str, Dict]) -> List[Dict[str, float | str]]:
    return [metrics_from_eval(method, split_name, outputs[split_name]) for split_name in SPLIT_NAMES]


def _add_structure_metrics(rows: List[Dict[str, float | str]], model: SFMNet, params: SCMParams) -> None:
    b_hat = model.incidence().detach().cpu().numpy()
    r_hat = model.task_mask().detach().cpu().numpy()
    metrics = permutation_invariant_structure_metrics(b_hat, r_hat, params.modality_mask, params.state_mask)
    metrics.update(mask_recovery_metrics(b_hat, params.modality_mask, prefix="incidence_direct"))
    metrics.update(mask_recovery_metrics(r_hat, params.state_mask, prefix="task_direct"))
    metrics["witness_loss"] = float(model.witness_loss().detach().cpu())
    strength = model.decoder_edge_strength().detach().cpu().numpy()
    active = b_hat >= 0.5
    metrics["active_edge_strength"] = float(strength[active].mean()) if active.any() else float("nan")
    for row in rows:
        row.update(metrics)


def train_and_evaluate(
    splits: Mapping[str, BCSplit],
    params: SCMParams,
    scfg: SCMConfig,
    tcfg: TrainConfig,
    output_dir: str,
    methods: Sequence[str] = ("concat", "warmup", "sfm_net", "sfm_oracle"),
) -> List[Dict[str, float | str]]:
    os.makedirs(output_dir, exist_ok=True)
    rows: List[Dict[str, float | str]] = []
    mcfg = _model_config(scfg, tcfg)

    warmup = None
    if "warmup" in methods:
        _set_seed(tcfg.seed + 104)
        warmup = train_warmup(splits["train"], splits["val"], scfg, tcfg)
        torch.save(warmup.state_dict(), os.path.join(output_dir, "warmup.pt"))

    if "concat" in methods:
        _set_seed(tcfg.seed + 101)
        concat = train_classifier(ConcatMLP(mcfg), splits["train"], splits["val"], tcfg, name="concat")
        torch.save(concat.state_dict(), os.path.join(output_dir, "concat.pt"))
        rows.extend(
            evaluate_all_splits(
                "concat",
                {name: evaluate_classifier(concat, split, params, tcfg) for name, split in splits.items()},
            )
        )

    if "concat_paired" in methods:
        _set_seed(tcfg.seed + 101)
        concat_paired = train_classifier(
            ConcatMLP(mcfg), splits["train"], splits["val"], tcfg, name="concat_paired", paired=True
        )
        torch.save(concat_paired.state_dict(), os.path.join(output_dir, "concat_paired.pt"))
        rows.extend(
            evaluate_all_splits(
                "concat_paired",
                {name: evaluate_classifier(concat_paired, split, params, tcfg) for name, split in splits.items()},
            )
        )

    if "late_fusion" in methods:
        _set_seed(tcfg.seed + 103)
        late = train_classifier(LateFusionMLP(mcfg), splits["train"], splits["val"], tcfg, name="late_fusion")
        torch.save(late.state_dict(), os.path.join(output_dir, "late_fusion.pt"))
        rows.extend(
            evaluate_all_splits(
                "late_fusion",
                {name: evaluate_classifier(late, split, params, tcfg) for name, split in splits.items()},
            )
        )

    if "multimodal_transformer" in methods or "transformer" in methods:
        _set_seed(tcfg.seed + 105)
        transformer = train_classifier(
            MultimodalTransformer(mcfg),
            splits["train"],
            splits["val"],
            tcfg,
            name="multimodal_transformer",
        )
        torch.save(transformer.state_dict(), os.path.join(output_dir, "multimodal_transformer.pt"))
        rows.extend(
            evaluate_all_splits(
                "multimodal_transformer",
                {
                    name: evaluate_classifier(transformer, split, params, tcfg)
                    for name, split in splits.items()
                },
            )
        )

    if warmup is not None:
        rows.extend(
            evaluate_all_splits(
                "warmup",
                {name: evaluate_warmup(warmup, split, params, tcfg) for name, split in splits.items()},
            )
        )

    learned_requested = "sfm_net" in methods or "bc_mcsgn" in methods
    if learned_requested:
        _set_seed(tcfg.seed + 201)
        sfm = train_sfm(splits["train"], splits["val"], params, scfg, tcfg, fixed_structure=tcfg.fixed_masks)
        torch.save(sfm.state_dict(), os.path.join(output_dir, "sfm_net.pt"))
        sfm_rows = evaluate_all_splits(
            "sfm_net", {name: evaluate_sfm(sfm, split, params, tcfg) for name, split in splits.items()}
        )
        _add_structure_metrics(sfm_rows, sfm, params)
        rows.extend(sfm_rows)

    if "sfm_self" in methods:
        _set_seed(tcfg.seed + 201)
        self_cfg = replace(tcfg, paired_intervention_weight=0.0)
        sfm_self = train_sfm(splits["train"], splits["val"], params, scfg, self_cfg, fixed_structure=False)
        torch.save(sfm_self.state_dict(), os.path.join(output_dir, "sfm_self.pt"))
        self_rows = evaluate_all_splits(
            "sfm_self", {name: evaluate_sfm(sfm_self, split, params, self_cfg) for name, split in splits.items()}
        )
        _add_structure_metrics(self_rows, sfm_self, params)
        rows.extend(self_rows)

    if "sfm_self_oracle" in methods:
        _set_seed(tcfg.seed + 202)
        self_cfg = replace(tcfg, paired_intervention_weight=0.0)
        self_oracle = train_sfm(splits["train"], splits["val"], params, scfg, self_cfg, fixed_structure=True)
        torch.save(self_oracle.state_dict(), os.path.join(output_dir, "sfm_self_oracle.pt"))
        self_oracle_rows = evaluate_all_splits(
            "sfm_self_oracle", {name: evaluate_sfm(self_oracle, split, params, self_cfg) for name, split in splits.items()}
        )
        _add_structure_metrics(self_oracle_rows, self_oracle, params)
        rows.extend(self_oracle_rows)

    if "sfm_oracle" in methods:
        _set_seed(tcfg.seed + 202)
        oracle = train_sfm(splits["train"], splits["val"], params, scfg, tcfg, fixed_structure=True)
        torch.save(oracle.state_dict(), os.path.join(output_dir, "sfm_oracle.pt"))
        oracle_rows = evaluate_all_splits(
            "sfm_oracle", {name: evaluate_sfm(oracle, split, params, tcfg) for name, split in splits.items()}
        )
        _add_structure_metrics(oracle_rows, oracle, params)
        rows.extend(oracle_rows)

    return rows
