from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import ISODataConfig, ISOParams, ISODataset, ISOSplit, collate_iso_batch
from .diagnostics import diagnostics_for_subset, modality_subsets, subset_name
from .metrics import binary_metrics, state_recovery_metrics
from .models import ConcatMLP, ISOPoE, LateFusionMLP, ModelConfig, OracleStateMLP, UnimodalMLP, kl_standard_normal


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    modality_dropout: float = 0.15
    recon_weight: float = 0.25
    beta_kl: float = 1e-3
    state_anchor_weight: float = 0.0
    seed: int = 0
    verbose: bool = False


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: Dict, device: str) -> Dict:
    return {
        "x": [xm.to(device) for xm in batch["x"]],
        "y": batch["y"].to(device),
        "s": batch["s"].to(device),
        "u": [um.to(device) for um in batch["u"]],
        "q": batch["q"].to(device),
    }


def _fixed_obs_mask(batch_size: int, n_modalities: int, subset: Sequence[int], device: str) -> torch.Tensor:
    mask = torch.zeros(batch_size, n_modalities, device=device)
    mask[:, list(subset)] = 1.0
    return mask


def _dropout_obs_mask(batch_size: int, n_modalities: int, device: str, drop_prob: float) -> torch.Tensor:
    if drop_prob <= 0:
        return torch.ones(batch_size, n_modalities, device=device)
    mask = (torch.rand(batch_size, n_modalities, device=device) > drop_prob).float()
    empty = mask.sum(dim=1) == 0
    if empty.any():
        rows = torch.where(empty)[0]
        cols = torch.randint(0, n_modalities, (rows.numel(),), device=device)
        mask[rows, cols] = 1.0
    return mask


def _loader(split: ISOSplit, tcfg: TrainConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        ISODataset(split),
        batch_size=tcfg.batch_size,
        shuffle=shuffle,
        collate_fn=collate_iso_batch,
        drop_last=False,
    )


@torch.no_grad()
def _validation_nll_classifier(model: torch.nn.Module, val: ISOSplit, scfg: ISODataConfig, tcfg: TrainConfig, subset: Sequence[int]) -> float:
    model.eval()
    losses: List[float] = []
    for batch0 in _loader(val, tcfg, shuffle=False):
        batch = _to_device(batch0, tcfg.device)
        mask = _fixed_obs_mask(batch["y"].shape[0], scfg.n_modalities, subset, tcfg.device)
        logits = model(batch["x"], mask)
        loss = F.binary_cross_entropy_with_logits(logits, batch["y"])
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def train_classifier(
    model: torch.nn.Module,
    train: ISOSplit,
    val: ISOSplit,
    scfg: ISODataConfig,
    tcfg: TrainConfig,
    subset: Sequence[int],
) -> torch.nn.Module:
    model.to(tcfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    best_state = None
    best_val = float("inf")
    for _ in tqdm(range(tcfg.epochs), desc=f"train {model.__class__.__name__}", leave=False, disable=not tcfg.verbose):
        model.train()
        for batch0 in _loader(train, tcfg, shuffle=True):
            batch = _to_device(batch0, tcfg.device)
            mask = _fixed_obs_mask(batch["y"].shape[0], scfg.n_modalities, subset, tcfg.device)
            opt.zero_grad(set_to_none=True)
            logits = model(batch["x"], mask)
            loss = F.binary_cross_entropy_with_logits(logits, batch["y"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val_nll = _validation_nll_classifier(model, val, scfg, tcfg, subset)
        if val_nll < best_val:
            best_val = val_nll
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _validation_nll_oracle(model: OracleStateMLP, val: ISOSplit, tcfg: TrainConfig) -> float:
    model.eval()
    losses: List[float] = []
    for batch0 in _loader(val, tcfg, shuffle=False):
        batch = _to_device(batch0, tcfg.device)
        logits = model(batch["s"])
        loss = F.binary_cross_entropy_with_logits(logits, batch["y"])
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def train_oracle(model: OracleStateMLP, train: ISOSplit, val: ISOSplit, tcfg: TrainConfig) -> OracleStateMLP:
    model.to(tcfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    best_state = None
    best_val = float("inf")
    for _ in tqdm(range(tcfg.epochs), desc="train oracle", leave=False, disable=not tcfg.verbose):
        model.train()
        for batch0 in _loader(train, tcfg, shuffle=True):
            batch = _to_device(batch0, tcfg.device)
            opt.zero_grad(set_to_none=True)
            logits = model(batch["s"])
            loss = F.binary_cross_entropy_with_logits(logits, batch["y"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val_nll = _validation_nll_oracle(model, val, tcfg)
        if val_nll < best_val:
            best_val = val_nll
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _poe_loss(model: ISOPoE, batch: Dict, obs_mask: torch.Tensor, scfg: ISODataConfig, tcfg: TrainConfig) -> torch.Tensor:
    out = model(batch["x"], obs_mask, sample=True)
    bce = F.binary_cross_entropy_with_logits(out["logits"], batch["y"])
    recon = torch.tensor(0.0, device=batch["y"].device)
    for m, rec in enumerate(out["recon"]):
        mse = F.mse_loss(rec, batch["x"][m], reduction="none").mean(dim=-1)
        noise_std = scfg.noisy_noise_std if scfg.scenario == "noisy_modality" and m == scfg.noisy_modality else scfg.noise_std
        recon = recon + (obs_mask[:, m] * mse / (2.0 * float(noise_std**2))).mean()
    kl = kl_standard_normal(out["s_mu"], out["s_logvar"]).mean()
    state_anchor = F.mse_loss(out["s_mu"], batch["s"]) if tcfg.state_anchor_weight > 0 else torch.tensor(0.0, device=batch["y"].device)
    return bce + tcfg.recon_weight * recon + tcfg.beta_kl * kl + tcfg.state_anchor_weight * state_anchor


@torch.no_grad()
def _validation_nll_poe(model: ISOPoE, val: ISOSplit, scfg: ISODataConfig, tcfg: TrainConfig) -> float:
    model.eval()
    losses: List[float] = []
    subset = tuple(range(scfg.n_modalities))
    for batch0 in _loader(val, tcfg, shuffle=False):
        batch = _to_device(batch0, tcfg.device)
        mask = _fixed_obs_mask(batch["y"].shape[0], scfg.n_modalities, subset, tcfg.device)
        out = model(batch["x"], mask, sample=False)
        losses.append(float(F.binary_cross_entropy_with_logits(out["logits"], batch["y"]).detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def train_poe(model: ISOPoE, train: ISOSplit, val: ISOSplit, scfg: ISODataConfig, tcfg: TrainConfig) -> ISOPoE:
    model.to(tcfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    best_state = None
    best_val = float("inf")
    for _ in tqdm(range(tcfg.epochs), desc="train ISO-PoE", leave=False, disable=not tcfg.verbose):
        model.train()
        for batch0 in _loader(train, tcfg, shuffle=True):
            batch = _to_device(batch0, tcfg.device)
            mask = _dropout_obs_mask(batch["y"].shape[0], scfg.n_modalities, tcfg.device, tcfg.modality_dropout)
            opt.zero_grad(set_to_none=True)
            loss = _poe_loss(model, batch, mask, scfg, tcfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val_nll = _validation_nll_poe(model, val, scfg, tcfg)
        if val_nll < best_val:
            best_val = val_nll
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_classifier(
    model: torch.nn.Module,
    split: ISOSplit,
    scfg: ISODataConfig,
    tcfg: TrainConfig,
    subset: Sequence[int],
) -> Dict[str, np.ndarray]:
    model.eval()
    probs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    reps: List[np.ndarray] = []
    states: List[np.ndarray] = []
    for batch0 in _loader(split, tcfg, shuffle=False):
        batch = _to_device(batch0, tcfg.device)
        mask = _fixed_obs_mask(batch["y"].shape[0], scfg.n_modalities, subset, tcfg.device)
        logits = model(batch["x"], mask)
        rep = model.representation(batch["x"], mask) if hasattr(model, "representation") else None
        probs.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(batch["y"].cpu().numpy())
        states.append(batch["s"].cpu().numpy())
        if rep is not None:
            reps.append(rep.cpu().numpy())
    return {
        "prob": np.concatenate(probs, axis=0),
        "y": np.concatenate(ys, axis=0),
        "s_true": np.concatenate(states, axis=0),
        "rep": np.concatenate(reps, axis=0) if reps else None,
    }


@torch.no_grad()
def evaluate_poe(model: ISOPoE, split: ISOSplit, scfg: ISODataConfig, tcfg: TrainConfig, subset: Sequence[int]) -> Dict[str, np.ndarray]:
    model.eval()
    probs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    reps: List[np.ndarray] = []
    states: List[np.ndarray] = []
    precisions: List[np.ndarray] = []
    for batch0 in _loader(split, tcfg, shuffle=False):
        batch = _to_device(batch0, tcfg.device)
        mask = _fixed_obs_mask(batch["y"].shape[0], scfg.n_modalities, subset, tcfg.device)
        out = model(batch["x"], mask, sample=False)
        expert_precision = torch.stack([torch.exp(-lv).mean(dim=-1) for lv in out["expert_logvar"]], dim=1)
        probs.append(torch.sigmoid(out["logits"]).cpu().numpy())
        ys.append(batch["y"].cpu().numpy())
        reps.append(out["s_mu"].cpu().numpy())
        states.append(batch["s"].cpu().numpy())
        precisions.append(expert_precision.cpu().numpy())
    precision_mean = np.concatenate(precisions, axis=0).mean(axis=0)
    return {
        "prob": np.concatenate(probs, axis=0),
        "y": np.concatenate(ys, axis=0),
        "s_true": np.concatenate(states, axis=0),
        "rep": np.concatenate(reps, axis=0),
        "expert_precision": precision_mean,
    }


@torch.no_grad()
def evaluate_oracle(model: OracleStateMLP, split: ISOSplit, tcfg: TrainConfig) -> Dict[str, np.ndarray]:
    model.eval()
    probs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    states: List[np.ndarray] = []
    for batch0 in _loader(split, tcfg, shuffle=False):
        batch = _to_device(batch0, tcfg.device)
        logits = model(batch["s"])
        probs.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(batch["y"].cpu().numpy())
        states.append(batch["s"].cpu().numpy())
    s_true = np.concatenate(states, axis=0)
    return {
        "prob": np.concatenate(probs, axis=0),
        "y": np.concatenate(ys, axis=0),
        "s_true": s_true,
        "rep": s_true,
    }


def _row_from_eval(
    pred: Dict[str, np.ndarray],
    model_name: str,
    split_name: str,
    scenario: str,
    seed: int,
    n_train: int,
    subset: Sequence[int] | str,
    diag: Optional[Dict],
) -> Dict:
    row: Dict = {
        "scenario": scenario,
        "seed": seed,
        "n_train": n_train,
        "model": model_name,
        "split": split_name,
        "subset": subset if isinstance(subset, str) else subset_name(subset),
        "subset_size": 0 if isinstance(subset, str) else len(subset),
    }
    row.update(binary_metrics(pred["y"], pred["prob"]))
    row.update(state_recovery_metrics(pred.get("rep"), pred["s_true"]))
    if "expert_precision" in pred:
        for idx, value in enumerate(np.asarray(pred["expert_precision"]).reshape(-1)):
            row[f"poe_expert_precision_m{idx}"] = float(value)
    if diag is not None:
        row.update(diag)
    elif subset == "oracle":
        row.update(
            {
                "lambda_y": 1.0,
                "ambiguity_proxy": 0.0,
                "oracle_effective_rank": float("inf"),
                "oracle_min_eig_norm": float("inf"),
                "oracle_trace": float("inf"),
            }
        )
    return row


def run_iso_suite(
    train: ISOSplit,
    val: ISOSplit,
    id_test: Optional[ISOSplit],
    test: ISOSplit,
    params: ISOParams,
    scfg: ISODataConfig,
    tcfg: TrainConfig,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Train all first-round ISO baselines and return long-form metrics."""

    _set_seed(tcfg.seed)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    split_items: List[Tuple[str, ISOSplit]] = []
    if id_test is not None:
        split_items.append(("id_test", id_test))
    split_items.append(("test", test))

    rows: List[Dict] = []
    full_subset = tuple(range(scfg.n_modalities))
    diag_lookup = {subset_name(sub): diagnostics_for_subset(params, scfg, sub) for sub in modality_subsets(scfg.n_modalities)}

    for m in range(scfg.n_modalities):
        subset = (m,)
        model = UnimodalMLP(m, scfg.x_dim, hidden_dim=tcfg.hidden_dim)
        train_classifier(model, train, val, scfg, tcfg, subset)
        for split_name, split in split_items:
            pred = evaluate_classifier(model, split, scfg, tcfg, subset)
            rows.append(
                _row_from_eval(pred, "unimodal", split_name, scfg.scenario, scfg.seed, scfg.n_train, subset, diag_lookup[subset_name(subset)])
            )

    concat = ConcatMLP(scfg.n_modalities, scfg.x_dim, hidden_dim=tcfg.hidden_dim)
    train_classifier(concat, train, val, scfg, tcfg, full_subset)
    for split_name, split in split_items:
        pred = evaluate_classifier(concat, split, scfg, tcfg, full_subset)
        rows.append(_row_from_eval(pred, "concat", split_name, scfg.scenario, scfg.seed, scfg.n_train, full_subset, diag_lookup[subset_name(full_subset)]))

    late = LateFusionMLP(scfg.n_modalities, scfg.x_dim, hidden_dim=tcfg.hidden_dim)
    train_classifier(late, train, val, scfg, tcfg, full_subset)
    for split_name, split in split_items:
        pred = evaluate_classifier(late, split, scfg, tcfg, full_subset)
        rows.append(_row_from_eval(pred, "late_fusion", split_name, scfg.scenario, scfg.seed, scfg.n_train, full_subset, diag_lookup[subset_name(full_subset)]))

    poe_cfg = ModelConfig(
        n_modalities=scfg.n_modalities,
        x_dim=scfg.x_dim,
        s_dim=scfg.s_dim,
        hidden_dim=tcfg.hidden_dim,
    )
    poe = ISOPoE(poe_cfg)
    train_poe(poe, train, val, scfg, tcfg)
    for subset in modality_subsets(scfg.n_modalities):
        for split_name, split in split_items:
            pred = evaluate_poe(poe, split, scfg, tcfg, subset)
            rows.append(_row_from_eval(pred, "iso_poe", split_name, scfg.scenario, scfg.seed, scfg.n_train, subset, diag_lookup[subset_name(subset)]))

    oracle = OracleStateMLP(scfg.s_dim, hidden_dim=tcfg.hidden_dim)
    train_oracle(oracle, train, val, tcfg)
    for split_name, split in split_items:
        pred = evaluate_oracle(oracle, split, tcfg)
        rows.append(_row_from_eval(pred, "oracle_state", split_name, scfg.scenario, scfg.seed, scfg.n_train, "oracle", None))

    df = pd.DataFrame(rows)
    df["split_is_ood"] = df["split"].eq("test")
    df["ood_residual_shift"] = scfg.ood_residual_shift
    df["train_nuisance_corr"] = scfg.train_nuisance_corr
    df["test_nuisance_corr"] = scfg.test_nuisance_corr
    df["ood_noise_multiplier"] = scfg.ood_noise_multiplier
    if output_dir is not None:
        df.to_csv(os.path.join(output_dir, "results.csv"), index=False)
    return df
