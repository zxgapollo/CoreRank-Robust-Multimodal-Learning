from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import BCDataset, BCSplit, SCMConfig, SCMParams, SPLIT_NAMES, collate_bc_batch
from .metrics import binary_metrics, graph_recovery_metrics, mask_recovery_metrics, ridge_r2, shortcut_sensitivity, state_recovery_metrics
from .models import BCMCSGN, ConcatMLP, LateFusionMLP, ModelConfig, WarmupNet, diag_gaussian_kl, kl_standard_normal


@dataclass
class TrainConfig:
    warmup_epochs: int = 40
    correction_epochs: int = 80
    baseline_epochs: int = 40
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    layers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    modality_dropout: float = 0.20
    lambda_cm: float = 0.10
    warmup_proto_weight: float = 1.0
    label_weight: float = 1.0
    recon_weight: float = 0.50
    proto_weight: float = 1.0
    beta_z: float = 1e-3
    beta_u: float = 1e-3
    graph_l1_weight: float = 1e-3
    dag_weight: float = 1e-3
    mask_l1_weight: float = 1e-3
    state_anchor_weight: float = 0.0
    delta_anchor_weight: float = 0.0
    proto_source: str = "warmup"  # warmup | true_biased
    fixed_graph: bool = False
    fixed_masks: bool = False
    graph_threshold: float = 0.15
    seed: int = 0
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.proto_source not in {"warmup", "true_biased"}:
            raise ValueError("proto_source must be 'warmup' or 'true_biased'")

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


def _model_config(scfg: SCMConfig, tcfg: TrainConfig) -> ModelConfig:
    return ModelConfig(
        n_modalities=scfg.n_modalities,
        x_dim=scfg.x_dim,
        k=scfg.k,
        u_dim=scfg.u_dim,
        hidden_dim=tcfg.hidden_dim,
        layers=tcfg.layers,
        fixed_graph=tcfg.fixed_graph,
        fixed_masks=tcfg.fixed_masks,
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
            consistency = _sym_kl(out_a, out_b)
            proto_teacher = F.mse_loss(out_full["proto_mu"], batch["s_tilde"])
            loss = bce + tcfg.lambda_cm * consistency + tcfg.warmup_proto_weight * proto_teacher
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


def train_classifier(model: torch.nn.Module, train: BCSplit, val: BCSplit, tcfg: TrainConfig, name: str) -> torch.nn.Module:
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


@torch.no_grad()
def _proto_for_batch(warmup: WarmupNet, batch: Dict, source: str) -> torch.Tensor:
    if source == "true_biased":
        return batch["s_tilde"]
    warmup.eval()
    return warmup(batch["x"], batch["obs_mask"])["proto_mu"].detach()


def _bc_loss(model: BCMCSGN, warmup: WarmupNet, batch: Dict, tcfg: TrainConfig) -> torch.Tensor:
    proto = _proto_for_batch(warmup, batch, tcfg.proto_source)
    out = model(batch["x"], batch["obs_mask"], proto, sample=True)
    label = F.binary_cross_entropy_with_logits(out["logits"], batch["y"])  # type: ignore[arg-type]
    recon = torch.tensor(0.0, device=batch["y"].device)
    denom = batch["obs_mask"].sum().clamp_min(1.0)
    for m, rec in enumerate(out["recon"]):  # type: ignore[union-attr]
        mse = F.mse_loss(rec, batch["x"][m], reduction="none").mean(dim=-1)
        recon = recon + (mse * batch["obs_mask"][:, m]).sum() / denom
    proto_loss = F.mse_loss(out["proto_recon"], proto)  # type: ignore[arg-type]
    kl_z = kl_standard_normal(out["z_mu"], out["z_logvar"]).mean()  # type: ignore[arg-type]
    kl_u = torch.stack([kl_standard_normal(mu, lv).mean() for mu, lv in zip(out["u_mu"], out["u_logvar"])]).mean()  # type: ignore[arg-type]
    graph = model.graph_l1()
    dag = model.dag_penalty()
    mask = model.mask_l1()
    state_anchor = F.mse_loss(out["s"], batch["s"]) if tcfg.state_anchor_weight > 0 else torch.tensor(0.0, device=batch["y"].device)  # type: ignore[arg-type]
    delta_anchor = F.mse_loss(out["delta_hat"], batch["delta"]) if tcfg.delta_anchor_weight > 0 else torch.tensor(0.0, device=batch["y"].device)  # type: ignore[arg-type]
    return (
        tcfg.label_weight * label
        + tcfg.recon_weight * recon
        + tcfg.proto_weight * proto_loss
        + tcfg.beta_z * kl_z
        + tcfg.beta_u * kl_u
        + tcfg.graph_l1_weight * graph
        + tcfg.dag_weight * dag
        + tcfg.mask_l1_weight * mask
        + tcfg.state_anchor_weight * state_anchor
        + tcfg.delta_anchor_weight * delta_anchor
    )


@torch.no_grad()
def _bc_val_loss(model: BCMCSGN, warmup: WarmupNet, split: BCSplit, tcfg: TrainConfig) -> float:
    model.eval()
    losses: List[float] = []
    for batch0 in _loader(split, tcfg, shuffle=False):
        batch = _to_device(batch0, tcfg.device)
        proto = _proto_for_batch(warmup, batch, tcfg.proto_source)
        out = model(batch["x"], batch["obs_mask"], proto, sample=False)
        bce = F.binary_cross_entropy_with_logits(out["logits"], batch["y"])  # type: ignore[arg-type]
        proto_loss = F.mse_loss(out["proto_recon"], proto)  # type: ignore[arg-type]
        losses.append(float((bce + 0.25 * proto_loss).detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def train_bc_mcsgn(train: BCSplit, val: BCSplit, params: SCMParams, scfg: SCMConfig, tcfg: TrainConfig, warmup: WarmupNet) -> BCMCSGN:
    true_graph = torch.tensor(params.graph, device=tcfg.device)
    true_state_mask = torch.tensor(params.state_mask, device=tcfg.device)
    true_modality_mask = torch.tensor(params.modality_mask, device=tcfg.device)
    model = BCMCSGN(
        _model_config(scfg, tcfg),
        true_graph=true_graph,
        true_state_mask=true_state_mask,
        true_modality_mask=true_modality_mask,
    ).to(tcfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    best = None
    best_val = float("inf")
    warmup.eval()
    for _ in tqdm(range(tcfg.correction_epochs), desc="bc-mcsgn", leave=False, disable=not tcfg.verbose):
        model.train()
        for batch0 in _loader(train, tcfg, shuffle=True):
            batch = _to_device(batch0, tcfg.device)
            loss = _bc_loss(model, warmup, batch, tcfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val_loss = _bc_val_loss(model, warmup, val, tcfg)
        if val_loss < best_val:
            best_val = val_loss
            best = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    if best is not None:
        model.load_state_dict(best)
    return model


@torch.no_grad()
def evaluate_classifier(model: torch.nn.Module, split: BCSplit, tcfg: TrainConfig) -> Dict[str, np.ndarray]:
    model.eval()
    probs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    states: List[np.ndarray] = []
    for batch0 in _loader(split, tcfg, shuffle=False):
        batch = _to_device(batch0, tcfg.device)
        logits = model(batch["x"], batch["obs_mask"])
        probs.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(batch["y"].cpu().numpy())
        states.append(batch["s"].cpu().numpy())
    return {"prob": np.concatenate(probs), "y": np.concatenate(ys), "s_true": np.concatenate(states), "rep": None}


@torch.no_grad()
def evaluate_warmup(model: WarmupNet, split: BCSplit, tcfg: TrainConfig) -> Dict[str, np.ndarray]:
    model.eval()
    probs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    states: List[np.ndarray] = []
    reps: List[np.ndarray] = []
    deltas: List[np.ndarray] = []
    for batch0 in _loader(split, tcfg, shuffle=False):
        batch = _to_device(batch0, tcfg.device)
        out = model(batch["x"], batch["obs_mask"])
        probs.append(torch.sigmoid(out["logits"]).cpu().numpy())
        ys.append(batch["y"].cpu().numpy())
        states.append(batch["s"].cpu().numpy())
        reps.append(out["proto_mu"].cpu().numpy())
        deltas.append(batch["delta"].cpu().numpy())
    return {
        "prob": np.concatenate(probs),
        "y": np.concatenate(ys),
        "s_true": np.concatenate(states),
        "rep": np.concatenate(reps),
        "delta_true": np.concatenate(deltas),
    }


@torch.no_grad()
def evaluate_bc(model: BCMCSGN, warmup: WarmupNet, split: BCSplit, tcfg: TrainConfig) -> Dict[str, np.ndarray]:
    model.eval()
    probs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    states: List[np.ndarray] = []
    reps: List[np.ndarray] = []
    deltas: List[np.ndarray] = []
    delta_hats: List[np.ndarray] = []
    for batch0 in _loader(split, tcfg, shuffle=False):
        batch = _to_device(batch0, tcfg.device)
        proto = _proto_for_batch(warmup, batch, tcfg.proto_source)
        out = model(batch["x"], batch["obs_mask"], proto, sample=False)
        probs.append(torch.sigmoid(out["logits"]).cpu().numpy())  # type: ignore[arg-type]
        ys.append(batch["y"].cpu().numpy())
        states.append(batch["s"].cpu().numpy())
        reps.append(out["s"].cpu().numpy())  # type: ignore[union-attr]
        deltas.append(batch["delta"].cpu().numpy())
        delta_hats.append(out["delta_hat"].cpu().numpy())  # type: ignore[union-attr]
    return {
        "prob": np.concatenate(probs),
        "y": np.concatenate(ys),
        "s_true": np.concatenate(states),
        "rep": np.concatenate(reps),
        "delta_true": np.concatenate(deltas),
        "delta_hat": np.concatenate(delta_hats),
    }


def _direct_r2(pred: np.ndarray | None, true: np.ndarray | None) -> float:
    if pred is None or true is None:
        return float("nan")
    yhat = np.asarray(pred, dtype=np.float64)
    y = np.asarray(true, dtype=np.float64)
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean(axis=0, keepdims=True)) ** 2).sum()) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def metrics_from_eval(method: str, split_name: str, out: Dict[str, np.ndarray]) -> Dict[str, float | str]:
    row: Dict[str, float | str] = {"method": method, "split": split_name}
    row.update(binary_metrics(out["y"], out["prob"]))
    row.update(state_recovery_metrics(out.get("rep"), out["s_true"]))
    row["shortcut_sensitivity"] = shortcut_sensitivity(out["prob"], out["s_true"])
    if "delta_hat" in out:
        row["delta_direct_r2"] = _direct_r2(out.get("delta_hat"), out.get("delta_true"))
        row["delta_ridge_r2"] = ridge_r2(out.get("delta_hat"), out.get("delta_true"))
    else:
        row["delta_direct_r2"] = float("nan")
        row["delta_ridge_r2"] = float("nan")
    return row


def evaluate_all_splits(method: str, outputs: Mapping[str, Dict[str, np.ndarray]]) -> List[Dict[str, float | str]]:
    return [metrics_from_eval(method, split_name, outputs[split_name]) for split_name in SPLIT_NAMES]


def train_and_evaluate(
    splits: Mapping[str, BCSplit],
    params: SCMParams,
    scfg: SCMConfig,
    tcfg: TrainConfig,
    output_dir: str,
    methods: Sequence[str] = ("concat", "warmup", "bc_mcsgn"),
) -> List[Dict[str, float | str]]:
    _set_seed(tcfg.seed)
    os.makedirs(output_dir, exist_ok=True)
    rows: List[Dict[str, float | str]] = []
    mcfg = _model_config(scfg, tcfg)
    trained: Dict[str, torch.nn.Module] = {}

    warmup = train_warmup(splits["train"], splits["val"], scfg, tcfg)
    trained["warmup"] = warmup
    torch.save(warmup.state_dict(), os.path.join(output_dir, "warmup.pt"))

    if "concat" in methods:
        concat = train_classifier(ConcatMLP(mcfg), splits["train"], splits["val"], tcfg, name="concat")
        trained["concat"] = concat
        torch.save(concat.state_dict(), os.path.join(output_dir, "concat.pt"))
        evals = {name: evaluate_classifier(concat, split, tcfg) for name, split in splits.items()}
        rows.extend(evaluate_all_splits("concat", evals))

    if "late_fusion" in methods:
        late = train_classifier(LateFusionMLP(mcfg), splits["train"], splits["val"], tcfg, name="late_fusion")
        trained["late_fusion"] = late
        torch.save(late.state_dict(), os.path.join(output_dir, "late_fusion.pt"))
        evals = {name: evaluate_classifier(late, split, tcfg) for name, split in splits.items()}
        rows.extend(evaluate_all_splits("late_fusion", evals))

    if "warmup" in methods:
        evals = {name: evaluate_warmup(warmup, split, tcfg) for name, split in splits.items()}
        rows.extend(evaluate_all_splits("warmup", evals))

    if "bc_mcsgn" in methods:
        bc = train_bc_mcsgn(splits["train"], splits["val"], params, scfg, tcfg, warmup)
        trained["bc_mcsgn"] = bc
        torch.save(bc.state_dict(), os.path.join(output_dir, "bc_mcsgn.pt"))
        evals = {name: evaluate_bc(bc, warmup, split, tcfg) for name, split in splits.items()}
        bc_rows = evaluate_all_splits("bc_mcsgn", evals)
        graph_metrics = graph_recovery_metrics(bc.adjacency().detach().cpu().numpy(), params.graph, threshold=tcfg.graph_threshold)
        state_mask_metrics = mask_recovery_metrics(bc.state_mask().detach().cpu().numpy(), params.state_mask, prefix="state_mask")
        modality_mask_metrics = mask_recovery_metrics(bc.modality_mask().detach().cpu().numpy(), params.modality_mask, prefix="modality_mask")
        for row in bc_rows:
            row.update(graph_metrics)
            row.update(state_mask_metrics)
            row.update(modality_mask_metrics)
        rows.extend(bc_rows)

    return rows
