from __future__ import annotations

import itertools
import json
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import SyntheticConfig, SyntheticDataset, SyntheticParams, SyntheticSplit, collate_batch, true_fisher_for_split
from .fisher import batch_core_information, rank_score_from_K
from .metrics import binary_metrics, directed_graph_metrics, footprint_metrics, linear_probe_r2, mean_corrcoef_matching, ridge_r2
from .models import CoreRankVAE, EarlyFusionClassifier, ModelConfig, kl_sem_normal, kl_standard_normal


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    beta_z: float = 1e-3
    beta_u: float = 1e-3
    recon_weight: float = 1.0
    recon_reduction: str = "mean"
    label_weight: float = 1.0
    structural_weight: float = 0.0
    dag_weight: float = 0.1
    graph_l1_weight: float = 0.01
    structural_warmup_epochs: int = 0
    bias_invariance_weight: float = 0.0
    domain_invariance_weight: float = 0.0
    use_sem_prior: bool = True
    rank_on_innovation: bool = True
    select_best: bool = True
    best_id_tolerance: float = 0.02
    best_leakage_weight: float = 0.0
    rank_kappa: float = 0.5
    sparse_budget: float = 9.0
    rho_rank: float = 1.0
    rho_sparse: float = 0.1
    rank_warmup_epochs: int = 5
    sparse_warmup_epochs: int = 10
    modality_dropout: float = 0.15
    max_fisher_batch: int = 64
    fisher_damping: float = 1e-3
    rank_eps: float = 1e-3
    no_rank: bool = False
    no_sparse: bool = False
    seed: int = 0
    eval_fisher_batches: int = 4
    eval_true_fisher_samples: int = 256
    hidden_dim: int = 64
    encoder_layers: int = 2
    decoder_layers: int = 2
    gate_temperature: float = 0.67
    init_gate_logit: float = 0.0
    gate_temperature_min: float = 0.2
    gate_anneal_epochs: int = 0
    gate_l1_weight: float = 0.0
    gate_binary_weight: float = 0.0
    structural_classifier: bool = True


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_obs_mask(batch_size: int, n_modalities: int, device: str, drop_prob: float) -> torch.Tensor:
    if drop_prob <= 0:
        return torch.ones(batch_size, n_modalities, device=device)
    mask = (torch.rand(batch_size, n_modalities, device=device) > drop_prob).float()
    # Ensure at least one observed modality.
    empty = mask.sum(dim=1) == 0
    if empty.any():
        empty_rows = torch.where(empty)[0]
        idx = torch.randint(0, n_modalities, (empty_rows.numel(),), device=device)
        mask[empty_rows] = 0.0
        mask[empty_rows, idx] = 1.0
    return mask


def _to_device(batch: Dict, device: str) -> Dict:
    return {
        "x": [xm.to(device) for xm in batch["x"]],
        "y": batch["y"].to(device),
        "z": batch["z"].to(device),
        "u": [um.to(device) for um in batch["u"]],
        "bias": [bm.to(device) for bm in batch["bias"]],
        "domain": batch["domain"].to(device),
    }


def _make_context(batch: Dict, scfg: SyntheticConfig) -> torch.Tensor:
    return torch.cat([batch["bias"][scfg.biased_modality], batch["domain"]], dim=-1)


def _standardize_batch(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True).clamp_min(1e-5)


def _linear_invariance_penalty(z: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if z.shape[0] < 2:
        return torch.tensor(0.0, device=z.device)
    z0 = _standardize_batch(z)
    t0 = _standardize_batch(target)
    corr = z0.T @ t0 / max(1, z.shape[0] - 1)
    return corr.pow(2).mean()


def _vae_step_losses(
    model: CoreRankVAE,
    batch: Dict,
    obs_mask: torch.Tensor,
    context: torch.Tensor,
    tcfg: TrainConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict]:
    out = model(batch["x"], obs_mask, context=context, sample=True)
    sigma2 = model.cfg.decoder_noise_std ** 2
    recon_loss = torch.tensor(0.0, device=batch["y"].device)
    for m, rec in enumerate(out["recon"]):
        mse_raw = F.mse_loss(rec, batch["x"][m], reduction="none")
        if tcfg.recon_reduction == "sum":
            mse_m = mse_raw.sum(dim=-1)
        elif tcfg.recon_reduction == "mean":
            mse_m = mse_raw.mean(dim=-1)
        else:
            raise ValueError(f"Unknown recon_reduction={tcfg.recon_reduction!r}")
        recon_loss = recon_loss + (obs_mask[:, m] * 0.5 * mse_m / sigma2).mean()
    label_loss = F.binary_cross_entropy_with_logits(out["logits"], batch["y"], reduction="mean")
    if tcfg.use_sem_prior:
        kl_z = kl_sem_normal(out["z_mu"], out["z_logvar"], model.core_graph, context).mean()
    else:
        kl_z = kl_standard_normal(out["z_mu"], out["z_logvar"]).mean()
    kl_u = torch.tensor(0.0, device=batch["y"].device)
    for m in range(model.cfg.n_modalities):
        kl_u_m = kl_standard_normal(out["u_mu"][m], out["u_logvar"][m])
        kl_u = kl_u + (obs_mask[:, m] * kl_u_m).mean()
    nll = tcfg.recon_weight * recon_loss + tcfg.label_weight * label_loss + tcfg.beta_z * kl_z + tcfg.beta_u * kl_u
    logs = {
        "nll": nll.detach(),
        "recon_loss": recon_loss.detach(),
        "label_loss": label_loss.detach(),
        "kl_z": kl_z.detach(),
        "kl_u": kl_u.detach(),
    }
    return nll, logs, out


def train_corerank(
    train: SyntheticSplit,
    val: SyntheticSplit,
    test: SyntheticSplit,
    params: SyntheticParams,
    scfg: SyntheticConfig,
    tcfg: TrainConfig,
    output_dir: str,
    id_test: Optional[SyntheticSplit] = None,
) -> Tuple[CoreRankVAE, Dict, pd.DataFrame]:
    _set_seed(tcfg.seed)
    os.makedirs(output_dir, exist_ok=True)
    device = tcfg.device

    model_cfg = ModelConfig(
        n_modalities=scfg.n_modalities,
        x_dim=scfg.x_dim,
        z_dim=scfg.z_dim,
        u_dim=scfg.u_dim,
        hidden_dim=tcfg.hidden_dim,
        encoder_layers=tcfg.encoder_layers,
        decoder_layers=tcfg.decoder_layers,
        gate_temperature=tcfg.gate_temperature,
        init_gate_logit=tcfg.init_gate_logit,
        decoder_noise_std=scfg.noise_std,
        context_dim=2,
        structural_classifier=tcfg.structural_classifier,
    )
    model = CoreRankVAE(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)

    loader = DataLoader(SyntheticDataset(train), batch_size=tcfg.batch_size, shuffle=True, collate_fn=collate_batch, drop_last=False)

    lambda_rank = torch.tensor(0.0, device=device)
    lambda_sparse = torch.tensor(0.0, device=device)
    history = []
    checkpoint_candidates = []

    for epoch in range(tcfg.epochs):
        model.train()
        if tcfg.gate_anneal_epochs > 0:
            frac = min(1.0, epoch / max(1, tcfg.gate_anneal_epochs - 1))
            start = max(tcfg.gate_temperature, 1e-6)
            end = max(tcfg.gate_temperature_min, 1e-6)
            model.gates.temperature = float(start * ((end / start) ** frac))
        pbar = tqdm(loader, desc=f"CoreRank epoch {epoch+1}/{tcfg.epochs}", leave=False)
        epoch_logs: Dict[str, List[float]] = {}
        rank_active = (epoch + 1) > tcfg.rank_warmup_epochs and not tcfg.no_rank
        sparse_active = (epoch + 1) > tcfg.sparse_warmup_epochs and not tcfg.no_sparse
        structural_active = (epoch + 1) > tcfg.structural_warmup_epochs

        for batch0 in pbar:
            batch = _to_device(batch0, device)
            context = _make_context(batch, scfg)
            obs_mask = _make_obs_mask(batch["y"].shape[0], scfg.n_modalities, device, tcfg.modality_dropout)
            opt.zero_grad(set_to_none=True)
            nll, logs, out = _vae_step_losses(model, batch, obs_mask, context, tcfg)
            loss = nll

            structural_z = _standardize_batch(out["z_mu"])
            structural_context = _standardize_batch(context)
            structural_loss = 0.5 * model.core_graph.innovation(structural_z, structural_context).pow(2).mean()
            dag_penalty = model.core_graph.acyclicity()
            graph_l1 = model.core_graph.l1()
            bias_invariance = torch.tensor(0.0, device=device)
            if scfg.bias_strength > 0 and tcfg.bias_invariance_weight > 0:
                bias_invariance = _linear_invariance_penalty(out["innovation_mu"], batch["bias"][scfg.biased_modality])
            domain_invariance = torch.tensor(0.0, device=device)
            if scfg.domain_shift_strength > 0 and tcfg.domain_invariance_weight > 0:
                domain_invariance = _linear_invariance_penalty(out["innovation_mu"], batch["domain"])
            if structural_active:
                if tcfg.structural_weight > 0:
                    loss = loss + tcfg.structural_weight * structural_loss
                if tcfg.dag_weight > 0:
                    loss = loss + tcfg.dag_weight * dag_penalty.pow(2)
                if tcfg.graph_l1_weight > 0:
                    loss = loss + tcfg.graph_l1_weight * graph_l1
            if tcfg.bias_invariance_weight > 0:
                loss = loss + tcfg.bias_invariance_weight * bias_invariance
            if tcfg.domain_invariance_weight > 0:
                loss = loss + tcfg.domain_invariance_weight * domain_invariance

            rank_score = torch.tensor(0.0, device=device)
            rank_violation = torch.tensor(0.0, device=device)
            if rank_active:
                K = batch_core_information(
                    model, out["z"], out["u"], obs_mask,
                    noise_std=scfg.noise_std,
                    damping=tcfg.fisher_damping,
                    max_fisher_batch=tcfg.max_fisher_batch,
                )
                if tcfg.rank_on_innovation:
                    K = model.core_graph.transform_information_to_innovation(K)
                rank_score, _ = rank_score_from_K(K, eps=tcfg.rank_eps)
                rank_violation = F.relu(torch.tensor(tcfg.rank_kappa, device=device) - rank_score)
                loss = loss + lambda_rank.detach() * rank_violation + 0.5 * tcfg.rho_rank * rank_violation.pow(2)

            omega = model.gates.expected_l0()
            gate = model.gates()
            gate_l1 = omega / gate.numel()
            gate_binary = (gate * (1.0 - gate)).mean()
            sparse_violation = torch.tensor(0.0, device=device)
            if sparse_active:
                sparse_violation = F.relu(omega - torch.tensor(tcfg.sparse_budget, device=device))
                loss = loss + lambda_sparse.detach() * sparse_violation + 0.5 * tcfg.rho_sparse * sparse_violation.pow(2)
                if tcfg.gate_l1_weight > 0:
                    loss = loss + tcfg.gate_l1_weight * gate_l1
                if tcfg.gate_binary_weight > 0:
                    loss = loss + tcfg.gate_binary_weight * gate_binary

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            with torch.no_grad():
                if rank_active:
                    lambda_rank = torch.clamp(lambda_rank + tcfg.rho_rank * rank_violation.detach(), min=0.0)
                if sparse_active:
                    lambda_sparse = torch.clamp(lambda_sparse + tcfg.rho_sparse * sparse_violation.detach(), min=0.0)

            step_logs = {
                **{k: float(v.detach().cpu()) for k, v in logs.items()},
                "loss": float(loss.detach().cpu()),
                "rank_score": float(rank_score.detach().cpu()),
                "rank_violation": float(rank_violation.detach().cpu()),
                "structural_loss": float(structural_loss.detach().cpu()),
                "dag_penalty": float(dag_penalty.detach().cpu()),
                "graph_l1": float(graph_l1.detach().cpu()),
                "bias_invariance": float(bias_invariance.detach().cpu()),
                "domain_invariance": float(domain_invariance.detach().cpu()),
                "omega_g": float(omega.detach().cpu()),
                "gate_l1": float(gate_l1.detach().cpu()),
                "gate_binary": float(gate_binary.detach().cpu()),
                "gate_temperature": float(model.gates.temperature),
                "sparse_violation": float(sparse_violation.detach().cpu()),
                "lambda_rank": float(lambda_rank.detach().cpu()),
                "lambda_sparse": float(lambda_sparse.detach().cpu()),
            }
            for k, v in step_logs.items():
                epoch_logs.setdefault(k, []).append(v)
            pbar.set_postfix({
                "loss": step_logs["loss"],
                "rank": step_logs["rank_score"],
                "G": step_logs["omega_g"],
                "dag": step_logs["dag_penalty"],
            })

        summary = {k: float(np.mean(v)) for k, v in epoch_logs.items()}
        summary["epoch"] = epoch + 1
        if tcfg.select_best:
            val_metrics = _quick_prediction_metrics(model, val, scfg, tcfg, tuple(range(scfg.n_modalities)))
            summary["val_full_auroc"] = val_metrics["auroc"]
            summary["val_full_accuracy"] = val_metrics["accuracy"]
            summary["val_context_leakage_r2"] = val_metrics["context_leakage_r2"]
            summary["val_robust_score"] = val_metrics["robust_score"]
            if np.isfinite(val_metrics["auroc"]):
                checkpoint_candidates.append({
                    "epoch": epoch + 1,
                    "state": deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
                    "val_auroc": float(val_metrics["auroc"]),
                    "val_accuracy": float(val_metrics["accuracy"]),
                    "val_context_leakage_r2": float(val_metrics["context_leakage_r2"]),
                    "val_robust_score": float(val_metrics["robust_score"]),
                })
        history.append(summary)

    # Evaluation.
    best_summary = None
    best_val_auc = float("nan")
    best_val_auc_floor = float("nan")
    if tcfg.select_best and checkpoint_candidates:
        best_val_auc = max(c["val_auroc"] for c in checkpoint_candidates)
        best_val_auc_floor = best_val_auc - max(0.0, tcfg.best_id_tolerance)
        feasible = [c for c in checkpoint_candidates if c["val_auroc"] >= best_val_auc_floor]
        if not feasible:
            feasible = checkpoint_candidates
        if tcfg.best_leakage_weight > 0:
            best_summary = max(
                feasible,
                key=lambda c: (c["val_robust_score"], c["val_auroc"], -c["val_context_leakage_r2"]),
            )
        else:
            best_summary = max(
                feasible,
                key=lambda c: (c["val_auroc"], -c["val_context_leakage_r2"]),
            )
        model.load_state_dict(best_summary["state"])
    metrics, subset_df = evaluate_model(model, train, val, test, params, scfg, tcfg, id_test=id_test)
    metrics["best_epoch"] = int(best_summary["epoch"]) if best_summary is not None else 0
    metrics["best_val_auroc"] = float(best_summary["val_auroc"]) if best_summary is not None else float("nan")
    metrics["best_val_accuracy"] = float(best_summary["val_accuracy"]) if best_summary is not None else float("nan")
    metrics["best_val_context_leakage_r2"] = float(best_summary["val_context_leakage_r2"]) if best_summary is not None else float("nan")
    metrics["best_val_robust_score"] = float(best_summary["val_robust_score"]) if best_summary is not None else float("nan")
    metrics["best_val_auc_target"] = float(best_val_auc)
    metrics["best_val_auc_floor"] = float(best_val_auc_floor)
    metrics["best_id_tolerance"] = float(tcfg.best_id_tolerance)
    metrics["train_history"] = history
    metrics["gates"] = model.gates().detach().cpu().numpy().tolist()
    metrics["true_footprint"] = params.footprint.tolist()
    learned_core_graph = model.core_graph.adjacency().detach().cpu().numpy()
    metrics["learned_core_graph"] = learned_core_graph.tolist()
    metrics["true_core_graph"] = params.core_graph.tolist()

    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    np.save(os.path.join(output_dir, "gates.npy"), model.gates().detach().cpu().numpy())
    np.save(os.path.join(output_dir, "true_footprint.npy"), params.footprint)
    np.save(os.path.join(output_dir, "learned_core_graph.npy"), learned_core_graph)
    np.save(os.path.join(output_dir, "true_core_graph.npy"), params.core_graph)
    pd.DataFrame(history).to_csv(os.path.join(output_dir, "train_history.csv"), index=False)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    subset_df.to_csv(os.path.join(output_dir, "subset_metrics.csv"), index=False)
    return model, metrics, subset_df


@torch.no_grad()
def _collect_predictions(model: CoreRankVAE, split: SyntheticSplit, scfg: SyntheticConfig, tcfg: TrainConfig, subset: Tuple[int, ...]) -> Dict[str, np.ndarray]:
    device = tcfg.device
    loader = DataLoader(SyntheticDataset(split), batch_size=tcfg.batch_size, shuffle=False, collate_fn=collate_batch)
    zs, e_hats, ztrue, probs, ys, biases, domains = [], [], [], [], [], [], []
    obs_template = torch.zeros(scfg.n_modalities, device=device)
    obs_template[list(subset)] = 1.0
    model.eval()
    for batch0 in loader:
        batch = _to_device(batch0, device)
        context = _make_context(batch, scfg)
        obs_mask = obs_template.unsqueeze(0).expand(batch["y"].shape[0], -1)
        out = model(batch["x"], obs_mask, context=context, sample=False)
        prob = torch.sigmoid(out["logits"])
        zs.append(out["z_mu"].cpu().numpy())
        e_hats.append(out["innovation_mu"].cpu().numpy())
        ztrue.append(batch["z"].cpu().numpy())
        probs.append(prob.cpu().numpy())
        ys.append(batch["y"].cpu().numpy())
        biases.append(np.concatenate([b.cpu().numpy() for b in batch["bias"]], axis=1))
        domains.append(batch["domain"].cpu().numpy())
    return {
        "z_hat": np.concatenate(zs, axis=0),
        "e_hat": np.concatenate(e_hats, axis=0),
        "z_true": np.concatenate(ztrue, axis=0),
        "prob": np.concatenate(probs, axis=0),
        "y": np.concatenate(ys, axis=0),
        "bias": np.concatenate(biases, axis=0),
        "domain": np.concatenate(domains, axis=0),
    }


def _true_innovation(split: SyntheticSplit, params: SyntheticParams) -> np.ndarray:
    z = split.z.numpy()
    transform = np.eye(params.core_graph.shape[0], dtype=np.float32) - params.core_graph
    return (z @ transform.T).astype(np.float32)


@torch.no_grad()
def _quick_prediction_metrics(
    model: CoreRankVAE,
    split: SyntheticSplit,
    scfg: SyntheticConfig,
    tcfg: TrainConfig,
    subset: Tuple[int, ...],
) -> Dict[str, float]:
    pred = _collect_predictions(model, split, scfg, tcfg, subset)
    metrics = binary_metrics(pred["y"], pred["prob"])
    leakage = 0.0
    if scfg.bias_strength > 0:
        leakage += linear_probe_r2(pred["e_hat"], pred["bias"][:, [scfg.biased_modality]])
    if scfg.domain_shift_strength > 0:
        leakage += linear_probe_r2(pred["e_hat"], pred["domain"])
    metrics["context_leakage_r2"] = float(leakage)
    metrics["robust_score"] = float(metrics["auroc"] - tcfg.best_leakage_weight * leakage)
    return metrics


def _estimate_rank_diagnostics(model: CoreRankVAE, split: SyntheticSplit, scfg: SyntheticConfig, tcfg: TrainConfig, subset: Tuple[int, ...]) -> Dict[str, float]:
    device = tcfg.device
    loader = DataLoader(SyntheticDataset(split), batch_size=tcfg.batch_size, shuffle=True, collate_fn=collate_batch)
    obs_template = torch.zeros(scfg.n_modalities, device=device)
    obs_template[list(subset)] = 1.0
    vals = []
    model.eval()
    count = 0
    for batch0 in loader:
        if count >= tcfg.eval_fisher_batches:
            break
        batch = _to_device(batch0, device)
        obs_mask = obs_template.unsqueeze(0).expand(batch["y"].shape[0], -1)
        context = _make_context(batch, scfg)
        out = model(batch["x"], obs_mask, context=context, sample=False)
        K = batch_core_information(
            model, out["z_mu"], out["u_mu"], obs_mask,
            noise_std=scfg.noise_std,
            damping=tcfg.fisher_damping,
            max_fisher_batch=tcfg.max_fisher_batch,
        )
        if tcfg.rank_on_innovation:
            K = model.core_graph.transform_information_to_innovation(K)
        score, diag = rank_score_from_K(K, eps=tcfg.rank_eps)
        vals.append({
            "rank_logdet": float(score.detach().cpu()),
            "effective_rank": float(diag["effective_rank"].mean().cpu()),
            "min_eig": float(diag["min_eig"].mean().cpu()),
            "trace_K": float(diag["trace"].mean().cpu()),
        })
        count += 1
    if not vals:
        return {"rank_logdet": float("nan"), "effective_rank": float("nan"), "min_eig": float("nan"), "trace_K": float("nan")}
    return {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}


def _slice_split(split: SyntheticSplit, n: int) -> SyntheticSplit:
    return SyntheticSplit(
        x=[xm[:n] for xm in split.x],
        y=split.y[:n],
        z=split.z[:n],
        u=[um[:n] for um in split.u],
        bias=[bm[:n] for bm in split.bias],
        domain=split.domain[:n],
    )


def _estimate_true_rank_diagnostics(
    split: SyntheticSplit,
    params: SyntheticParams,
    scfg: SyntheticConfig,
    tcfg: TrainConfig,
    subset: Tuple[int, ...],
) -> Dict[str, float]:
    n = min(tcfg.eval_true_fisher_samples, split.y.shape[0])
    K = true_fisher_for_split(_slice_split(split, n), params, scfg, list(subset))
    if tcfg.rank_on_innovation:
        graph = torch.tensor(params.core_graph, dtype=K.dtype)
        jac = torch.linalg.inv(torch.eye(graph.shape[0], dtype=K.dtype) - graph)
        K = torch.einsum("ij,bjk,kl->bil", jac.T, K, jac)
    score, diag = rank_score_from_K(K, eps=tcfg.rank_eps)
    return {
        "true_rank_logdet": float(score.detach().cpu()),
        "true_effective_rank": float(diag["effective_rank"].mean().cpu()),
        "true_min_eig": float(diag["min_eig"].mean().cpu()),
        "true_trace_K": float(diag["trace"].mean().cpu()),
    }


def evaluate_model(
    model: CoreRankVAE,
    train: SyntheticSplit,
    val: SyntheticSplit,
    test: SyntheticSplit,
    params: SyntheticParams,
    scfg: SyntheticConfig,
    tcfg: TrainConfig,
    id_test: Optional[SyntheticSplit] = None,
) -> Tuple[Dict, pd.DataFrame]:
    all_rows = []
    split_items = [("train", train), ("val", val)]
    if id_test is not None:
        split_items.append(("id_test", id_test))
    split_items.append(("test", test))
    for split_name, split in split_items:
        for k in range(1, scfg.n_modalities + 1):
            for subset in itertools.combinations(range(scfg.n_modalities), k):
                pred = _collect_predictions(model, split, scfg, tcfg, subset)
                row: Dict[str, float | str] = {
                    "split": split_name,
                    "subset": "".join(str(i) for i in subset),
                    "subset_size": len(subset),
                }
                row.update(binary_metrics(pred["y"], pred["prob"]))
                row["latent_r2"] = ridge_r2(pred["z_hat"], pred["z_true"])
                row["latent_mcc"] = mean_corrcoef_matching(pred["z_hat"], pred["z_true"])
                true_e = _true_innovation(split, params)
                row["innovation_r2"] = ridge_r2(pred["e_hat"], true_e)
                row["innovation_mcc"] = mean_corrcoef_matching(pred["e_hat"], true_e)
                if scfg.bias_strength > 0:
                    bias_target = pred["bias"][:, [scfg.biased_modality]]
                    row["bias_leakage_r2"] = linear_probe_r2(pred["e_hat"], bias_target)
                    row["z_bias_leakage_r2"] = linear_probe_r2(pred["z_hat"], bias_target)
                else:
                    row["bias_leakage_r2"] = float("nan")
                    row["z_bias_leakage_r2"] = float("nan")
                if scfg.domain_shift_strength > 0:
                    row["domain_leakage_r2"] = linear_probe_r2(pred["e_hat"], pred["domain"])
                    row["z_domain_leakage_r2"] = linear_probe_r2(pred["z_hat"], pred["domain"])
                else:
                    row["domain_leakage_r2"] = float("nan")
                    row["z_domain_leakage_r2"] = float("nan")
                if split_name in {"val", "test"}:
                    row.update(_estimate_rank_diagnostics(model, split, scfg, tcfg, subset))
                    row.update(_estimate_true_rank_diagnostics(split, params, scfg, tcfg, subset))
                all_rows.append(row)
    subset_df = pd.DataFrame(all_rows)

    gates = model.gates().detach().cpu().numpy()
    learned_core_graph = model.core_graph.adjacency().detach().cpu().numpy()
    gate_metrics = footprint_metrics(gates, params.footprint)
    graph_metrics = directed_graph_metrics(learned_core_graph, params.core_graph)
    main = {
        "gate_metrics": gate_metrics,
        "graph_metrics": graph_metrics,
        "final_gates": gates.tolist(),
        "learned_core_graph": learned_core_graph.tolist(),
        "true_core_graph": params.core_graph.tolist(),
        "full_test": subset_df[(subset_df.split == "test") & (subset_df.subset_size == scfg.n_modalities)].iloc[0].to_dict(),
    }
    if id_test is not None:
        main["id_full_test"] = subset_df[(subset_df.split == "id_test") & (subset_df.subset_size == scfg.n_modalities)].iloc[0].to_dict()
    return main, subset_df


def train_erm_baseline(
    train: SyntheticSplit,
    val: SyntheticSplit,
    test: SyntheticSplit,
    scfg: SyntheticConfig,
    tcfg: TrainConfig,
    output_dir: str,
    epochs: Optional[int] = None,
    id_test: Optional[SyntheticSplit] = None,
) -> Dict:
    _set_seed(tcfg.seed + 991)
    device = tcfg.device
    epochs = epochs or tcfg.epochs
    model = EarlyFusionClassifier(scfg.n_modalities, scfg.x_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    loader = DataLoader(SyntheticDataset(train), batch_size=tcfg.batch_size, shuffle=True, collate_fn=collate_batch)
    for _ in tqdm(range(epochs), desc="ERM baseline", leave=False):
        model.train()
        for batch0 in loader:
            batch = _to_device(batch0, device)
            obs_mask = torch.ones(batch["y"].shape[0], scfg.n_modalities, device=device)
            opt.zero_grad(set_to_none=True)
            logits = model(batch["x"], obs_mask)
            loss = F.binary_cross_entropy_with_logits(logits, batch["y"])
            loss.backward()
            opt.step()

    # Evaluate full modality and all subsets.
    rows = []
    model.eval()
    split_items = [("train", train), ("val", val)]
    if id_test is not None:
        split_items.append(("id_test", id_test))
    split_items.append(("test", test))
    for split_name, split in split_items:
        loader_eval = DataLoader(SyntheticDataset(split), batch_size=tcfg.batch_size, shuffle=False, collate_fn=collate_batch)
        for k in range(1, scfg.n_modalities + 1):
            for subset in itertools.combinations(range(scfg.n_modalities), k):
                probs, ys = [], []
                obs_template = torch.zeros(scfg.n_modalities, device=device)
                obs_template[list(subset)] = 1.0
                with torch.no_grad():
                    for batch0 in loader_eval:
                        batch = _to_device(batch0, device)
                        obs_mask = obs_template.unsqueeze(0).expand(batch["y"].shape[0], -1)
                        logits = model(batch["x"], obs_mask)
                        probs.append(torch.sigmoid(logits).cpu().numpy())
                        ys.append(batch["y"].cpu().numpy())
                row = {"split": split_name, "subset": "".join(str(i) for i in subset), "subset_size": k}
                row.update(binary_metrics(np.concatenate(ys, axis=0), np.concatenate(probs, axis=0)))
                rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "erm_subset_metrics.csv"), index=False)
    torch.save(model.state_dict(), os.path.join(output_dir, "erm_model.pt"))
    out = {"erm_full_test": df[(df.split == "test") & (df.subset_size == scfg.n_modalities)].iloc[0].to_dict()}
    if id_test is not None:
        out["erm_id_full_test"] = df[(df.split == "id_test") & (df.subset_size == scfg.n_modalities)].iloc[0].to_dict()
    return out
