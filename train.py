from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from bc_mcsgn.metrics import graph_recovery_metrics, mask_recovery_metrics, state_recovery_metrics

from .data import ADDataset, ADParams, ADSCMConfig, ADSplit, collate_ad_batch


@dataclass
class TrainConfig:
    epochs: int = 40
    correction_epochs: int = 40
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    layers: int = 3
    u_dim: int = 2
    modality_dropout: float = 0.10
    warmup_proto_weight: float = 0.0
    label_weight: float = 1.0
    recon_weight: float = 0.5
    proto_weight: float = 0.5
    beta_z: float = 1e-3
    beta_u: float = 1e-3
    graph_l1_weight: float = 1e-3
    dag_weight: float = 0.0
    mask_l1_weight: float = 1e-3
    edge_entropy_weight: float = 0.0
    mask_entropy_weight: float = 0.0
    graph_separation_weight: float = 0.0
    graph_label_mix: float = 0.0
    state_anchor_weight: float = 0.0
    graph_threshold: float = 0.15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    verbose: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


def make_mlp(in_dim: int, out_dim: int, hidden_dim: int, layers: int) -> nn.Sequential:
    mods: List[nn.Module] = []
    last = in_dim
    for _ in range(max(0, layers - 1)):
        mods.append(nn.Linear(last, hidden_dim))
        mods.append(nn.SiLU())
        last = hidden_dim
    mods.append(nn.Linear(last, out_dim))
    return nn.Sequential(*mods)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample: bool) -> torch.Tensor:
    if not sample:
        return mu
    return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)


def kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.sum(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar, dim=-1)


def bernoulli_entropy(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


class ConcatClassifier(nn.Module):
    def __init__(self, x_dim: int, modalities: Sequence[int], hidden_dim: int, layers: int, n_classes: int = 3):
        super().__init__()
        self.modalities = tuple(modalities)
        self.x_dim = x_dim
        self.net = make_mlp(len(self.modalities) * x_dim, n_classes, hidden_dim, layers)

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        return self.net(torch.cat([xs[m] for m in self.modalities], dim=-1))


class LateFusionClassifier(nn.Module):
    def __init__(self, x_dim: int, n_modalities: int, hidden_dim: int, layers: int, n_classes: int = 3):
        super().__init__()
        self.n_modalities = n_modalities
        self.encoders = nn.ModuleList([make_mlp(x_dim, hidden_dim, hidden_dim, 2) for _ in range(n_modalities)])
        self.head = make_mlp(hidden_dim, n_classes, hidden_dim, layers)

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        hs = torch.stack([enc(x) for enc, x in zip(self.encoders, xs)], dim=1)
        return self.head(hs.mean(dim=1))


class WarmupStateNet(nn.Module):
    """Multimodal encoder q_psi(S_tilde | X_1:m) with a 3-class head."""

    def __init__(self, x_dim: int, n_modalities: int, k: int, hidden_dim: int, layers: int, n_classes: int = 3):
        super().__init__()
        self.k = k
        self.encoders = nn.ModuleList([make_mlp(x_dim, hidden_dim, hidden_dim, layers) for _ in range(n_modalities)])
        self.proto_head = make_mlp(hidden_dim, 2 * k, hidden_dim, layers)
        self.classifier = make_mlp(k, n_classes, hidden_dim, layers)

    def fuse(self, xs: List[torch.Tensor]) -> torch.Tensor:
        hs = torch.stack([enc(x) for enc, x in zip(self.encoders, xs)], dim=1)
        return hs.mean(dim=1)

    def forward(self, xs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = self.fuse(xs)
        mu, logvar = torch.split(self.proto_head(h), self.k, dim=-1)
        logvar = logvar.clamp(-6.0, 3.0)
        return {"logits": self.classifier(mu), "proto_mu": mu, "proto_logvar": logvar}


class ADCausalCorrectionNet(nn.Module):
    """AD adaptation of CSGN/BC-MCSGN with learned latent graph and masks.

    The S-conditioned variant follows the paper's idea that causal weights
    w_ij can depend on a latent label/setting S. In this simulation S is
    represented by a 3-class setting posterior from the warmup encoder.
    """

    def __init__(
        self,
        x_dim: int,
        n_modalities: int,
        k: int,
        u_dim: int,
        hidden_dim: int,
        layers: int,
        n_classes: int = 3,
        conditioned_graph: bool = False,
        decoder_uses_state: bool = True,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.n_modalities = n_modalities
        self.k = k
        self.u_dim = u_dim
        self.n_classes = n_classes
        self.conditioned_graph = conditioned_graph
        self.decoder_uses_state = decoder_uses_state
        self.modality_encoders = nn.ModuleList([make_mlp(x_dim, hidden_dim, hidden_dim, layers) for _ in range(n_modalities)])
        self.z_encoder = make_mlp(hidden_dim + k, 2 * k, hidden_dim, layers)
        self.u_encoders = nn.ModuleList([make_mlp(x_dim + k, 2 * u_dim, hidden_dim, layers) for _ in range(n_modalities)])
        decoder_in_dim = k + u_dim + (k if decoder_uses_state else 0)
        self.decoders = nn.ModuleList([make_mlp(decoder_in_dim, x_dim, hidden_dim, layers) for _ in range(n_modalities)])
        self.bias_net = make_mlp(n_modalities * u_dim, k, hidden_dim, layers)
        self.classifier = make_mlp(k, n_classes, hidden_dim, layers)
        lower = torch.tril(torch.ones(k, k), diagonal=-1)
        self.register_buffer("lower_mask", lower)
        if conditioned_graph:
            self.graph_logits = nn.Parameter(-0.20 + 0.03 * torch.randn(n_classes, k, k))
            self.graph_weight = nn.Parameter(0.10 * torch.randn(n_classes, k, k))
        else:
            self.graph_raw = nn.Parameter(0.03 * torch.randn(k, k))
        self.state_mask_logits = nn.Parameter(torch.zeros(k))
        self.modality_mask_logits = nn.Parameter(torch.zeros(n_modalities, k))

    def adjacency_by_class(self) -> torch.Tensor:
        if self.conditioned_graph:
            edge_prob = torch.sigmoid(self.graph_logits) * self.lower_mask[None, :, :]
            return torch.tanh(self.graph_weight) * edge_prob
        return (self.graph_raw * self.lower_mask)[None, :, :]

    def adjacency(self, setting_probs: torch.Tensor | None = None) -> torch.Tensor:
        a_by_class = self.adjacency_by_class()
        if not self.conditioned_graph:
            return a_by_class[0]
        if setting_probs is None:
            return a_by_class.mean(dim=0)
        return torch.einsum("bc,cij->bij", setting_probs, a_by_class)

    def edge_prob_by_class(self) -> torch.Tensor:
        if self.conditioned_graph:
            return torch.sigmoid(self.graph_logits) * self.lower_mask[None, :, :]
        return torch.sigmoid(self.graph_raw.abs())[None, :, :] * self.lower_mask[None, :, :]

    def state_mask(self) -> torch.Tensor:
        return torch.sigmoid(self.state_mask_logits)

    def modality_mask(self) -> torch.Tensor:
        return torch.sigmoid(self.modality_mask_logits)

    def fuse_modalities(self, xs: List[torch.Tensor]) -> torch.Tensor:
        hs = torch.stack([enc(x) for enc, x in zip(self.modality_encoders, xs)], dim=1)
        return hs.mean(dim=1)

    def encode(self, xs: List[torch.Tensor], proto: torch.Tensor) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        fused = self.fuse_modalities(xs)
        z_mu, z_logvar = torch.split(self.z_encoder(torch.cat([fused, proto], dim=-1)), self.k, dim=-1)
        z_logvar = z_logvar.clamp(-6.0, 3.0)
        u_mu: List[torch.Tensor] = []
        u_logvar: List[torch.Tensor] = []
        for enc, x in zip(self.u_encoders, xs):
            mu, logvar = torch.split(enc(torch.cat([x, proto], dim=-1)), self.u_dim, dim=-1)
            u_mu.append(mu)
            u_logvar.append(logvar.clamp(-6.0, 3.0))
        return {"z_mu": z_mu, "z_logvar": z_logvar, "u_mu": u_mu, "u_logvar": u_logvar}

    def state_from_z(self, z: torch.Tensor, setting_probs: torch.Tensor | None = None) -> torch.Tensor:
        a = self.adjacency(setting_probs)
        if a.dim() == 2:
            graph_features = z + z @ a.T
        else:
            graph_features = z + torch.bmm(a, z.unsqueeze(-1)).squeeze(-1)
        return graph_features * self.state_mask()[None, :]

    def forward(
        self,
        xs: List[torch.Tensor],
        proto: torch.Tensor,
        setting_probs: torch.Tensor | None = None,
        sample: bool = True,
    ) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        enc = self.encode(xs, proto)
        z_mu = enc["z_mu"]  # type: ignore[assignment]
        z_logvar = enc["z_logvar"]  # type: ignore[assignment]
        u_mu = enc["u_mu"]  # type: ignore[assignment]
        u_logvar = enc["u_logvar"]  # type: ignore[assignment]
        z = reparameterize(z_mu, z_logvar, sample=sample)
        us = [reparameterize(mu, logvar, sample=sample) for mu, logvar in zip(u_mu, u_logvar)]
        s = self.state_from_z(z, setting_probs)
        delta_hat = self.bias_net(torch.cat(us, dim=-1))
        proto_recon = s + delta_hat
        masks = self.modality_mask()
        recons = []
        for m, dec in enumerate(self.decoders):
            dec_inputs = [z * masks[m][None, :]]
            if self.decoder_uses_state:
                dec_inputs.append(s)
            dec_inputs.append(us[m])
            recons.append(dec(torch.cat(dec_inputs, dim=-1)))
        return {
            "logits": self.classifier(s),
            "z": z,
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "u": us,
            "u_mu": u_mu,
            "u_logvar": u_logvar,
            "s": s,
            "delta_hat": delta_hat,
            "proto_recon": proto_recon,
            "recon": recons,
        }

    def graph_l1(self) -> torch.Tensor:
        return self.adjacency_by_class().abs().mean()

    def mask_l1(self) -> torch.Tensor:
        return self.state_mask().mean() + self.modality_mask().mean()

    def graph_entropy(self) -> torch.Tensor:
        edge_prob = self.edge_prob_by_class()
        denom = self.lower_mask.sum().clamp_min(1.0) * edge_prob.shape[0]
        return (bernoulli_entropy(edge_prob) * self.lower_mask[None, :, :]).sum() / denom

    def mask_entropy(self) -> torch.Tensor:
        return bernoulli_entropy(self.state_mask()).mean() + bernoulli_entropy(self.modality_mask()).mean()

    def graph_separation(self) -> torch.Tensor:
        if not self.conditioned_graph or self.n_classes < 2:
            return torch.tensor(0.0, device=self.lower_mask.device)
        a = self.adjacency_by_class().reshape(self.n_classes, -1)
        vals = []
        for i in range(self.n_classes):
            for j in range(i + 1, self.n_classes):
                vals.append((a[i] - a[j]).pow(2).mean())
        return torch.stack(vals).mean() if vals else torch.tensor(0.0, device=a.device)

    def dag_penalty(self) -> torch.Tensor:
        vals = []
        for a in self.adjacency_by_class():
            vals.append(torch.trace(torch.matrix_exp(a * a)) - a.shape[0])
        return torch.stack(vals).mean()


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(split: ADSplit, cfg: TrainConfig, shuffle: bool) -> DataLoader:
    return DataLoader(ADDataset(split), batch_size=cfg.batch_size, shuffle=shuffle, collate_fn=collate_ad_batch)


def _to_device(batch: Mapping, device: str) -> Dict:
    return {
        "x": [x.to(device) for x in batch["x"]],
        "y": batch["y"].to(device),
        "latent": batch["latent"].to(device),
        "state": batch["state"].to(device),
    }


def _active_modalities(active_modalities: Sequence[int] | None, n_modalities: int) -> Tuple[int, ...]:
    return tuple(range(n_modalities)) if active_modalities is None else tuple(active_modalities)


def _select_xs(xs: List[torch.Tensor], active_modalities: Sequence[int] | None) -> List[torch.Tensor]:
    if active_modalities is None:
        return xs
    return [xs[m] for m in active_modalities]


def _drop_modalities(xs: List[torch.Tensor], p: float) -> List[torch.Tensor]:
    if p <= 0:
        return xs
    batch = xs[0].shape[0]
    n_modalities = len(xs)
    device = xs[0].device
    keep = (torch.rand(batch, n_modalities, device=device) > p).float()
    empty = keep.sum(dim=1) == 0
    if bool(empty.any()):
        keep[empty, :] = 1.0
    return [x * keep[:, m : m + 1] for m, x in enumerate(xs)]


@torch.no_grad()
def _val_loss(model: nn.Module, split: ADSplit, cfg: TrainConfig) -> float:
    model.eval()
    vals: List[float] = []
    for batch0 in _loader(split, cfg, shuffle=False):
        batch = _to_device(batch0, cfg.device)
        vals.append(float(F.cross_entropy(model(batch["x"]), batch["y"]).detach().cpu()))
    return float(np.mean(vals)) if vals else float("inf")


def train_classifier(model: nn.Module, train: ADSplit, val: ADSplit, cfg: TrainConfig, name: str) -> nn.Module:
    model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_state = None
    best_val = float("inf")
    for _ in tqdm(range(cfg.epochs), desc=name, leave=False, disable=not cfg.verbose):
        model.train()
        for batch0 in _loader(train, cfg, shuffle=True):
            batch = _to_device(batch0, cfg.device)
            loss = F.cross_entropy(model(batch["x"]), batch["y"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val_loss = _val_loss(model, val, cfg)
        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _warmup_val_loss(model: WarmupStateNet, split: ADSplit, cfg: TrainConfig, active_modalities: Sequence[int] | None = None) -> float:
    model.eval()
    vals: List[float] = []
    for batch0 in _loader(split, cfg, shuffle=False):
        batch = _to_device(batch0, cfg.device)
        out = model(_select_xs(batch["x"], active_modalities))
        loss = F.cross_entropy(out["logits"], batch["y"])
        if cfg.warmup_proto_weight > 0:
            loss = loss + cfg.warmup_proto_weight * F.mse_loss(out["proto_mu"], batch["state"])
        vals.append(float(loss.detach().cpu()))
    return float(np.mean(vals)) if vals else float("inf")


def train_warmup(
    train: ADSplit,
    val: ADSplit,
    scfg: ADSCMConfig,
    cfg: TrainConfig,
    active_modalities: Sequence[int] | None = None,
    name: str = "warmup",
) -> WarmupStateNet:
    active = _active_modalities(active_modalities, scfg.n_modalities)
    model = WarmupStateNet(scfg.x_dim, len(active), scfg.k, cfg.hidden_dim, cfg.layers).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_state = None
    best_val = float("inf")
    for _ in tqdm(range(cfg.epochs), desc=name, leave=False, disable=not cfg.verbose):
        model.train()
        for batch0 in _loader(train, cfg, shuffle=True):
            batch = _to_device(batch0, cfg.device)
            xs = _drop_modalities(_select_xs(batch["x"], active), cfg.modality_dropout)
            out = model(xs)
            loss = F.cross_entropy(out["logits"], batch["y"])
            if cfg.warmup_proto_weight > 0:
                loss = loss + cfg.warmup_proto_weight * F.mse_loss(out["proto_mu"], batch["state"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val_loss = _warmup_val_loss(model, val, cfg, active)
        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _proto_for_batch(warmup: WarmupStateNet, batch: Dict, active_modalities: Sequence[int] | None = None) -> torch.Tensor:
    warmup.eval()
    with torch.no_grad():
        return warmup(_select_xs(batch["x"], active_modalities))["proto_mu"].detach()


def _warmup_for_batch(
    warmup: WarmupStateNet,
    batch: Dict,
    active_modalities: Sequence[int] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    warmup.eval()
    with torch.no_grad():
        out = warmup(_select_xs(batch["x"], active_modalities))
        proto = out["proto_mu"].detach()
        setting_probs = torch.softmax(out["logits"], dim=-1).detach()
    return proto, setting_probs


def _mix_setting_probs(setting_probs: torch.Tensor, y: torch.Tensor, mix: float, n_classes: int) -> torch.Tensor:
    if mix <= 0:
        return setting_probs
    mix = min(float(mix), 1.0)
    y_onehot = F.one_hot(y, num_classes=n_classes).float()
    out = (1.0 - mix) * setting_probs + mix * y_onehot
    return out / out.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def _ad_bc_loss(
    model: ADCausalCorrectionNet,
    warmup: WarmupStateNet,
    batch: Dict,
    cfg: TrainConfig,
    active_modalities: Sequence[int] | None = None,
) -> torch.Tensor:
    proto, setting_probs = _warmup_for_batch(warmup, batch, active_modalities)
    if model.conditioned_graph:
        setting_probs = _mix_setting_probs(setting_probs, batch["y"], cfg.graph_label_mix, model.n_classes)
    else:
        setting_probs = None  # type: ignore[assignment]
    xs_full = _select_xs(batch["x"], active_modalities)
    xs = _drop_modalities(xs_full, cfg.modality_dropout)
    out = model(xs, proto, setting_probs=setting_probs, sample=True)
    label = F.cross_entropy(out["logits"], batch["y"])  # type: ignore[arg-type]
    recon = torch.tensor(0.0, device=batch["y"].device)
    for rec, x in zip(out["recon"], xs_full):  # type: ignore[union-attr]
        recon = recon + F.mse_loss(rec, x)
    recon = recon / len(xs_full)
    proto_loss = F.mse_loss(out["proto_recon"], proto)  # type: ignore[arg-type]
    kl_z = kl_standard_normal(out["z_mu"], out["z_logvar"]).mean()  # type: ignore[arg-type]
    kl_u = torch.stack([kl_standard_normal(mu, lv).mean() for mu, lv in zip(out["u_mu"], out["u_logvar"])]).mean()  # type: ignore[arg-type]
    state_anchor = (
        F.mse_loss(out["s"], batch["state"])  # type: ignore[arg-type]
        if cfg.state_anchor_weight > 0
        else torch.tensor(0.0, device=batch["y"].device)
    )
    return (
        cfg.label_weight * label
        + cfg.recon_weight * recon
        + cfg.proto_weight * proto_loss
        + cfg.beta_z * kl_z
        + cfg.beta_u * kl_u
        + cfg.graph_l1_weight * model.graph_l1()
        + cfg.dag_weight * model.dag_penalty()
        + cfg.mask_l1_weight * model.mask_l1()
        + cfg.edge_entropy_weight * model.graph_entropy()
        + cfg.mask_entropy_weight * model.mask_entropy()
        - cfg.graph_separation_weight * model.graph_separation()
        + cfg.state_anchor_weight * state_anchor
    )


@torch.no_grad()
def _ad_bc_val_loss(
    model: ADCausalCorrectionNet,
    warmup: WarmupStateNet,
    split: ADSplit,
    cfg: TrainConfig,
    active_modalities: Sequence[int] | None = None,
) -> float:
    model.eval()
    vals: List[float] = []
    for batch0 in _loader(split, cfg, shuffle=False):
        batch = _to_device(batch0, cfg.device)
        proto, setting_probs = _warmup_for_batch(warmup, batch, active_modalities)
        if not model.conditioned_graph:
            setting_probs = None  # type: ignore[assignment]
        out = model(_select_xs(batch["x"], active_modalities), proto, setting_probs=setting_probs, sample=False)
        loss = F.cross_entropy(out["logits"], batch["y"]) + 0.25 * F.mse_loss(out["proto_recon"], proto)  # type: ignore[arg-type]
        vals.append(float(loss.detach().cpu()))
    return float(np.mean(vals)) if vals else float("inf")


def train_ad_bc_mcsgn(
    train: ADSplit,
    val: ADSplit,
    scfg: ADSCMConfig,
    cfg: TrainConfig,
    warmup: WarmupStateNet,
    active_modalities: Sequence[int] | None = None,
    name: str = "ad_bc_mcsgn",
    conditioned_graph: bool = False,
    decoder_uses_state: bool = True,
) -> ADCausalCorrectionNet:
    active = _active_modalities(active_modalities, scfg.n_modalities)
    model = ADCausalCorrectionNet(
        scfg.x_dim,
        len(active),
        scfg.k,
        cfg.u_dim,
        cfg.hidden_dim,
        cfg.layers,
        conditioned_graph=conditioned_graph,
        decoder_uses_state=decoder_uses_state,
    ).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_state = None
    best_val = float("inf")
    warmup.eval()
    for _ in tqdm(range(cfg.correction_epochs), desc=name, leave=False, disable=not cfg.verbose):
        model.train()
        for batch0 in _loader(train, cfg, shuffle=True):
            batch = _to_device(batch0, cfg.device)
            loss = _ad_bc_loss(model, warmup, batch, cfg, active)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val_loss = _ad_bc_val_loss(model, warmup, val, cfg, active)
        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, split: ADSplit, cfg: TrainConfig) -> Dict[str, np.ndarray]:
    model.eval()
    probs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    states: List[np.ndarray] = []
    for batch0 in _loader(split, cfg, shuffle=False):
        batch = _to_device(batch0, cfg.device)
        probs.append(torch.softmax(model(batch["x"]), dim=-1).cpu().numpy())
        ys.append(batch["y"].cpu().numpy())
        states.append(batch["state"].cpu().numpy())
    return {"prob": np.concatenate(probs), "y": np.concatenate(ys), "state_true": np.concatenate(states), "rep": None}


@torch.no_grad()
def evaluate_warmup(
    model: WarmupStateNet,
    split: ADSplit,
    cfg: TrainConfig,
    active_modalities: Sequence[int] | None = None,
) -> Dict[str, np.ndarray]:
    model.eval()
    probs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    states: List[np.ndarray] = []
    reps: List[np.ndarray] = []
    for batch0 in _loader(split, cfg, shuffle=False):
        batch = _to_device(batch0, cfg.device)
        out = model(_select_xs(batch["x"], active_modalities))
        probs.append(torch.softmax(out["logits"], dim=-1).cpu().numpy())
        ys.append(batch["y"].cpu().numpy())
        states.append(batch["state"].cpu().numpy())
        reps.append(out["proto_mu"].cpu().numpy())
    return {"prob": np.concatenate(probs), "y": np.concatenate(ys), "state_true": np.concatenate(states), "rep": np.concatenate(reps)}


@torch.no_grad()
def evaluate_ad_bc(
    model: ADCausalCorrectionNet,
    warmup: WarmupStateNet,
    split: ADSplit,
    cfg: TrainConfig,
    active_modalities: Sequence[int] | None = None,
) -> Dict[str, np.ndarray]:
    model.eval()
    probs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    states: List[np.ndarray] = []
    reps: List[np.ndarray] = []
    for batch0 in _loader(split, cfg, shuffle=False):
        batch = _to_device(batch0, cfg.device)
        proto, setting_probs = _warmup_for_batch(warmup, batch, active_modalities)
        if not model.conditioned_graph:
            setting_probs = None  # type: ignore[assignment]
        out = model(_select_xs(batch["x"], active_modalities), proto, setting_probs=setting_probs, sample=False)
        probs.append(torch.softmax(out["logits"], dim=-1).cpu().numpy())  # type: ignore[arg-type]
        ys.append(batch["y"].cpu().numpy())
        states.append(batch["state"].cpu().numpy())
        reps.append(out["s"].cpu().numpy())  # type: ignore[union-attr]
    return {"prob": np.concatenate(probs), "y": np.concatenate(ys), "state_true": np.concatenate(states), "rep": np.concatenate(reps)}


def _ad_bc_structure_metrics(
    model: ADCausalCorrectionNet,
    params: ADParams,
    tcfg: TrainConfig,
    true_modality_mask: np.ndarray,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    avg_graph = model.adjacency().detach().cpu().numpy()
    metrics.update(graph_recovery_metrics(avg_graph, params.graph, threshold=tcfg.graph_threshold))
    metrics.update(mask_recovery_metrics(model.state_mask().detach().cpu().numpy(), params.state_mask, prefix="state_mask"))
    metrics.update(mask_recovery_metrics(model.modality_mask().detach().cpu().numpy(), true_modality_mask, prefix="modality_mask"))
    graphs = model.adjacency_by_class().detach().cpu().numpy()
    for c in range(graphs.shape[0]):
        metrics[f"edge_D_to_Z3_c{c}"] = float(graphs[c, 4, 0])
        metrics[f"edge_S_to_Z3_c{c}"] = float(graphs[c, 4, 1])
        metrics[f"edge_Z2_to_Z3_c{c}"] = float(graphs[c, 4, 3])
    metrics["edge_D_to_Z3_mean"] = float(graphs[:, 4, 0].mean())
    metrics["edge_S_to_Z3_mean"] = float(graphs[:, 4, 1].mean())
    metrics["edge_Z2_to_Z3_mean"] = float(graphs[:, 4, 3].mean())
    metrics["graph_entropy"] = float(model.graph_entropy().detach().cpu())
    metrics["mask_entropy"] = float(model.mask_entropy().detach().cpu())
    metrics["graph_separation"] = float(model.graph_separation().detach().cpu())
    return metrics


def multiclass_metrics(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    y = y_true.astype(int).reshape(-1)
    p = np.clip(prob, 1e-8, 1.0)
    pred = p.argmax(axis=1)
    out = {
        "acc": float((pred == y).mean()),
        "nll": float(-np.log(p[np.arange(y.shape[0]), y]).mean()),
    }
    try:
        from sklearn.metrics import f1_score, roc_auc_score

        out["macro_f1"] = float(f1_score(y, pred, average="macro"))
        out["macro_auc_ovr"] = float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
    except Exception:
        out["macro_f1"] = float("nan")
        out["macro_auc_ovr"] = float("nan")
    return out


def metrics_from_eval(method: str, split_name: str, out: Dict[str, np.ndarray]) -> Dict[str, float | str]:
    row: Dict[str, float | str] = {"method": method, "split": split_name}
    row.update(multiclass_metrics(out["y"], out["prob"]))
    row.update(state_recovery_metrics(out.get("rep"), out["state_true"]))
    return row


def _evaluate_rows(method: str, outputs: Mapping[str, Dict[str, np.ndarray]]) -> List[Dict[str, float | str]]:
    return [metrics_from_eval(method, split_name, out) for split_name, out in outputs.items()]


def train_and_evaluate_ad(
    splits: Mapping[str, ADSplit],
    scfg: ADSCMConfig,
    tcfg: TrainConfig,
    methods: Sequence[str],
    checkpoint_dir: str | None = None,
    params: ADParams | None = None,
) -> List[Dict[str, float | str]]:
    _set_seed(tcfg.seed)
    rows: List[Dict[str, float | str]] = []
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    baseline_specs = {
        "concat": lambda: ConcatClassifier(scfg.x_dim, range(scfg.n_modalities), tcfg.hidden_dim, tcfg.layers),
        "no_demo": lambda: ConcatClassifier(scfg.x_dim, [0, 1, 2], tcfg.hidden_dim, tcfg.layers),
        "demo_only": lambda: ConcatClassifier(scfg.x_dim, [3], tcfg.hidden_dim, tcfg.layers),
        "late_fusion": lambda: LateFusionClassifier(scfg.x_dim, scfg.n_modalities, tcfg.hidden_dim, tcfg.layers),
    }
    allowed = set(baseline_specs) | {
        "warmup",
        "ad_bc_mcsgn",
        "ad_bc_mcsgn_scond",
        "warmup_no_demo",
        "ad_bc_mcsgn_no_demo",
    }
    for method in methods:
        if method not in allowed:
            raise ValueError(f"Unknown AD method {method!r}; expected {sorted(allowed)}")

    for method in methods:
        if method not in baseline_specs:
            continue
        model = train_classifier(baseline_specs[method](), splits["train"], splits["val"], tcfg, name=method)
        if checkpoint_dir:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"{method}.pt"))
        outputs = {split_name: evaluate(model, split, tcfg) for split_name, split in splits.items()}
        rows.extend(_evaluate_rows(method, outputs))

    warmup: WarmupStateNet | None = None
    if "warmup" in methods or "ad_bc_mcsgn" in methods or "ad_bc_mcsgn_scond" in methods:
        warmup = train_warmup(splits["train"], splits["val"], scfg, tcfg)
        if checkpoint_dir:
            torch.save(warmup.state_dict(), os.path.join(checkpoint_dir, "warmup.pt"))

    if "warmup" in methods and warmup is not None:
        outputs = {split_name: evaluate_warmup(warmup, split, tcfg) for split_name, split in splits.items()}
        rows.extend(_evaluate_rows("warmup", outputs))

    if "ad_bc_mcsgn" in methods and warmup is not None:
        model = train_ad_bc_mcsgn(splits["train"], splits["val"], scfg, tcfg, warmup)
        if checkpoint_dir:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "ad_bc_mcsgn.pt"))
        outputs = {split_name: evaluate_ad_bc(model, warmup, split, tcfg) for split_name, split in splits.items()}
        bc_rows = _evaluate_rows("ad_bc_mcsgn", outputs)
        if params is not None:
            structure_metrics = _ad_bc_structure_metrics(model, params, tcfg, params.modality_mask)
            for row in bc_rows:
                row.update(structure_metrics)
        rows.extend(bc_rows)

    if "ad_bc_mcsgn_scond" in methods and warmup is not None:
        model = train_ad_bc_mcsgn(
            splits["train"],
            splits["val"],
            scfg,
            tcfg,
            warmup,
            name="ad_bc_mcsgn_scond",
            conditioned_graph=True,
            decoder_uses_state=False,
        )
        if checkpoint_dir:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "ad_bc_mcsgn_scond.pt"))
        outputs = {split_name: evaluate_ad_bc(model, warmup, split, tcfg) for split_name, split in splits.items()}
        bc_rows = _evaluate_rows("ad_bc_mcsgn_scond", outputs)
        if params is not None:
            structure_metrics = _ad_bc_structure_metrics(model, params, tcfg, params.modality_mask)
            for row in bc_rows:
                row.update(structure_metrics)
        rows.extend(bc_rows)

    active_no_demo = (0, 1, 2)
    warmup_no_demo: WarmupStateNet | None = None
    if "warmup_no_demo" in methods or "ad_bc_mcsgn_no_demo" in methods:
        warmup_no_demo = train_warmup(
            splits["train"],
            splits["val"],
            scfg,
            tcfg,
            active_modalities=active_no_demo,
            name="warmup_no_demo",
        )
        if checkpoint_dir:
            torch.save(warmup_no_demo.state_dict(), os.path.join(checkpoint_dir, "warmup_no_demo.pt"))

    if "warmup_no_demo" in methods and warmup_no_demo is not None:
        outputs = {
            split_name: evaluate_warmup(warmup_no_demo, split, tcfg, active_modalities=active_no_demo)
            for split_name, split in splits.items()
        }
        rows.extend(_evaluate_rows("warmup_no_demo", outputs))

    if "ad_bc_mcsgn_no_demo" in methods and warmup_no_demo is not None:
        model = train_ad_bc_mcsgn(
            splits["train"],
            splits["val"],
            scfg,
            tcfg,
            warmup_no_demo,
            active_modalities=active_no_demo,
            name="ad_bc_mcsgn_no_demo",
        )
        if checkpoint_dir:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "ad_bc_mcsgn_no_demo.pt"))
        outputs = {
            split_name: evaluate_ad_bc(model, warmup_no_demo, split, tcfg, active_modalities=active_no_demo)
            for split_name, split in splits.items()
        }
        bc_rows = _evaluate_rows("ad_bc_mcsgn_no_demo", outputs)
        if params is not None:
            true_modality_mask = params.modality_mask[list(active_no_demo)]
            structure_metrics = _ad_bc_structure_metrics(model, params, tcfg, true_modality_mask)
            for row in bc_rows:
                row.update(structure_metrics)
        rows.extend(bc_rows)
    return rows
