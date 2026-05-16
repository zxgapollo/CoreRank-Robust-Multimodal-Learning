from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

from .data import SyntheticConfig, make_synthetic_data
from .train import TrainConfig, train_corerank, train_erm_baseline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CoreRank synthetic benchmark.")
    p.add_argument("--scenario", type=str, default="complementary", choices=["complementary", "redundant", "biased", "domain"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--z-dim", type=int, default=6)
    p.add_argument("--u-dim", type=int, default=3)
    p.add_argument("--n-modalities", type=int, default=3)
    p.add_argument("--x-dim", type=int, default=16)
    p.add_argument("--noise-std", type=float, default=0.35)
    p.add_argument("--bias-strength", type=float, default=None)
    p.add_argument("--biased-modality", type=int, default=0)
    p.add_argument("--train-bias-corr", type=float, default=0.85)
    p.add_argument("--test-bias-corr", type=float, default=-0.50)
    p.add_argument("--domain-shift-strength", type=float, default=None)
    p.add_argument("--domain-shifted-modality", type=int, default=0)
    p.add_argument("--core-graph-strength", type=float, default=0.35)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--beta-z", type=float, default=1e-3)
    p.add_argument("--beta-u", type=float, default=1e-3)
    p.add_argument("--recon-weight", type=float, default=1.0)
    p.add_argument("--recon-reduction", type=str, default="mean", choices=["mean", "sum"])
    p.add_argument("--label-weight", type=float, default=1.0)
    p.add_argument("--structural-weight", type=float, default=0.0)
    p.add_argument("--dag-weight", type=float, default=0.1)
    p.add_argument("--graph-l1-weight", type=float, default=0.01)
    p.add_argument("--structural-warmup-epochs", type=int, default=0)
    p.add_argument("--bias-invariance-weight", type=float, default=0.0)
    p.add_argument("--domain-invariance-weight", type=float, default=0.0)
    p.add_argument("--no-sem-prior", action="store_true")
    p.add_argument("--rank-on-z", action="store_true")
    p.add_argument("--no-select-best", action="store_true")
    p.add_argument("--best-id-tolerance", type=float, default=0.02)
    p.add_argument("--best-leakage-weight", type=float, default=0.0)
    p.add_argument("--rank-kappa", type=float, default=0.5)
    p.add_argument("--sparse-budget", type=float, default=9.0)
    p.add_argument("--rho-rank", type=float, default=1.0)
    p.add_argument("--rho-sparse", type=float, default=0.1)
    p.add_argument("--modality-dropout", type=float, default=0.15)
    p.add_argument("--max-fisher-batch", type=int, default=64)
    p.add_argument("--eval-fisher-batches", type=int, default=4)
    p.add_argument("--eval-true-fisher-samples", type=int, default=256)
    p.add_argument("--rank-warmup-epochs", type=int, default=5)
    p.add_argument("--sparse-warmup-epochs", type=int, default=10)
    p.add_argument("--fisher-damping", type=float, default=1e-3)
    p.add_argument("--rank-eps", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--encoder-layers", type=int, default=2)
    p.add_argument("--decoder-layers", type=int, default=2)
    p.add_argument("--gate-temperature", type=float, default=0.67)
    p.add_argument("--init-gate-logit", type=float, default=0.0)
    p.add_argument("--gate-temperature-min", type=float, default=0.2)
    p.add_argument("--gate-anneal-epochs", type=int, default=0)
    p.add_argument("--gate-l1-weight", type=float, default=0.0)
    p.add_argument("--gate-binary-weight", type=float, default=0.0)
    p.add_argument("--no-structural-classifier", action="store_true")
    p.add_argument("--no-rank", action="store_true")
    p.add_argument("--no-sparse", action="store_true")
    p.add_argument("--skip-erm", action="store_true")
    p.add_argument("--output-dir", type=str, default="outputs/run")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    bias_strength = args.bias_strength
    if bias_strength is None:
        bias_strength = 2.0 if args.scenario == "biased" else 0.0
    domain_shift_strength = args.domain_shift_strength
    if domain_shift_strength is None:
        domain_shift_strength = 1.5 if args.scenario == "domain" else 0.0

    scfg = SyntheticConfig(
        scenario=args.scenario,
        seed=args.seed,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        z_dim=args.z_dim,
        u_dim=args.u_dim,
        n_modalities=args.n_modalities,
        x_dim=args.x_dim,
        noise_std=args.noise_std,
        bias_strength=bias_strength,
        biased_modality=args.biased_modality,
        train_bias_corr=args.train_bias_corr,
        test_bias_corr=args.test_bias_corr,
        domain_shift_strength=domain_shift_strength,
        domain_shifted_modality=args.domain_shifted_modality,
        core_graph_strength=args.core_graph_strength,
    )
    tcfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        beta_z=args.beta_z,
        beta_u=args.beta_u,
        recon_weight=args.recon_weight,
        recon_reduction=args.recon_reduction,
        label_weight=args.label_weight,
        structural_weight=args.structural_weight,
        dag_weight=args.dag_weight,
        graph_l1_weight=args.graph_l1_weight,
        structural_warmup_epochs=args.structural_warmup_epochs,
        bias_invariance_weight=args.bias_invariance_weight,
        domain_invariance_weight=args.domain_invariance_weight,
        use_sem_prior=not args.no_sem_prior,
        rank_on_innovation=not args.rank_on_z,
        select_best=not args.no_select_best,
        best_id_tolerance=args.best_id_tolerance,
        best_leakage_weight=args.best_leakage_weight,
        rank_kappa=args.rank_kappa,
        sparse_budget=args.sparse_budget,
        rho_rank=args.rho_rank,
        rho_sparse=args.rho_sparse,
        modality_dropout=args.modality_dropout,
        max_fisher_batch=args.max_fisher_batch,
        eval_fisher_batches=args.eval_fisher_batches,
        eval_true_fisher_samples=args.eval_true_fisher_samples,
        rank_warmup_epochs=args.rank_warmup_epochs,
        sparse_warmup_epochs=args.sparse_warmup_epochs,
        fisher_damping=args.fisher_damping,
        rank_eps=args.rank_eps,
        no_rank=args.no_rank,
        no_sparse=args.no_sparse,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        gate_temperature=args.gate_temperature,
        init_gate_logit=args.init_gate_logit,
        gate_temperature_min=args.gate_temperature_min,
        gate_anneal_epochs=args.gate_anneal_epochs,
        gate_l1_weight=args.gate_l1_weight,
        gate_binary_weight=args.gate_binary_weight,
        structural_classifier=not args.no_structural_classifier,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({"synthetic": scfg.to_dict(), "train": asdict(tcfg)}, f, indent=2)

    train, val, id_test, test, params = make_synthetic_data(scfg, include_id_test=True)
    model, metrics, subset_df = train_corerank(train, val, test, params, scfg, tcfg, args.output_dir, id_test=id_test)
    result = {"corerank": metrics}
    if not args.skip_erm:
        erm_metrics = train_erm_baseline(train, val, test, scfg, tcfg, args.output_dir, epochs=max(3, min(args.epochs, 30)), id_test=id_test)
        result["erm"] = erm_metrics
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2)[:4000])


if __name__ == "__main__":
    main()
