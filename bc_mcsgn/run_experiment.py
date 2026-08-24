from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch

from .data import SCMConfig, load_fixed_dataset, save_fixed_dataset, split_missing_label_correlations, split_sizes, split_u_label_correlations
from .figures import make_figures
from .train import TrainConfig, train_and_evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run BC-MCSGN fixed-SCM synthetic experiments.")
    p.add_argument("--output-dir", type=str, default="outputs/bc_mcsgn_run")
    p.add_argument("--dataset-path", type=str, default=None)
    p.add_argument("--generate-only", action="store_true")
    p.add_argument("--force-generate", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-train", type=int, default=20_000)
    p.add_argument("--n-val", type=int, default=5_000)
    p.add_argument("--n-test", type=int, default=5_000)
    p.add_argument("--x-dim", type=int, default=10)
    p.add_argument("--u-dim", type=int, default=2)
    p.add_argument("--alpha", type=float, default=1.5)
    p.add_argument("--ood-var-scale", type=float, default=2.5)
    p.add_argument("--ood-a-modality", type=int, default=0)
    p.add_argument("--ood-b-modality", type=int, default=1)
    p.add_argument("--ood-c-modality", type=int, default=2)
    p.add_argument("--ood-d-modality", type=int, default=3)
    p.add_argument("--ood-a-alpha-scale", type=float, default=0.35)
    p.add_argument("--ood-b-mean-shift", type=float, default=1.25)
    p.add_argument("--ood-d-base-drop", type=float, default=0.25)
    p.add_argument("--ood-d-gap-scale", type=float, default=0.50)
    p.add_argument("--noise-std", type=float, default=0.15)
    p.add_argument("--proto-noise-std", type=float, default=0.10)
    p.add_argument("--delta-strength", type=float, default=0.85)

    p.add_argument("--methods", type=str, default="concat,warmup,bc_mcsgn")
    p.add_argument("--warmup-epochs", type=int, default=40)
    p.add_argument("--correction-epochs", type=int, default=80)
    p.add_argument("--baseline-epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--modality-dropout", type=float, default=0.20)
    p.add_argument("--lambda-cm", type=float, default=0.10)
    p.add_argument("--warmup-proto-weight", type=float, default=1.0)
    p.add_argument("--recon-weight", type=float, default=0.50)
    p.add_argument("--proto-weight", type=float, default=1.0)
    p.add_argument("--beta-z", type=float, default=1e-3)
    p.add_argument("--beta-u", type=float, default=1e-3)
    p.add_argument("--graph-l1-weight", type=float, default=1e-3)
    p.add_argument("--dag-weight", type=float, default=1e-3)
    p.add_argument("--mask-l1-weight", type=float, default=1e-3)
    p.add_argument("--state-anchor-weight", type=float, default=0.0)
    p.add_argument("--delta-anchor-weight", type=float, default=0.0)
    p.add_argument("--proto-source", type=str, default="warmup", choices=["warmup", "true_biased"])
    p.add_argument("--fixed-graph", action="store_true")
    p.add_argument("--fixed-masks", action="store_true")
    p.add_argument("--graph-threshold", type=float, default=0.15)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path) if args.dataset_path else output_dir / "data" / f"bc_mcsgn_seed{args.seed}.npz"

    scfg = SCMConfig(
        seed=args.seed,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        x_dim=args.x_dim,
        u_dim=args.u_dim,
        alpha=args.alpha,
        ood_var_scale=args.ood_var_scale,
        ood_a_modality=args.ood_a_modality,
        ood_b_modality=args.ood_b_modality,
        ood_c_modality=args.ood_c_modality,
        ood_d_modality=args.ood_d_modality,
        ood_a_alpha_scale=args.ood_a_alpha_scale,
        ood_b_mean_shift=args.ood_b_mean_shift,
        ood_d_base_drop=args.ood_d_base_drop,
        ood_d_gap_scale=args.ood_d_gap_scale,
        noise_std=args.noise_std,
        proto_noise_std=args.proto_noise_std,
        delta_strength=args.delta_strength,
    )
    if args.force_generate or not dataset_path.exists():
        splits, params = save_fixed_dataset(dataset_path, scfg)
    else:
        splits, params, scfg = load_fixed_dataset(dataset_path)

    diagnostics = {
        "split_sizes": split_sizes(splits),
        "u_label_correlations": {name: split_u_label_correlations(split) for name, split in splits.items()},
        "missing_label_correlations": {name: split_missing_label_correlations(split) for name, split in splits.items()},
    }
    with open(output_dir / "dataset_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)
    with open(output_dir / "config.json", "w") as f:
        json.dump(
            {
                "dataset_path": str(dataset_path),
                "synthetic": scfg.to_dict(),
                "diagnostics": diagnostics,
            },
            f,
            indent=2,
        )
    if args.generate_only:
        print(json.dumps({"dataset_path": str(dataset_path), "diagnostics": diagnostics}, indent=2)[:4000])
        return

    tcfg = TrainConfig(
        warmup_epochs=args.warmup_epochs,
        correction_epochs=args.correction_epochs,
        baseline_epochs=args.baseline_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        device=_resolve_device(args.device),
        modality_dropout=args.modality_dropout,
        lambda_cm=args.lambda_cm,
        warmup_proto_weight=args.warmup_proto_weight,
        recon_weight=args.recon_weight,
        proto_weight=args.proto_weight,
        beta_z=args.beta_z,
        beta_u=args.beta_u,
        graph_l1_weight=args.graph_l1_weight,
        dag_weight=args.dag_weight,
        mask_l1_weight=args.mask_l1_weight,
        state_anchor_weight=args.state_anchor_weight,
        delta_anchor_weight=args.delta_anchor_weight,
        proto_source=args.proto_source,
        fixed_graph=args.fixed_graph,
        fixed_masks=args.fixed_masks,
        graph_threshold=args.graph_threshold,
        seed=args.seed,
        verbose=args.verbose,
    )
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(tcfg.to_dict(), f, indent=2)

    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    rows = train_and_evaluate(splits, params, scfg, tcfg, str(output_dir / "checkpoints"), methods=methods)
    df = pd.DataFrame(rows)
    metrics_csv = output_dir / "metrics.csv"
    df.to_csv(metrics_csv, index=False)

    summary = {
        "dataset_path": str(dataset_path),
        "metrics_csv": str(metrics_csv),
        "methods": list(methods),
        "device": tcfg.device,
        "test_auc": df.pivot(index="method", columns="split", values="auc").to_dict(orient="index"),
    }
    if not args.no_plots:
        summary["figures"] = make_figures(metrics_csv, output_dir / "figures")
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2)[:4000])


if __name__ == "__main__":
    main()
