from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .data import ADSCMConfig, class_counts, demo_shortcut_label_corr, make_ad_dataset, save_ad_dataset, shortcut_component_label_corr
from .figures import make_sweep_figures
from .train import TrainConfig, train_and_evaluate_ad


def _float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run AD-SCM multimodal OOD sweeps.")
    p.add_argument("--output-dir", type=str, default="outputs/ad_scm_sweep")
    p.add_argument("--mode", type=str, default="shortcut", choices=["shortcut", "noise"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=2000)
    p.add_argument("--x-dim", type=int, default=8)
    p.add_argument("--base-noise", type=float, default=0.25)
    p.add_argument("--disease-noise-scale", type=float, default=1.0)
    p.add_argument("--demo-noise-scale", type=float, default=1.0)
    p.add_argument("--demo-to-s-strength", type=float, default=0.35)
    p.add_argument("--shortcut-strengths", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0")
    p.add_argument("--shortcut-test-strength", type=float, default=0.1)
    p.add_argument("--noise-train-scale", type=float, default=1.0)
    p.add_argument("--noise-scales", type=str, default="1.0,1.5,2.0,3.0,4.0")
    p.add_argument("--noise-modality", type=str, default="mri")
    p.add_argument("--methods", type=str, default="concat,no_demo,late_fusion")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--correction-epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--u-dim", type=int, default=2)
    p.add_argument("--modality-dropout", type=float, default=0.10)
    p.add_argument("--warmup-proto-weight", type=float, default=0.0)
    p.add_argument("--recon-weight", type=float, default=0.5)
    p.add_argument("--proto-weight", type=float, default=0.5)
    p.add_argument("--beta-z", type=float, default=1e-3)
    p.add_argument("--beta-u", type=float, default=1e-3)
    p.add_argument("--graph-l1-weight", type=float, default=1e-3)
    p.add_argument("--dag-weight", type=float, default=0.0)
    p.add_argument("--mask-l1-weight", type=float, default=1e-3)
    p.add_argument("--edge-entropy-weight", type=float, default=0.0)
    p.add_argument("--mask-entropy-weight", type=float, default=0.0)
    p.add_argument("--graph-separation-weight", type=float, default=0.0)
    p.add_argument("--graph-label-mix", type=float, default=0.0)
    p.add_argument("--state-anchor-weight", type=float, default=0.0)
    p.add_argument("--graph-threshold", type=float, default=0.15)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    tcfg = TrainConfig(
        epochs=args.epochs,
        correction_epochs=args.correction_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        u_dim=args.u_dim,
        modality_dropout=args.modality_dropout,
        warmup_proto_weight=args.warmup_proto_weight,
        recon_weight=args.recon_weight,
        proto_weight=args.proto_weight,
        beta_z=args.beta_z,
        beta_u=args.beta_u,
        graph_l1_weight=args.graph_l1_weight,
        dag_weight=args.dag_weight,
        mask_l1_weight=args.mask_l1_weight,
        edge_entropy_weight=args.edge_entropy_weight,
        mask_entropy_weight=args.mask_entropy_weight,
        graph_separation_weight=args.graph_separation_weight,
        graph_label_mix=args.graph_label_mix,
        state_anchor_weight=args.state_anchor_weight,
        graph_threshold=args.graph_threshold,
        device=_device(args.device),
        seed=args.seed,
        verbose=args.verbose,
    )
    sweep_values = _float_list(args.shortcut_strengths if args.mode == "shortcut" else args.noise_scales)
    all_rows = []
    diagnostics = {}
    for i, value in enumerate(sweep_values):
        if args.mode == "shortcut":
            scfg = ADSCMConfig(
                seed=args.seed,
                n_train=args.n_train,
                n_val=args.n_val,
                n_test=args.n_test,
                x_dim=args.x_dim,
                base_noise=args.base_noise,
                disease_noise_scale=args.disease_noise_scale,
                demo_noise_scale=args.demo_noise_scale,
                demo_to_s_strength=args.demo_to_s_strength,
                shortcut_strength=value,
                shortcut_test_strength=args.shortcut_test_strength,
                noise_train_scale=args.noise_train_scale,
                noise_ood_scale=1.0,
                noise_modality=args.noise_modality,
            )
            sweep_name = "D_to_Z3_train_strength"
        else:
            base_shortcut = _float_list(args.shortcut_strengths)[0]
            scfg = ADSCMConfig(
                seed=args.seed,
                n_train=args.n_train,
                n_val=args.n_val,
                n_test=args.n_test,
                x_dim=args.x_dim,
                base_noise=args.base_noise,
                disease_noise_scale=args.disease_noise_scale,
                demo_noise_scale=args.demo_noise_scale,
                demo_to_s_strength=args.demo_to_s_strength,
                shortcut_strength=base_shortcut,
                shortcut_test_strength=base_shortcut,
                noise_train_scale=value,
                noise_ood_scale=1.0,
                noise_modality=args.noise_modality,
            )
            sweep_name = f"{args.noise_modality}_train_noise_scale_clean_test"
        splits, params = make_ad_dataset(scfg)
        save_ad_dataset(output_dir / "data" / f"{args.mode}_{i:02d}.npz", scfg)
        diagnostics[str(value)] = {
            "class_counts": {name: class_counts(split) for name, split in splits.items()},
            "demo_label_corr": {name: demo_shortcut_label_corr(split) for name, split in splits.items()},
            "shortcut_component_label_corr": {name: shortcut_component_label_corr(split) for name, split in splits.items()},
        }
        rows = train_and_evaluate_ad(splits, scfg, tcfg, methods, checkpoint_dir=None, params=params)
        for row in rows:
            row["sweep_name"] = sweep_name
            row["sweep_value"] = value
            row["mode"] = args.mode
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    metrics_csv = output_dir / "metrics.csv"
    df.to_csv(metrics_csv, index=False)
    with open(output_dir / "config.json", "w") as f:
        json.dump({"args": vars(args), "train": tcfg.to_dict()}, f, indent=2)
    with open(output_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)
    summary = {
        "metrics_csv": str(metrics_csv),
        "mode": args.mode,
        "device": tcfg.device,
        "methods": list(methods),
    }
    if not args.no_plots:
        summary["figures"] = make_sweep_figures(metrics_csv, output_dir / "figures")
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2)[:4000])


if __name__ == "__main__":
    main()
