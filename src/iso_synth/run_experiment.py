from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from .data import ISODataConfig, SCENARIOS, canonical_scenario, make_iso_data
from .figures import make_core_figures
from .train import TrainConfig, run_iso_suite


def _csv_list(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def _int_list(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ISO synthetic multimodal experiments.")
    p.add_argument("--scenarios", type=str, default=",".join(SCENARIOS), help="Comma-separated scenarios.")
    p.add_argument("--seeds", type=str, default="0", help="Comma-separated random seeds.")
    p.add_argument("--n-train-grid", type=str, default="128,512", help="Comma-separated training sample sizes.")
    p.add_argument("--n-val", type=int, default=256)
    p.add_argument("--n-test", type=int, default=512)
    p.add_argument("--s-dim", type=int, default=6)
    p.add_argument("--u-dim", type=int, default=3)
    p.add_argument("--n-modalities", type=int, default=3)
    p.add_argument("--x-dim", type=int, default=16)
    p.add_argument("--noise-std", type=float, default=0.35)
    p.add_argument("--noisy-noise-std", type=float, default=1.8)
    p.add_argument("--train-shortcut-corr", type=float, default=0.85)
    p.add_argument("--test-shortcut-corr", type=float, default=-0.65)
    p.add_argument("--shortcut-strength", type=float, default=0.0)
    p.add_argument("--ood-residual-shift", type=float, default=0.65)
    p.add_argument("--train-nuisance-corr", type=float, default=0.35)
    p.add_argument("--test-nuisance-corr", type=float, default=-0.25)
    p.add_argument("--ood-noise-multiplier", type=float, default=1.35)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--modality-dropout", type=float, default=0.15)
    p.add_argument("--recon-weight", type=float, default=0.25)
    p.add_argument("--beta-kl", type=float, default=1e-3)
    p.add_argument("--state-anchor-weight", type=float, default=0.0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/iso_synth"))
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = [canonical_scenario(item) for item in _csv_list(args.scenarios)]
    invalid = sorted(set(scenarios) - set(SCENARIOS))
    if invalid:
        raise ValueError(f"Unknown scenarios {invalid}; valid scenarios are {SCENARIOS}")
    seeds = _int_list(args.seeds)
    n_train_grid = _int_list(args.n_train_grid)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_root = args.output_dir / "runs"
    run_root.mkdir(parents=True, exist_ok=True)

    all_results: List[pd.DataFrame] = []
    for scenario in scenarios:
        for n_train in n_train_grid:
            for seed in seeds:
                run_dir = run_root / f"{scenario}_n{n_train}_seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                scfg = ISODataConfig(
                    scenario=scenario,
                    seed=seed,
                    n_train=n_train,
                    n_val=args.n_val,
                    n_test=args.n_test,
                    s_dim=args.s_dim,
                    u_dim=args.u_dim,
                    n_modalities=args.n_modalities,
                    x_dim=args.x_dim,
                    noise_std=args.noise_std,
                    noisy_noise_std=args.noisy_noise_std,
                    train_shortcut_corr=args.train_shortcut_corr,
                    test_shortcut_corr=args.test_shortcut_corr,
                    shortcut_strength=args.shortcut_strength,
                    ood_residual_shift=args.ood_residual_shift,
                    train_nuisance_corr=args.train_nuisance_corr,
                    test_nuisance_corr=args.test_nuisance_corr,
                    ood_noise_multiplier=args.ood_noise_multiplier,
                )
                tcfg = TrainConfig(
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    hidden_dim=args.hidden_dim,
                    device=device,
                    modality_dropout=args.modality_dropout,
                    recon_weight=args.recon_weight,
                    beta_kl=args.beta_kl,
                    state_anchor_weight=args.state_anchor_weight,
                    seed=seed,
                    verbose=args.verbose,
                )
                with open(run_dir / "config.json", "w") as f:
                    json.dump({"synthetic": scfg.to_dict(), "train": asdict(tcfg)}, f, indent=2)
                train, val, id_test, test, params = make_iso_data(scfg, include_id_test=True)
                np.save(run_dir / "footprint.npy", params.footprint)
                np.save(run_dir / "state_graph.npy", params.state_graph)
                np.save(run_dir / "beta_y.npy", params.beta_y)
                print(f"[ISO] scenario={scenario} n_train={n_train} seed={seed} device={device}")
                df = run_iso_suite(train, val, id_test, test, params, scfg, tcfg, output_dir=str(run_dir))
                all_results.append(df)

    results = pd.concat(all_results, ignore_index=True)
    results_path = args.output_dir / "results.csv"
    results.to_csv(results_path, index=False)
    if not args.no_figures:
        make_core_figures(results_path, args.output_dir / "figures")
    print(f"Wrote {results_path}")
    if not args.no_figures:
        print(f"Wrote figures to {args.output_dir / 'figures'}")


if __name__ == "__main__":
    main()
