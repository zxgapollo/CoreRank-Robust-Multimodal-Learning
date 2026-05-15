from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _load_results(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    summary_rows: List[dict] = []
    subset_rows: List[pd.DataFrame] = []
    gates: Dict[str, List[np.ndarray]] = {}
    true_fp: Dict[str, np.ndarray] = {}

    for run_dir in sorted(base_dir.glob("*_seed*")):
        if not run_dir.is_dir():
            continue
        try:
            scenario, seed_text = run_dir.name.rsplit("_seed", 1)
            seed = int(seed_text)
        except ValueError:
            continue

        summary_path = run_dir / "summary.json"
        subset_path = run_dir / "subset_metrics.csv"
        if not summary_path.exists() or not subset_path.exists():
            continue

        summary = json.loads(summary_path.read_text())
        corerank = summary["corerank"]
        full_test = corerank["full_test"]
        erm_full = summary.get("erm", {}).get("erm_full_test", {})
        gate_metrics = corerank["gate_metrics"]
        summary_rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "corerank_auroc": full_test.get("auroc", np.nan),
                "erm_auroc": erm_full.get("auroc", np.nan),
                "corerank_accuracy": full_test.get("accuracy", np.nan),
                "erm_accuracy": erm_full.get("accuracy", np.nan),
                "latent_r2": full_test.get("latent_r2", np.nan),
                "latent_mcc": full_test.get("latent_mcc", np.nan),
                "rank_logdet": full_test.get("rank_logdet", np.nan),
                "effective_rank": full_test.get("effective_rank", np.nan),
                "true_rank_logdet": full_test.get("true_rank_logdet", np.nan),
                "true_effective_rank": full_test.get("true_effective_rank", np.nan),
                "bias_leakage_r2": full_test.get("bias_leakage_r2", np.nan),
                "domain_leakage_r2": full_test.get("domain_leakage_r2", np.nan),
                "gate_f1": gate_metrics.get("gate_f1", np.nan),
            }
        )

        subsets = pd.read_csv(subset_path, dtype={"subset": str})
        subsets["scenario"] = scenario
        subsets["seed"] = seed
        subset_rows.append(subsets)

        gates.setdefault(scenario, []).append(np.asarray(corerank["final_gates"], dtype=float))
        true_fp.setdefault(scenario, np.asarray(corerank["true_footprint"], dtype=float))

    if not summary_rows:
        raise RuntimeError(f"No complete result runs found under {base_dir}")

    summary_df = pd.DataFrame(summary_rows).sort_values(["scenario", "seed"])
    subset_df = pd.concat(subset_rows, ignore_index=True)
    mean_gates = {scenario: np.stack(vals).mean(axis=0) for scenario, vals in gates.items()}
    return summary_df, subset_df, mean_gates, true_fp


def _mean_sem(df: pd.DataFrame, group_cols: list[str], value: str) -> pd.DataFrame:
    out = df.groupby(group_cols)[value].agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    return out


def make_plots(base_dir: Path, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    summary_df, subset_df, mean_gates, true_fp = _load_results(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_dir / "full_test_summary.csv", index=False)
    subset_df.to_csv(out_dir / "all_subset_metrics.csv", index=False)

    scenario_order = ["complementary", "redundant", "biased", "domain"]
    scenarios = [s for s in scenario_order if s in set(summary_df["scenario"])]
    scenario_labels = {"complementary": "Complementary", "redundant": "Redundant", "biased": "Biased", "domain": "Domain"}
    colors = {"CoreRank": "#2f6fbb", "ERM": "#cc6b49"}

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(scenarios))
    width = 0.34
    core = _mean_sem(summary_df, ["scenario"], "corerank_auroc").set_index("scenario").reindex(scenarios)
    erm = _mean_sem(summary_df, ["scenario"], "erm_auroc").set_index("scenario").reindex(scenarios)
    ax.bar(x - width / 2, core["mean"], width, yerr=core["sem"], capsize=3, label="CoreRank", color=colors["CoreRank"])
    ax.bar(x + width / 2, erm["mean"], width, yerr=erm["sem"], capsize=3, label="ERM fusion", color=colors["ERM"])
    ax.set_xticks(x, [scenario_labels[s] for s in scenarios])
    ax.set_ylabel("Full-modality test AUROC")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)
    ax.set_title("Robust prediction across synthetic scenarios")
    fig.tight_layout()
    fig.savefig(out_dir / "full_test_auroc_by_scenario.png")
    plt.close(fig)

    test_subset = subset_df[subset_df["split"] == "test"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharex=True)
    for scenario in scenarios:
        sdf = _mean_sem(test_subset[test_subset["scenario"] == scenario], ["subset_size"], "auroc")
        axes[0].errorbar(sdf["subset_size"], sdf["mean"], yerr=sdf["sem"], marker="o", capsize=3, label=scenario_labels[scenario])
        rdf = _mean_sem(test_subset[test_subset["scenario"] == scenario], ["subset_size"], "latent_r2")
        axes[1].errorbar(rdf["subset_size"], rdf["mean"], yerr=rdf["sem"], marker="o", capsize=3, label=scenario_labels[scenario])
    axes[0].set_ylabel("Test AUROC")
    axes[1].set_ylabel("Latent linear R2")
    for ax in axes:
        ax.set_xlabel("Number of observed modalities")
        ax.set_xticks([1, 2, 3])
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Missing-modality degradation curves")
    fig.tight_layout()
    fig.savefig(out_dir / "subset_size_curves.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    for scenario in scenarios:
        sdf = test_subset[test_subset["scenario"] == scenario]
        axes[0].scatter(sdf["rank_logdet"], sdf["latent_r2"], s=28, alpha=0.75, label=scenario_labels[scenario])
        axes[1].scatter(sdf["effective_rank"], sdf["latent_r2"], s=28, alpha=0.75, label=scenario_labels[scenario])
    axes[0].set_xlabel("Normalized logdet rank score")
    axes[0].set_ylabel("Latent linear R2")
    axes[1].set_xlabel("Effective rank")
    axes[1].set_ylabel("Latent linear R2")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Rank diagnostics vs. core recovery")
    fig.tight_layout()
    fig.savefig(out_dir / "rank_vs_recovery.png")
    plt.close(fig)

    if {"true_rank_logdet", "true_effective_rank"}.issubset(test_subset.columns) and test_subset["true_rank_logdet"].notna().any():
        fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
        for scenario in scenarios:
            sdf = test_subset[test_subset["scenario"] == scenario]
            axes[0].scatter(sdf["true_rank_logdet"], sdf["latent_r2"], s=28, alpha=0.75, label=scenario_labels[scenario])
            axes[1].scatter(sdf["true_effective_rank"], sdf["latent_r2"], s=28, alpha=0.75, label=scenario_labels[scenario])
        axes[0].set_xlabel("Oracle normalized logdet rank score")
        axes[0].set_ylabel("Latent linear R2")
        axes[1].set_xlabel("Oracle effective rank")
        axes[1].set_ylabel("Latent linear R2")
        for ax in axes:
            ax.grid(alpha=0.25)
        axes[0].legend(frameon=False)
        fig.suptitle("Oracle rank diagnostics vs. core recovery")
        fig.tight_layout()
        fig.savefig(out_dir / "true_rank_vs_recovery.png")
        plt.close(fig)

    if "biased" in scenarios:
        biased = test_subset[test_subset["scenario"] == "biased"].copy()
        full_biased = summary_df[summary_df["scenario"] == "biased"].copy()
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
        vals = [
            full_biased["corerank_auroc"].dropna().to_numpy(),
            full_biased["erm_auroc"].dropna().to_numpy(),
        ]
        axes[0].boxplot(vals, tick_labels=["CoreRank", "ERM"], patch_artist=True)
        axes[0].scatter(np.repeat(1, len(vals[0])), vals[0], color=colors["CoreRank"], zorder=3)
        axes[0].scatter(np.repeat(2, len(vals[1])), vals[1], color=colors["ERM"], zorder=3)
        axes[0].set_ylabel("OOD test AUROC")
        axes[0].set_title("Biased scenario")
        leak = _mean_sem(biased, ["subset_size"], "bias_leakage_r2")
        axes[1].errorbar(leak["subset_size"], leak["mean"], yerr=leak["sem"], marker="o", capsize=3, color="#6b5ca5")
        axes[1].set_xticks([1, 2, 3])
        axes[1].set_xlabel("Number of observed modalities")
        axes[1].set_ylabel("Bias leakage R2 from z")
        axes[1].set_title("Lower leakage is better")
        for ax in axes:
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "biased_ood_and_leakage.png")
        plt.close(fig)

    if "domain" in scenarios:
        domain = test_subset[test_subset["scenario"] == "domain"].copy()
        full_domain = summary_df[summary_df["scenario"] == "domain"].copy()
        fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
        vals = [
            full_domain["corerank_auroc"].dropna().to_numpy(),
            full_domain["erm_auroc"].dropna().to_numpy(),
        ]
        axes[0].boxplot(vals, tick_labels=["CoreRank", "ERM"], patch_artist=True)
        axes[0].scatter(np.repeat(1, len(vals[0])), vals[0], color=colors["CoreRank"], zorder=3)
        axes[0].scatter(np.repeat(2, len(vals[1])), vals[1], color=colors["ERM"], zorder=3)
        axes[0].set_ylabel("Shifted-domain test AUROC")
        axes[0].set_title("Domain scenario")
        leak = _mean_sem(domain, ["subset_size"], "domain_leakage_r2")
        axes[1].errorbar(leak["subset_size"], leak["mean"], yerr=leak["sem"], marker="o", capsize=3, color="#5d7f55")
        axes[1].set_xticks([1, 2, 3])
        axes[1].set_xlabel("Number of observed modalities")
        axes[1].set_ylabel("Domain leakage R2 from z")
        axes[1].set_title("Lower leakage is better")
        for ax in axes:
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "domain_shift_and_leakage.png")
        plt.close(fig)

    fig, axes = plt.subplots(len(scenarios), 2, figsize=(7.8, 8.4), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row, scenario in enumerate(scenarios):
        for col, (title, mat) in enumerate([("Learned mean gate", mean_gates[scenario]), ("True footprint", true_fp[scenario])]):
            ax = axes[row, col]
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax.set_title(f"{scenario_labels[scenario]}: {title}")
            ax.set_xlabel("Core dimension")
            ax.set_ylabel("Modality")
            ax.set_xticks(range(mat.shape[1]))
            ax.set_yticks(range(mat.shape[0]))
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white" if mat[i, j] < 0.65 else "black", fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.75, label="Gate value")
    fig.savefig(out_dir / "gate_heatmaps.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CoreRank synthetic experiment results.")
    parser.add_argument("--base-dir", type=Path, default=Path("outputs/main_grid_gpu"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or args.base_dir / "figures"
    make_plots(args.base_dir, out_dir)
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
