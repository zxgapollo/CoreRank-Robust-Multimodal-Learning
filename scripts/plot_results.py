from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))
try:
    from corerank_synth.metrics import dag_metrics as _dag_metrics
    from corerank_synth.metrics import directed_graph_metrics as _directed_graph_metrics
except Exception:
    _dag_metrics = None
    _directed_graph_metrics = None


def _load_results(
    base_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    summary_rows: List[dict] = []
    subset_rows: List[pd.DataFrame] = []
    gates: Dict[str, List[np.ndarray]] = {}
    true_fp: Dict[str, np.ndarray] = {}
    core_graphs: Dict[str, List[np.ndarray]] = {}
    true_core_graph: Dict[str, np.ndarray] = {}

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
        id_full_test = corerank.get("id_full_test", {})
        erm_full = summary.get("erm", {}).get("erm_full_test", {})
        erm_id_full = summary.get("erm", {}).get("erm_id_full_test", {})
        gate_metrics = corerank["gate_metrics"]
        graph_metrics = corerank.get("graph_metrics", {})
        dag_diag = corerank.get("dag_metrics", {})
        learned_graph = np.asarray(corerank.get("learned_core_graph", []), dtype=float)
        true_graph = np.asarray(corerank.get("true_core_graph", []), dtype=float)
        if learned_graph.ndim == 2 and learned_graph.size:
            if _dag_metrics is not None and not dag_diag:
                dag_diag = _dag_metrics(learned_graph)
            if _directed_graph_metrics is not None and true_graph.ndim == 2 and true_graph.size:
                computed_graph_metrics = _directed_graph_metrics(learned_graph, true_graph)
                graph_metrics = {**computed_graph_metrics, **graph_metrics}
        summary_rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "corerank_auroc": full_test.get("auroc", np.nan),
                "erm_auroc": erm_full.get("auroc", np.nan),
                "corerank_id_auroc": id_full_test.get("auroc", np.nan),
                "erm_id_auroc": erm_id_full.get("auroc", np.nan),
                "corerank_accuracy": full_test.get("accuracy", np.nan),
                "erm_accuracy": erm_full.get("accuracy", np.nan),
                "latent_r2": full_test.get("latent_r2", np.nan),
                "innovation_r2": full_test.get("innovation_r2", np.nan),
                "latent_mcc": full_test.get("latent_mcc", np.nan),
                "rank_logdet": full_test.get("rank_logdet", np.nan),
                "effective_rank": full_test.get("effective_rank", np.nan),
                "true_rank_logdet": full_test.get("true_rank_logdet", np.nan),
                "true_effective_rank": full_test.get("true_effective_rank", np.nan),
                "bias_leakage_r2": full_test.get("bias_leakage_r2", np.nan),
                "domain_leakage_r2": full_test.get("domain_leakage_r2", np.nan),
                "gate_f1": gate_metrics.get("gate_f1", np.nan),
                "graph_precision": graph_metrics.get("graph_precision", np.nan),
                "graph_recall": graph_metrics.get("graph_recall", np.nan),
                "core_graph_f1": graph_metrics.get("graph_f1", np.nan),
                "core_graph_active": graph_metrics.get("graph_active", np.nan),
                "core_graph_true_edges": graph_metrics.get("graph_true_edges", np.nan),
                "core_graph_skeleton_f1": graph_metrics.get("graph_skeleton_f1", np.nan),
                "core_graph_directed_hamming": graph_metrics.get("graph_directed_hamming", np.nan),
                "core_graph_reversed_edges": graph_metrics.get("graph_reversed_edges", np.nan),
                "core_graph_edge_auroc": graph_metrics.get("graph_edge_auroc", np.nan),
                "core_graph_edge_auprc": graph_metrics.get("graph_edge_auprc", np.nan),
                "dag_acyclicity": dag_diag.get("dag_acyclicity", np.nan),
                "dag_threshold_is_acyclic": dag_diag.get("dag_threshold_is_acyclic", np.nan),
                "dag_active_edges": dag_diag.get("dag_active_edges", np.nan),
                "dag_density": dag_diag.get("dag_density", np.nan),
                "dag_l1": dag_diag.get("dag_l1", np.nan),
                "dag_l2": dag_diag.get("dag_l2", np.nan),
                "dag_max_abs_weight": dag_diag.get("dag_max_abs_weight", np.nan),
                "best_epoch": corerank.get("best_epoch", np.nan),
                "best_val_auroc": corerank.get("best_val_auroc", np.nan),
                "best_val_auc_target": corerank.get("best_val_auc_target", np.nan),
                "best_val_auc_floor": corerank.get("best_val_auc_floor", np.nan),
                "best_val_context_leakage_r2": corerank.get("best_val_context_leakage_r2", np.nan),
                "best_val_robust_score": corerank.get("best_val_robust_score", np.nan),
            }
        )

        subsets = pd.read_csv(subset_path, dtype={"subset": str})
        subsets["scenario"] = scenario
        subsets["seed"] = seed
        subset_rows.append(subsets)

        gates.setdefault(scenario, []).append(np.asarray(corerank["final_gates"], dtype=float))
        true_fp.setdefault(scenario, np.asarray(corerank["true_footprint"], dtype=float))
        if learned_graph.ndim == 2 and learned_graph.size and true_graph.ndim == 2 and true_graph.size:
            core_graphs.setdefault(scenario, []).append(learned_graph)
            true_core_graph.setdefault(scenario, true_graph)

    if not summary_rows:
        raise RuntimeError(f"No complete result runs found under {base_dir}")

    summary_df = pd.DataFrame(summary_rows).sort_values(["scenario", "seed"])
    subset_df = pd.concat(subset_rows, ignore_index=True)
    mean_gates = {scenario: np.stack(vals).mean(axis=0) for scenario, vals in gates.items()}
    mean_core_graphs = {scenario: np.stack(vals).mean(axis=0) for scenario, vals in core_graphs.items()}
    return summary_df, subset_df, mean_gates, true_fp, mean_core_graphs, true_core_graph


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

    summary_df, subset_df, mean_gates, true_fp, mean_core_graphs, true_core_graph = _load_results(base_dir)
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
        axes[1].set_ylabel("Bias leakage R2 from innovation")
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
        axes[1].set_ylabel("Domain leakage R2 from innovation")
        axes[1].set_title("Lower leakage is better")
        for ax in axes:
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "domain_shift_and_leakage.png")
        plt.close(fig)

    if summary_df["corerank_id_auroc"].notna().any():
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
        core_id = _mean_sem(summary_df, ["scenario"], "corerank_id_auroc").set_index("scenario").reindex(scenarios)
        core_ood = _mean_sem(summary_df, ["scenario"], "corerank_auroc").set_index("scenario").reindex(scenarios)
        erm_id = _mean_sem(summary_df, ["scenario"], "erm_id_auroc").set_index("scenario").reindex(scenarios)
        erm_ood = _mean_sem(summary_df, ["scenario"], "erm_auroc").set_index("scenario").reindex(scenarios)
        axes[0].bar(x - width / 2, core_id["mean"], width, yerr=core_id["sem"], capsize=3, label="CoreRank", color=colors["CoreRank"])
        axes[0].bar(x + width / 2, erm_id["mean"], width, yerr=erm_id["sem"], capsize=3, label="ERM", color=colors["ERM"])
        axes[1].bar(x - width / 2, core_ood["mean"], width, yerr=core_ood["sem"], capsize=3, label="CoreRank", color=colors["CoreRank"])
        axes[1].bar(x + width / 2, erm_ood["mean"], width, yerr=erm_ood["sem"], capsize=3, label="ERM", color=colors["ERM"])
        for ax, title in zip(axes, ["ID test", "OOD / shifted test"]):
            ax.set_xticks(x, [scenario_labels[s] for s in scenarios], rotation=15)
            ax.set_ylim(0.0, 1.0)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel("Full-modality AUROC")
        axes[0].legend(frameon=False)
        fig.suptitle("ID vs. OOD performance")
        fig.tight_layout()
        fig.savefig(out_dir / "id_vs_ood_auroc.png")
        plt.close(fig)

    if summary_df["dag_acyclicity"].notna().any():
        fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.7))
        dag_h = _mean_sem(summary_df, ["scenario"], "dag_acyclicity").set_index("scenario").reindex(scenarios)
        dag_edges = _mean_sem(summary_df, ["scenario"], "dag_active_edges").set_index("scenario").reindex(scenarios)
        skel = _mean_sem(summary_df, ["scenario"], "core_graph_skeleton_f1").set_index("scenario").reindex(scenarios)
        directed = _mean_sem(summary_df, ["scenario"], "core_graph_f1").set_index("scenario").reindex(scenarios)
        axes[0].bar(x, dag_h["mean"], yerr=dag_h["sem"], capsize=3, color="#6b5ca5")
        axes[0].set_title("DAG penalty")
        axes[0].set_ylabel("NOTEARS h(A)")
        axes[1].bar(x, dag_edges["mean"], yerr=dag_edges["sem"], capsize=3, color="#5d7f55")
        axes[1].set_title("Active edges")
        axes[1].set_ylabel("Edges >= threshold")
        axes[2].bar(x, skel["mean"], yerr=skel["sem"], capsize=3, color="#a67843")
        axes[2].set_title("Skeleton recovery")
        axes[2].set_ylabel("Skeleton F1")
        axes[3].bar(x, directed["mean"], yerr=directed["sem"], capsize=3, color="#2f6fbb")
        axes[3].set_title("Directed recovery")
        axes[3].set_ylabel("Directed F1")
        for ax in axes:
            ax.set_xticks(x, [scenario_labels[s] for s in scenarios], rotation=20)
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle("Structural graph diagnostics")
        fig.tight_layout()
        fig.savefig(out_dir / "dag_graph_diagnostics.png")
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

    if mean_core_graphs:
        graph_scenarios = [s for s in scenarios if s in mean_core_graphs and s in true_core_graph]
        fig, axes = plt.subplots(len(graph_scenarios), 2, figsize=(7.8, 8.4), constrained_layout=True)
        axes = np.atleast_2d(axes)
        vmax = max(float(np.max(np.abs(mean_core_graphs[s]))) for s in graph_scenarios)
        vmax = max(vmax, 0.05)
        for row, scenario in enumerate(graph_scenarios):
            for col, (title, mat) in enumerate([
                ("Learned mean core graph", mean_core_graphs[scenario]),
                ("True core graph", true_core_graph[scenario]),
            ]):
                ax = axes[row, col]
                im = ax.imshow(mat, vmin=-vmax, vmax=vmax, cmap="coolwarm", aspect="equal")
                ax.set_title(f"{scenario_labels[scenario]}: {title}")
                ax.set_xlabel("Parent core dimension")
                ax.set_ylabel("Child core dimension")
                ax.set_xticks(range(mat.shape[1]))
                ax.set_yticks(range(mat.shape[0]))
                for i in range(mat.shape[0]):
                    for j in range(mat.shape[1]):
                        ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=axes, shrink=0.75, label="Directed effect")
        fig.savefig(out_dir / "core_graph_heatmaps.png")
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
