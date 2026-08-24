from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MODEL_LABELS = {
    "unimodal": "Unimodal",
    "concat": "Concat",
    "late_fusion": "Late fusion",
    "iso_poe": "ISO-PoE",
    "oracle_state": "Oracle state",
}

MODEL_COLORS = {
    "unimodal": "#7a7a7a",
    "concat": "#c76b4a",
    "late_fusion": "#8d6ab8",
    "iso_poe": "#2f6fbb",
    "oracle_state": "#4f8a57",
}


def _mean_sem(df: pd.DataFrame, group_cols: list[str], value: str) -> pd.DataFrame:
    out = df.groupby(group_cols, dropna=False)[value].agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"].fillna(0.0) / np.sqrt(out["count"].clip(lower=1))
    return out


def make_core_figures(results_csv: str | Path, out_dir: str | Path) -> None:
    import matplotlib.pyplot as plt

    results_csv = Path(results_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_csv, dtype={"subset": str})
    test = df[df["split"] == "test"].copy()

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    plot_df = test[np.isfinite(test["lambda_y"]) & np.isfinite(test["auc"])].copy()
    for model, sdf in plot_df.groupby("model"):
        ax.scatter(
            sdf["lambda_y"],
            sdf["auc"],
            s=36 if model == "iso_poe" else 28,
            alpha=0.72,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model),
        )
    ax.set_xlabel("Label-relevant observability lambda_Y(M)")
    ax.set_ylabel("OOD test AUC")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_observability_vs_auc.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    plot_df = test[np.isfinite(test["ambiguity_proxy"]) & np.isfinite(test["nll"])].copy()
    for model, sdf in plot_df.groupby("model"):
        ax.scatter(
            sdf["ambiguity_proxy"],
            sdf["nll"],
            s=36 if model == "iso_poe" else 28,
            alpha=0.72,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model),
        )
    ax.set_xlabel("State ambiguity proxy Uhat_Y(M)")
    ax.set_ylabel("OOD test NLL")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_ambiguity_vs_nll.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    sample = test[test["scenario"].isin(["complementary", "noisy_modality"])].copy()
    full = sample[(sample["subset_size"] == sample.groupby(["scenario", "seed", "n_train"])["subset_size"].transform("max")) | (sample["model"] == "oracle_state")]
    best_uni = (
        sample[sample["model"] == "unimodal"]
        .groupby(["scenario", "seed", "n_train"], as_index=False)["auc"]
        .max()
        .assign(model="best_unimodal")
    )
    full_models = full[full["model"].isin(["concat", "late_fusion", "iso_poe", "oracle_state"])][["scenario", "seed", "n_train", "model", "auc"]]
    curve_df = pd.concat([best_uni, full_models], ignore_index=True)
    label_map = {**MODEL_LABELS, "best_unimodal": "Best unimodal"}
    color_map = {**MODEL_COLORS, "best_unimodal": "#555555"}
    if not curve_df.empty:
        agg = _mean_sem(curve_df, ["n_train", "model"], "auc")
        order = ["best_unimodal", "concat", "late_fusion", "iso_poe", "oracle_state"]
        for model in order:
            sdf = agg[agg["model"] == model].sort_values("n_train")
            if sdf.empty:
                continue
            ax.errorbar(
                sdf["n_train"],
                sdf["mean"],
                yerr=sdf["sem"],
                marker="o",
                capsize=3,
                label=label_map.get(model, model),
                color=color_map.get(model),
            )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Training samples")
    ax.set_ylabel("OOD test AUC")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_sample_complexity.png")
    plt.close(fig)

    shortcut = df[df["scenario"] == "shortcut"].copy()
    if not shortcut.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        full_size = int(shortcut["subset_size"].max())
        bars = shortcut[(shortcut["subset_size"].isin([full_size, 0])) & shortcut["model"].isin(["concat", "late_fusion", "iso_poe", "oracle_state"])]
        agg = _mean_sem(bars, ["split", "model"], "auc")
        models = [m for m in ["concat", "late_fusion", "iso_poe", "oracle_state"] if m in set(agg["model"])]
        x = np.arange(len(models))
        width = 0.34
        for offset, split_name, color in [(-width / 2, "id_test", "#7aa6d9"), (width / 2, "test", "#d9795c")]:
            vals = agg[agg["split"] == split_name].set_index("model").reindex(models)
            ax.bar(x + offset, vals["mean"], width, yerr=vals["sem"], capsize=3, label="ID" if split_name == "id_test" else "OOD", color=color)
        ax.set_xticks(x, [MODEL_LABELS.get(m, m) for m in models], rotation=12)
        ax.set_ylabel("AUC")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.22)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_shortcut_ood.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    state_df = test[test["model"].isin(["unimodal", "concat", "late_fusion", "iso_poe", "oracle_state"])].copy()
    full_size = int(state_df["subset_size"].max()) if not state_df.empty else 0
    state_df = state_df[(state_df["subset_size"].isin([full_size, 0])) | (state_df["model"] == "unimodal")]
    best_uni_state = (
        state_df[state_df["model"] == "unimodal"]
        .groupby(["scenario", "seed", "n_train"], as_index=False)["state_r2"]
        .max()
        .assign(model="best_unimodal")
    )
    full_state = state_df[state_df["model"].isin(["concat", "late_fusion", "iso_poe", "oracle_state"])][
        ["scenario", "seed", "n_train", "model", "state_r2"]
    ]
    state_plot = pd.concat([best_uni_state, full_state], ignore_index=True)
    if not state_plot.empty:
        agg = _mean_sem(state_plot, ["model"], "state_r2")
        order = [m for m in ["best_unimodal", "concat", "late_fusion", "iso_poe", "oracle_state"] if m in set(agg["model"])]
        vals = agg.set_index("model").reindex(order)
        ax.bar(
            np.arange(len(order)),
            vals["mean"],
            yerr=vals["sem"],
            capsize=3,
            color=[color_map.get(m, "#777777") for m in order],
        )
        ax.set_xticks(np.arange(len(order)), [label_map.get(m, m) for m in order], rotation=12)
    ax.set_ylabel("State recovery R2")
    ax.set_ylim(0.0, 1.02)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_state_recovery.png")
    plt.close(fig)
