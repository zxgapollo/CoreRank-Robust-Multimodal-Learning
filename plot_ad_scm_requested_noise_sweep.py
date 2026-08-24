from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


LABELS = {
    "demo_only": "D only",
    "late_fusion_no_demo": "Late Fusion w/o D",
    "concat_no_demo": "Concat w/o D",
    "no_demo": "Concat w/o D",
    "concat": "Concat w/D",
    "late_fusion": "Late Fusion w/D",
    "mlcsl": "MLCSL w/D",
    "mlcsl_no_demo": "MLCSL w/o D",
}

ORDER = (
    "demo_only",
    "late_fusion_no_demo",
    "concat_no_demo",
    "no_demo",
    "concat",
    "late_fusion",
    "mlcsl",
    "mlcsl_no_demo",
)

STYLE = {
    "demo_only": dict(color="#7f7f7f", marker="s", linestyle=(0, (1.5, 3.5))),
    "late_fusion_no_demo": dict(color="#2ca02c", marker="P", linestyle=(0, (4, 2))),
    "concat_no_demo": dict(color="#111111", marker="o", linestyle="-"),
    "no_demo": dict(color="#111111", marker="o", linestyle="-"),
    "concat": dict(color="#d62728", marker="D", linestyle="-"),
    "late_fusion": dict(color="#ff7f0e", marker="v", linestyle=(0, (5, 2))),
    "mlcsl": dict(color="#1f77b4", marker="o", linestyle="-"),
    "mlcsl_no_demo": dict(color="#6a3d9a", marker="^", linestyle=(0, (7, 2))),
}


def _series(df: pd.DataFrame, method: str, split: str, metric: str) -> pd.Series:
    sub = df[df["method"].eq(method) & df["split"].eq(split)]
    return sub.groupby("sweep_value")[metric].mean().sort_index()


def _available_methods(df: pd.DataFrame) -> list[str]:
    seen = set(df["method"])
    methods = [m for m in ORDER if m in seen]
    methods.extend(sorted(seen - set(methods)))
    return methods


def _plot_panel(ax, df: pd.DataFrame, methods: list[str], split: str, metric: str) -> None:
    for method in methods:
        vals = _series(df, method, split, metric)
        if vals.empty:
            continue
        style = STYLE.get(method, dict(marker="o", linestyle="-"))
        ax.plot(
            vals.index,
            vals.values,
            linewidth=2.0,
            markersize=5.8,
            markeredgewidth=0.45,
            markeredgecolor="white",
            label=LABELS.get(method, method),
            **style,
        )


def _plot_delta(ax, df: pd.DataFrame, methods: list[str], split: str = "test_ood_noise") -> None:
    ref_method = "concat_no_demo" if "concat_no_demo" in set(df["method"]) else "no_demo"
    ref = _series(df, ref_method, split, "acc")
    for method in methods:
        vals = _series(df, method, split, "acc")
        if vals.empty:
            continue
        aligned = vals.subtract(ref, fill_value=float("nan")).dropna()
        if aligned.empty:
            continue
        style = STYLE.get(method, dict(marker="o", linestyle="-"))
        ax.plot(
            aligned.index,
            aligned.values,
            linewidth=2.0,
            markersize=5.8,
            markeredgewidth=0.45,
            markeredgecolor="white",
            label=LABELS.get(method, method),
            **style,
        )
    ax.axhline(0.0, color="#333333", linewidth=0.8, linestyle=(0, (2, 2)))


def make_figure(metrics_csv: Path, output: Path) -> None:
    df = pd.read_csv(metrics_csv)
    df = df[df["mode"].eq("noise")].copy()
    methods = _available_methods(df)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.5))
    axes = axes.reshape(2, 2)

    panels = [
        (axes[0, 0], "a", "Clean-test accuracy", "test_ood_noise", "acc", "Accuracy"),
        (axes[0, 1], "b", "Clean-test macro AUC", "test_ood_noise", "macro_auc_ovr", "Macro AUC OVR"),
        (axes[1, 0], "c", "ID accuracy", "test_id", "acc", "Accuracy"),
    ]
    for ax, letter, title, split, metric, ylabel in panels:
        _plot_panel(ax, df, methods, split, metric)
        ax.set_title(title, loc="left", pad=10, fontweight="bold")
        ax.text(-0.17, 1.16, letter, transform=ax.transAxes, fontsize=10.5, fontweight="bold")
        ax.set_ylabel(ylabel)

    ax = axes[1, 1]
    _plot_delta(ax, df, methods)
    ax.set_title("Clean-test accuracy change vs Concat w/o D", loc="left", pad=10, fontweight="bold")
    ax.text(-0.17, 1.16, "d", transform=ax.transAxes, fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Delta accuracy")

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#dddddd", linewidth=0.45)
        ax.set_xlabel("")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[1, 0].set_xlabel("Train MRI noise scale")
    axes[1, 1].set_xlabel("Train MRI noise scale")

    handles: list[Line2D] = []
    labels: list[str] = []
    for method in methods:
        label = LABELS.get(method, method)
        if label in labels:
            continue
        style = STYLE.get(method, dict(color="#333333", marker="o", linestyle="-"))
        handles.append(
            Line2D(
                [0],
                [0],
                linewidth=2.0,
                markersize=6.0,
                markeredgewidth=0.45,
                markeredgecolor="white",
                label=label,
                **style,
            )
        )
        labels.append(label)

    fig.suptitle("AD-SCM MRI noise sweep", fontsize=13.5, fontweight="bold", y=0.98)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False,
        handlelength=2.4,
        columnspacing=2.6,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.18, wspace=0.35, hspace=0.70)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics_csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    make_figure(args.metrics_csv, args.output)


if __name__ == "__main__":
    main()
