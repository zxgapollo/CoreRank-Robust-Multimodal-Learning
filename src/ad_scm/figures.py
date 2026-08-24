from __future__ import annotations

from pathlib import Path

import pandas as pd


METHOD_LABELS = {
    "demo_only": "D only",
    "late_fusion_no_demo": "Late Fusion w/o D",
    "concat_no_demo": "Concat w/o D",
    "no_demo": "Concat w/o D",
    "concat": "Concat w/D",
    "late_fusion": "Late Fusion w/D",
    "mlcsl": "MLCSL w/D",
    "mlcsl_no_demo": "MLCSL w/o D",
}

METHOD_ORDER = (
    "demo_only",
    "late_fusion_no_demo",
    "concat_no_demo",
    "no_demo",
    "concat",
    "late_fusion",
    "mlcsl",
    "mlcsl_no_demo",
)


def make_sweep_figures(metrics_csv: str | Path, output_dir: str | Path) -> list[str]:
    import matplotlib.pyplot as plt

    df = pd.read_csv(metrics_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for metric, ylabel in [("acc", "Accuracy"), ("macro_auc_ovr", "Macro AUC OVR"), ("nll", "NLL")]:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ood = df[df["split"].str.startswith("test_ood")]
        methods = [method for method in METHOD_ORDER if method in set(ood["method"])]
        methods.extend(sorted(set(ood["method"]) - set(methods)))
        for method in methods:
            sub = ood[ood["method"].eq(method)]
            grouped = sub.groupby("sweep_value")[metric].mean().sort_index()
            ax.plot(grouped.index, grouped.values, marker="o", label=METHOD_LABELS.get(method, method))
        ax.set_xlabel(df["sweep_name"].iloc[0])
        ax.set_ylabel(ylabel)
        ax.set_title(f"AD-SCM OOD {ylabel}")
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"ood_{metric}_curve.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))
    return paths
