from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def make_figures(metrics_csv: str | Path, output_dir: str | Path) -> list[str]:
    import matplotlib.pyplot as plt

    metrics_csv = Path(metrics_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metrics_csv)
    paths: list[str] = []

    order = ["test_id", "test_ood_a", "test_ood_b", "test_ood_c", "test_ood_d"]
    auc_df = df[df["split"].isin(order)].pivot(index="method", columns="split", values="auc")
    auc_df = auc_df[[c for c in order if c in auc_df.columns]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    auc_df.plot(kind="bar", ax=ax)
    ax.set_ylabel("AUC")
    ax.set_xlabel("")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("BC-MCSGN fixed SCM: ID/OOD AUC")
    ax.legend(title="split", ncols=2, fontsize=8)
    fig.tight_layout()
    path = output_dir / "auc_id_ood.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    state_df = df[df["split"].eq("test_id")].set_index("method")
    if "state_r2" in state_df.columns:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        state_df["state_r2"].plot(kind="bar", ax=ax, color="#4477aa")
        ax.set_ylabel("R2(rep, true S)")
        ax.set_xlabel("")
        ax.set_title("State recovery on ID test")
        fig.tight_layout()
        path = output_dir / "state_recovery_id.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))

    if "shortcut_sensitivity" in df.columns:
        sens = df[df["split"].isin(order)].pivot(index="method", columns="split", values="shortcut_sensitivity")
        sens = sens[[c for c in order if c in sens.columns]]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        sens.plot(kind="bar", ax=ax)
        ax.set_ylabel("Residual prediction variance after conditioning on true S")
        ax.set_xlabel("")
        ax.set_title("Shortcut sensitivity proxy")
        ax.legend(title="split", ncols=2, fontsize=8)
        fig.tight_layout()
        path = output_dir / "shortcut_sensitivity.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))

    return paths
