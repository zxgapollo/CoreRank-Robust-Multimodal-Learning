from __future__ import annotations

import numpy as np

from corerank_synth.metrics import dag_metrics, directed_graph_metrics


def test_dag_metrics_distinguish_chain_from_cycle() -> None:
    chain = np.zeros((3, 3), dtype=float)
    chain[1, 0] = 0.4
    chain[2, 1] = -0.3

    cycle = chain.copy()
    cycle[0, 2] = 0.2

    chain_metrics = dag_metrics(chain)
    cycle_metrics = dag_metrics(cycle)

    assert abs(chain_metrics["dag_acyclicity"]) < 1e-8
    assert chain_metrics["dag_threshold_is_acyclic"] == 1.0
    assert cycle_metrics["dag_acyclicity"] > 0.0
    assert cycle_metrics["dag_threshold_is_acyclic"] == 0.0


def test_directed_graph_metrics_report_skeleton_and_reversals() -> None:
    true = np.zeros((3, 3), dtype=float)
    true[1, 0] = 0.4
    true[2, 1] = -0.3

    perfect = true.copy()
    perfect_metrics = directed_graph_metrics(perfect, true, threshold=0.05)
    assert perfect_metrics["graph_f1"] == 1.0
    assert perfect_metrics["graph_skeleton_f1"] == 1.0
    assert perfect_metrics["graph_directed_hamming"] == 0.0

    reversed_one = np.zeros((3, 3), dtype=float)
    reversed_one[0, 1] = 0.4
    reversed_one[2, 1] = -0.3
    reversed_metrics = directed_graph_metrics(reversed_one, true, threshold=0.05)

    assert reversed_metrics["graph_reversed_edges"] == 1.0
    assert reversed_metrics["graph_skeleton_f1"] == 1.0
    assert reversed_metrics["graph_f1"] < 1.0
