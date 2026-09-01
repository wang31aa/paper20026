#!/usr/bin/env python3
"""Run the preregistered topology-aware extension on physical R1 traces."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

PATTERN = re.compile(r"ST_(\d+)_(\d+)\.dat$")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--edges", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results"))
    args = p.parse_args()

    edges = np.loadtxt(args.edges, dtype=int)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edge list must have two columns")
    edges = edges - 1
    if edges.min() < 0 or edges.max() >= 28 or np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("invalid node index or self-loop")
    undirected = np.sort(edges, axis=1)
    if len(np.unique(undirected, axis=0)) != len(undirected):
        raise ValueError("duplicate undirected edge")
    adjacency = np.zeros((28, 28), dtype=int)
    adjacency[undirected[:, 0], undirected[:, 1]] = 1
    adjacency += adjacency.T
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency

    rows = []
    for path in sorted(args.input.rglob("ST_*_*.dat")):
        match = PATTERN.search(path.name)
        if match is None:
            continue
        coupling, repeat = map(int, match.groups())
        values = np.loadtxt(path)
        if values.shape != (30000, 28) or not np.isfinite(values).all():
            raise ValueError(f"invalid trace: {path}")
        values = values[6000:]
        centred = values - values.mean(axis=0, keepdims=True)
        amplitude = np.sqrt(np.mean(centred**2))
        diffs = values[:, undirected[:, 0]] - values[:, undirected[:, 1]]
        edge_residual = np.sqrt(np.mean(diffs**2)) / amplitude
        energy = np.einsum("ti,ij,tj->t", values, laplacian, values)
        graph_residual = np.sqrt(np.mean(energy) / len(undirected)) / amplitude
        rows.append({"file": path.name, "coupling_index": coupling,
                     "repeat": repeat, "normalized_edge_residual": edge_residual,
                     "normalized_graph_energy_residual": graph_residual,
                     "identity_abs_difference": abs(edge_residual - graph_residual)})

    frame = pd.DataFrame(rows).sort_values(["coupling_index", "repeat"])
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output / "topology_metrics.csv", index=False)
    med = frame.groupby("coupling_index").normalized_edge_residual.median()
    rho, pvalue = spearmanr(med.index.to_numpy(), med.to_numpy())
    indexed = frame.set_index(["repeat", "coupling_index"])
    repeat_pass = {str(r): bool(indexed.loc[(r, 100), "normalized_edge_residual"] <
                                indexed.loc[(r, 0), "normalized_edge_residual"])
                   for r in (1, 2, 3)}
    result = {
        "structure_file": args.edges.name,
        "nodes": 28,
        "undirected_edges": int(len(undirected)),
        "symmetric": True,
        "degree_min": int(adjacency.sum(axis=1).min()),
        "degree_max": int(adjacency.sum(axis=1).max()),
        "connected_components": int(connected_components(adjacency, directed=False)[0]),
        "input_file_count": int(len(frame)),
        "all_303_finite": bool(len(frame) == 303 and np.isfinite(frame.select_dtypes("number")).all().all()),
        "max_identity_abs_difference": float(frame.identity_abs_difference.max()),
        "identity_tolerance_1e_10_pass": bool(frame.identity_abs_difference.max() <= 1e-10),
        "endpoint_median_x0": float(med.loc[0]),
        "endpoint_median_x100": float(med.loc[100]),
        "endpoint_direction_pass": bool(med.loc[100] < med.loc[0]),
        "spearman_rho": float(rho),
        "spearman_two_sided_p": float(pvalue),
        "endpoint_direction_by_repeat": repeat_pass,
        "all_repeat_endpoint_pass": bool(all(repeat_pass.values())),
        "claim_boundary": "graph-aware measurement residual only; no controller, observer, theorem or HIL validation"
    }
    (args.output / "topology_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
