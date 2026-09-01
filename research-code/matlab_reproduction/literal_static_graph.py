#!/usr/bin/env python3
"""Literal port of the static graph-1/graph-2 core recurrences.

The legacy scripts solve LMIs before simulation.  This first-stage port imports
the archived LMI-derived F and H matrices, but independently executes the
plant, distributed model estimator, state reconstructor, and error equations.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.io import loadmat


VARIABLES = [
    *(f"A{i}{i}" for i in range(1, 9)),
    *(f"B{i}{i}" for i in range(1, 9)),
    *(f"H{i}{i}" for i in range(1, 8)),
    "F", "Coup", "L", "L2", "h", "Len",
]


def phi(x: float) -> float:
    return 0.5 * (abs(x + 1.0) - abs(x - 1.0))


def legacy_rk4(x: np.ndarray, A: np.ndarray, B: np.ndarray, h: float,
               injection: np.ndarray) -> np.ndarray:
    """Exact legacy stage algebra; injection occurs in stage 1 only."""
    nd1 = np.array([phi(x[0]), 0.0, 0.0])
    k1 = h * (A @ x + B @ nd1 + injection)
    nd2 = np.array([phi(x[0]) + 0.5 * k1[0], 0.0, 0.0])
    k2 = h * (A @ (x + 0.5 * k1) + B @ nd2)
    nd3 = np.array([phi(x[0]) + 0.5 * k2[0], 0.0, 0.0])
    k3 = h * (A @ (x + 0.5 * k2) + B @ nd3)
    nd4 = np.array([phi(x[0]) + k3[0], 0.0, 0.0])
    k4 = h * (A @ (x + k3) + B @ nd4)
    return x + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def disagreement_matrices(M: np.ndarray, graph: np.ndarray,
                          leader: np.ndarray) -> np.ndarray:
    # graph has seven follower columns and one leader column.
    return np.einsum("ij,jab->iab", graph[:, :7], M) + graph[:, 7, None, None] * leader


def disagreement_states(xg: np.ndarray, graph: np.ndarray,
                        leader: np.ndarray) -> np.ndarray:
    return graph[:, :7] @ xg + graph[:, 7, None] * leader


def run(mat_path: Path, graph_name: str, steps: int | None = None,
        posthoc_graph2_ab3_ma4: bool = False) -> dict[str, np.ndarray]:
    raw = loadmat(mat_path, variable_names=VARIABLES)
    A = np.stack([np.asarray(raw[f"A{i}{i}"], dtype=np.float64) for i in range(1, 9)])
    B = np.stack([np.asarray(raw[f"B{i}{i}"], dtype=np.float64) for i in range(1, 9)])
    H = np.stack([np.asarray(raw[f"H{i}{i}"], dtype=np.float64) for i in range(1, 8)])
    F = np.asarray(raw["F"], dtype=np.float64)
    coup = float(raw["Coup"].item())
    h = float(raw["h"].item())
    total = int(raw["Len"].item())
    n_steps = total if steps is None else min(int(steps), total)
    graph = np.asarray(raw["L" if graph_name == "graph1" else "L2"], dtype=np.float64)[:7]

    # Source initial conditions.  The fourth state deliberately receives x3.
    xx = np.zeros((8, 3, n_steps), dtype=np.float64)
    xx[:, 2, 0] = 0.1
    xx[3, :, 0] = xx[2, :, 0]  # literal: xx4(:,1)=x3
    xg = np.zeros((7, 3, n_steps), dtype=np.float64)

    MA = np.stack([(i + 1.0) * np.eye(3) for i in range(7)])
    MB = MA.copy()
    AA = disagreement_matrices(MA, graph, A[7])
    AB = disagreement_matrices(MB, graph, B[7])
    z = disagreement_states(xg[:, :, 0], graph, xx[7, :, 0])
    if graph_name == "graph2":
        # Literal source defect: z7(:,1)=2*xg7(:,1)-x8(:,1)-x6(:,1).
        z[6] = 2.0 * xg[6, :, 0] - np.array([0.0, 0.0, 0.1]) - np.array([0.0, 0.0, 0.1])

    # Store model matrices because the archived workspace makes them testable.
    MA_hist = np.empty((7, 3, 3, n_steps), dtype=np.float64)
    MB_hist = np.empty_like(MA_hist)
    MA_hist[:, :, :, 0] = MA
    MB_hist[:, :, :, 0] = MB

    for n in range(1, n_steps):
        MA = MA - h * coup * AA
        MB = MB - h * coup * AB
        MA_hist[:, :, :, n] = MA
        MB_hist[:, :, :, n] = MB

        old_xg = xg[:, :, n - 1]
        old_xx = xx[:, :, n - 1]
        for i in range(7):
            xg[i, :, n] = legacy_rk4(old_xg[i], MA[i], MB[i], h, -coup * (F @ z[i]))
            coupling = graph[i, :7] @ old_xx[:7] + graph[i, 7] * old_xx[7]
            xx[i, :, n] = legacy_rk4(old_xx[i], A[i], B[i], h, -coup * (H[i] @ coupling))
        xx[7, :, n] = legacy_rk4(old_xx[7], A[7], B[7], h, np.zeros(3))

        AA = disagreement_matrices(MA, graph, A[7])
        AB = disagreement_matrices(MB, graph, B[7])
        if graph_name == "graph2" and posthoc_graph2_ab3_ma4:
            # Proven source transcription missed by frozen v1: after the first
            # step MATLAB uses MA4 (not MB4) in AB3.  Initial AB3 still uses MB4.
            AB[2] = 2.0 * MB[2] - MB[1] - MA[3]
        z = disagreement_states(xg[:, :, n], graph, xx[7, :, n])

    errors = np.linalg.norm(xx[:7] - xx[7:8], axis=1)
    aggregate = np.sqrt(np.mean(errors * errors, axis=0))
    return {"xx": xx, "xg": xg, "MA": MA_hist, "MB": MB_hist,
            "Error_agents": errors, "Error": aggregate}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mat", type=Path, required=True)
    p.add_argument("--graph", choices=("graph1", "graph2"), required=True)
    p.add_argument("--steps", type=int)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = run(args.mat, args.graph, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    meta = {
        "graph": args.graph, "mat": str(args.mat), "steps": int(result["xx"].shape[-1]),
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "scipy": scipy.__version__, "implementation": "literal-static-core-v1",
    }
    args.output.with_suffix(".environment.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
