#!/usr/bin/env python3
"""Stage-consistent RK4 for the intended coupled continuous equations."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.io import loadmat

from literal_static_graph import VARIABLES, disagreement_matrices, disagreement_states


def phi_vec(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    out[..., 0] = 0.5 * (np.abs(x[..., 0] + 1.0) - np.abs(x[..., 0] - 1.0))
    return out


def derivative(state, A, B, H, F, coup, graph):
    xx, xg, MA, MB = state
    dMA = -coup * disagreement_matrices(MA, graph, A[7])
    dMB = -coup * disagreement_matrices(MB, graph, B[7])
    plant_dis = graph[:, :7] @ xx[:7] + graph[:, 7, None] * xx[7]
    z = disagreement_states(xg, graph, xx[7])
    dxx = np.empty_like(xx)
    dxx[:7] = np.einsum("iab,ib->ia", A[:7], xx[:7]) + np.einsum("iab,ib->ia", B[:7], phi_vec(xx[:7])) - coup * np.einsum("iab,ib->ia", H, plant_dis)
    dxx[7] = A[7] @ xx[7] + B[7] @ phi_vec(xx[7])
    dxg = np.einsum("iab,ib->ia", MA, xg) + np.einsum("iab,ib->ia", MB, phi_vec(xg)) - coup * (z @ F.T)
    return dxx, dxg, dMA, dMB


def add(state, deriv, factor):
    return tuple(x + factor * dx for x, dx in zip(state, deriv))


def rk4_step(state, h, args):
    k1 = derivative(state, *args)
    k2 = derivative(add(state, k1, 0.5 * h), *args)
    k3 = derivative(add(state, k2, 0.5 * h), *args)
    k4 = derivative(add(state, k3, h), *args)
    return tuple(x + h * (a + 2*b + 2*c + d) / 6.0 for x, a, b, c, d in zip(state, k1, k2, k3, k4))


def run(mat_path: Path, graph_name: str, h: float, horizon: float = 100.0, sample_dt: float = 0.001):
    raw = loadmat(mat_path, variable_names=VARIABLES)
    A = np.stack([np.asarray(raw[f"A{i}{i}"], float) for i in range(1, 9)])
    B = np.stack([np.asarray(raw[f"B{i}{i}"], float) for i in range(1, 9)])
    H = np.stack([np.asarray(raw[f"H{i}{i}"], float) for i in range(1, 8)])
    F = np.asarray(raw["F"], float)
    coup = float(raw["Coup"].item())
    graph = np.asarray(raw["L" if graph_name == "graph1" else "L2"], float)[:7]
    ratio = sample_dt / h
    stride = int(round(ratio))
    if not np.isclose(stride * h, sample_dt): raise ValueError("sample_dt must be integer multiple of h")
    n = int(round(horizon / h))
    ns = int(round(horizon / sample_dt)) + 1
    xx = np.zeros((8, 3)); xx[:, 2] = 0.1
    xg = np.zeros((7, 3))
    MA = np.stack([(i + 1.0) * np.eye(3) for i in range(7)])
    MB = MA.copy()
    state = (xx, xg, MA, MB)
    xx_out = np.empty((8, 3, ns)); xg_out = np.empty((7, 3, ns))
    xx_out[..., 0] = xx; xg_out[..., 0] = xg
    j = 1
    args = (A, B, H, F, coup, graph)
    for step in range(1, n + 1):
        state = rk4_step(state, h, args)
        if step % stride == 0:
            xx_out[..., j] = state[0]; xg_out[..., j] = state[1]; j += 1
    err = np.linalg.norm(xx_out[:7] - xx_out[7:8], axis=1)
    agg = np.sqrt(np.mean(err * err, axis=0))
    rec = np.linalg.norm(xg_out - xx_out[7:8], axis=1)
    recagg = np.sqrt(np.mean(rec * rec, axis=0))
    return {"xx": xx_out, "xg": xg_out, "Error_agents": err, "Error": agg,
            "ReconstructionError_agents": rec, "ReconstructionError": recagg}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", type=Path, required=True)
    p.add_argument("--graph", choices=("graph1", "graph2"), required=True)
    p.add_argument("--h", type=float, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = run(args.mat, args.graph, args.h)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    meta = {"graph": args.graph, "h": args.h, "horizon": 100.0, "sample_dt": 0.001,
            "implementation": "intended-continuous-stage-consistent-rk4-v1",
            "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__}
    args.output.with_suffix(".environment.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__": main()

