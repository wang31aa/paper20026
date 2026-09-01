#!/usr/bin/env python3
"""Independent conic-solver check of the frozen robust-metric preflight."""
from __future__ import annotations

import json
from pathlib import Path
import argparse

import cvxpy as cp
import numpy as np
import scipy
import clarabel
import scs

import run_preflight as rp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def solve(l1: np.ndarray, corners: list[np.ndarray], solver: str) -> dict[str, object]:
    n = l1.shape[0]
    g = cp.Variable(n)
    margin = cp.Variable()
    G = cp.diag(g)
    constraints = [cp.sum(g) == n, g >= rp.PROTOCOL["metric_problem"]["minimum_g"]]
    for rho in corners:
        d = np.diag(rho)
        constraints.append(G @ d @ l1 + l1.T @ d @ G - margin * np.eye(n) >> 0)
    problem = cp.Problem(cp.Maximize(margin), constraints)
    if solver == "CLARABEL":
        problem.solve(solver=solver, tol_gap_abs=1e-10, tol_feas=1e-10,
                      tol_gap_rel=1e-10, max_iter=1000)
    else:
        problem.solve(solver=solver, eps=1e-7, max_iters=200000)
    return {"status": problem.status, "margin": float(margin.value),
            "g": np.asarray(g.value, dtype=float).tolist()}


def main(out: Path, allow_overwrite_frozen: bool = False) -> None:
    vertices = rp.PROTOCOL["frozen_parameter_vertices"]
    b0 = rp.PROTOCOL["nominal_pwm_gain"]
    lo = min(v["b"] for v in vertices) / b0
    hi = max(v["b"] for v in vertices) / b0
    corners = rp.corners(lo, hi, rp.PROTOCOL["followers"])
    graphs = {}
    for name, w in rp.topologies().items():
        l1 = rp.follower_laplacian(w)
        graphs[name] = {solver: solve(l1, corners, solver)
                        for solver in ("CLARABEL", "SCS")}
    payload = {
        "protocol_sha256": rp.sha256(rp.PROTOCOL_PATH),
        "cvxpy_version": cp.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "clarabel_version": clarabel.__version__,
        "scs_version": scs.__version__,
        "solvers": cp.installed_solvers(),
        "graphs": graphs,
        "decision_rule": "positive common margin required for every graph",
        "all_graphs_positive": all(item["CLARABEL"]["margin"] > 0
                                   and item["SCS"]["margin"] > 0
                                   for item in graphs.values())
    }
    if out.resolve() == (HERE / "results").resolve() and not allow_overwrite_frozen:
        raise SystemExit("refusing to overwrite frozen results; choose a temporary --out directory")
    if not out.is_dir():
        raise SystemExit("run_preflight.py must create --out before the independent check")
    (out / "independent_sdp.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    sums_path = out / "SHA256SUMS.json"
    sums = {
        "PREFLIGHT_PROTOCOL.json": rp.sha256(HERE / "PREFLIGHT_PROTOCOL.json"),
        "run_preflight.py": rp.sha256(HERE / "run_preflight.py"),
        "independent_sdp_check.py": rp.sha256(HERE / "independent_sdp_check.py"),
        "validate.py": rp.sha256(HERE / "validate.py"),
        "test_preflight.py": rp.sha256(HERE / "test_preflight.py"),
        "reproduce_temp.py": rp.sha256(HERE / "reproduce_temp.py"),
        "README.md": rp.sha256(HERE / "README.md"),
        "RESULTS.md": rp.sha256(HERE / "RESULTS.md"),
        "../requirements-linux-py313.lock": rp.sha256(ROOT / "requirements-linux-py313.lock"),
        "../openmct_greybox_round42/PROTOCOL.json": rp.sha256(ROOT / "openmct_greybox_round42" / "PROTOCOL.json"),
        "../openmct_greybox_round42/results/qualification.json": rp.sha256(ROOT / "openmct_greybox_round42" / "results" / "qualification.json"),
        "preflight.json": rp.sha256(out / "preflight.json"),
        "independent_sdp.json": rp.sha256(out / "independent_sdp.json")
    }
    sums_path.write_text(json.dumps(sums, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {s: x["margin"] for s, x in item.items()}
                      for name, item in graphs.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-overwrite-frozen", action="store_true")
    args = parser.parse_args()
    main(args.out, args.allow_overwrite_frozen)
