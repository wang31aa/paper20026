#!/usr/bin/env python3
"""Solve the frozen Round46 common-diagonal-metric preflight."""
from __future__ import annotations

import hashlib
import itertools
import json
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROTOCOL_PATH = HERE / "PREFLIGHT_PROTOCOL.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def topologies() -> dict[str, np.ndarray]:
    n = PROTOCOL["followers"]
    out: dict[str, np.ndarray] = {}
    w = np.zeros((n, n + 1)); w[0, n] = 1; w[1, 0] = 1; w[2, 1] = 1; w[3, 2] = 1; w[4, 3] = 1
    out["chain"] = w
    w = np.zeros((n, n + 1)); w[0, n] = 1; w[1, 0] = 1; w[2, 0] = 1; w[3, 1] = 1; w[3, 2] = .5; w[4, 2] = 1
    out["branch"] = w
    w = np.zeros((n, n + 1)); w[0, n] = 1; w[1, 0] = 1; w[2, 1] = 1; w[3, 2] = 1; w[4, 3] = 1; w[0, 4] = .25; w[2, 0] = .4
    out["cyclic"] = w
    return out


def follower_laplacian(w: np.ndarray) -> np.ndarray:
    n = PROTOCOL["followers"]
    lap = np.zeros((n + 1, n + 1))
    lap[:n, :] = -w
    lap[np.arange(n), np.arange(n)] += w.sum(axis=1)
    return lap[:n, :n]


def corners(lo: float, hi: float, n: int) -> list[np.ndarray]:
    return [np.asarray(bits, dtype=float) for bits in itertools.product((lo, hi), repeat=n)]


def solve_graph(l1: np.ndarray, rho_corners: list[np.ndarray]) -> dict[str, object]:
    n = l1.shape[0]
    gmin = PROTOCOL["metric_problem"]["minimum_g"]

    def matrix(g: np.ndarray, rho: np.ndarray) -> np.ndarray:
        d = np.diag(rho)
        return np.diag(g) @ d @ l1 + l1.T @ d @ np.diag(g)

    def inequalities(x: np.ndarray) -> np.ndarray:
        g, margin = x[:n], x[-1]
        return np.r_[g - gmin,
                     [np.linalg.eigvalsh(matrix(g, rho)).min() - margin
                      for rho in rho_corners]]

    result = minimize(lambda x: -x[-1], np.r_[np.ones(n), 0.01],
                      method="SLSQP", bounds=[(gmin, n)] * n + [(-100.0, 100.0)],
                      constraints=[{"type": "eq", "fun": lambda x: x[:n].sum() - n},
                                   {"type": "ineq", "fun": inequalities}],
                      options={"ftol": 1e-12, "maxiter": 5000, "disp": False})
    if not result.success:
        return {"solver_status": f"failed:{result.message}", "g": None, "euclidean_margin": None,
                "generalized_margins": [], "minimum_generalized_margin": None}
    gv = np.asarray(result.x[:n], dtype=float)
    invsqrt = np.diag(gv ** -0.5)
    generalized = []
    euclidean = []
    for rho in rho_corners:
        s = matrix(gv, rho)
        euclidean.append(float(np.linalg.eigvalsh(s).min()))
        generalized.append(float(np.linalg.eigvalsh(invsqrt @ s @ invsqrt).min()))
    return {
        "solver_status": "optimal_slsqp",
        "solver_message": result.message,
        "solver_iterations": int(result.nit),
        "g": gv.tolist(),
        "minimum_g": float(gv.min()),
        "euclidean_margin": float(result.x[-1]),
        "minimum_recomputed_euclidean_margin": min(euclidean),
        "generalized_margins": generalized,
        "minimum_generalized_margin": min(generalized),
        "worst_corner_index": int(np.argmin(generalized)),
        "worst_corner_rho": rho_corners[int(np.argmin(generalized))].tolist(),
    }


def main(out: Path, allow_overwrite_frozen: bool = False) -> None:
    prior_protocol = ROOT / "openmct_greybox_round42" / "PROTOCOL.json"
    prior_result = ROOT / "openmct_greybox_round42" / "results" / "qualification.json"
    q = json.loads(prior_result.read_text(encoding="utf-8"))
    vertices = PROTOCOL["frozen_parameter_vertices"]
    b0 = PROTOCOL["nominal_pwm_gain"]
    rho_lo = min(v["b"] for v in vertices) / b0
    rho_hi = max(v["b"] for v in vertices) / b0
    rho_corners = corners(rho_lo, rho_hi, PROTOCOL["followers"])
    source_checks = {
        "prior_protocol_hash": sha256(prior_protocol) == PROTOCOL["prior_result"]["protocol_sha256"],
        "prior_qualification_failed": q["qualification_passed"] is False,
        "prior_network_unauthorized": q["network_simulation_authorized"] is False,
        "prior_failed_gates_exact": q["stop_reason"] == PROTOCOL["prior_result"]["failed_gates"],
        "parameter_vertices_match_prior": (
            np.allclose([vertices[0][k] for k in ("a", "b", "c")],
                        [q["identifiability"]["theta"][k] for k in ("a", "b", "c")], rtol=0, atol=1e-12)
            and all(np.allclose([v[k] for k in ("a", "b", "c")], item["theta"], rtol=0, atol=1e-12)
                    for v, item in zip(vertices[1:], q["identifiability"]["leave_one_out"]))
        ),
        "corner_count": len(rho_corners) == PROTOCOL["rho_corner_count"],
    }
    graph_results = {name: solve_graph(follower_laplacian(w), rho_corners)
                     for name, w in topologies().items()}
    tol = PROTOCOL["metric_problem"]["pass_tolerance"]
    graph_checks = {
        name: bool(item["solver_status"] == "optimal_slsqp"
                   and item["minimum_g"] >= PROTOCOL["metric_problem"]["minimum_g"] - 1e-10
                   and item["minimum_recomputed_euclidean_margin"] > tol
                   and item["minimum_generalized_margin"] > tol)
        for name, item in graph_results.items()
    }
    supported = bool(all(source_checks.values()) and all(graph_checks.values()))
    result = {
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "classification": PROTOCOL["status"],
        "rho_interval": [rho_lo, rho_hi],
        "rho_corner_count": len(rho_corners),
        "source_checks": source_checks,
        "graph_checks": graph_checks,
        "graphs": graph_results,
        "exploratory_metric_preflight_supported": supported,
        "exploratory_sensitivity_simulation_authorized": False,
        "qualified_openmct_network_model": False,
        "physical_evidence": False,
        "claim_boundary": "necessary robust-metric feasibility only; no network trajectory, plant qualification, hardware execution or physical evidence"
    }
    if out.resolve() == (HERE / "results").resolve() and not allow_overwrite_frozen:
        raise SystemExit("refusing to overwrite frozen results; choose a temporary --out directory")
    out.mkdir(parents=True, exist_ok=allow_overwrite_frozen)
    (out / "preflight.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out / "SHA256SUMS.json").write_text(json.dumps({
        "PREFLIGHT_PROTOCOL.json": sha256(PROTOCOL_PATH),
        "run_preflight.py": sha256(HERE / "run_preflight.py"),
        "preflight.json": sha256(out / "preflight.json")
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"supported": supported, "graph_checks": graph_checks,
                      "rho_interval": [rho_lo, rho_hi]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-overwrite-frozen", action="store_true")
    args = parser.parse_args()
    main(args.out, args.allow_overwrite_frozen)
