#!/usr/bin/env python3
"""Fail-closed validator for the Round46 metric preflight."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import run_preflight as rp

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


p = json.loads((HERE / "PREFLIGHT_PROTOCOL.json").read_text(encoding="utf-8"))
r = json.loads((HERE / "results" / "preflight.json").read_text(encoding="utf-8"))
sdp = json.loads((HERE / "results" / "independent_sdp.json").read_text(encoding="utf-8"))
h = json.loads((HERE / "results" / "SHA256SUMS.json").read_text(encoding="utf-8"))
errors: list[str] = []
for rel, expected in h.items():
    path = (HERE / rel).resolve() if rel not in {"preflight.json", "independent_sdp.json"} else HERE / "results" / rel
    if not path.is_file() or sha256(path) != expected:
        errors.append(f"hash mismatch: {rel}")
if r["protocol_sha256"] != sha256(HERE / "PREFLIGHT_PROTOCOL.json"):
    errors.append("protocol binding mismatch")
if r["rho_corner_count"] != 2 ** p["followers"]:
    errors.append("incomplete rho corners")
expected_corners = {tuple(x) for x in itertools.product(r["rho_interval"], repeat=p["followers"])}
ordered_corners = rp.corners(*r["rho_interval"], p["followers"])
tol = p["metric_problem"]["pass_tolerance"]
for name in p["graphs"]:
    item = r["graphs"].get(name)
    if item is None:
        errors.append(f"missing graph: {name}")
        continue
    if len(item["generalized_margins"]) != r["rho_corner_count"]:
        errors.append(f"incomplete margins: {name}")
    g = np.asarray(item["g"], dtype=float)
    if not np.isclose(g.sum(), p["followers"], rtol=0, atol=1e-9):
        errors.append(f"normalization mismatch: {name}")
    if g.min() < p["metric_problem"]["minimum_g"] - 1e-10:
        errors.append(f"minimum g mismatch: {name}")
    l1 = rp.follower_laplacian(rp.topologies()[name])
    invsqrt = np.diag(g ** -0.5)
    euclidean = []
    generalized = []
    for rho in ordered_corners:
        d = np.diag(rho)
        mat = np.diag(g) @ d @ l1 + l1.T @ d @ np.diag(g)
        euclidean.append(float(np.linalg.eigvalsh(mat).min()))
        generalized.append(float(np.linalg.eigvalsh(invsqrt @ mat @ invsqrt).min()))
    if not np.allclose(generalized, item["generalized_margins"], rtol=0, atol=1e-12):
        errors.append(f"corner margins mismatch: {name}")
    if not np.isclose(min(euclidean), item["minimum_recomputed_euclidean_margin"], rtol=0, atol=1e-12):
        errors.append(f"Euclidean margin mismatch: {name}")
    if tuple(item["worst_corner_rho"]) not in expected_corners:
        errors.append(f"invalid worst corner: {name}")
    recomputed = min(item["generalized_margins"])
    if not np.isclose(recomputed, item["minimum_generalized_margin"], rtol=0, atol=1e-12):
        errors.append(f"minimum mismatch: {name}")
    expected_pass = bool(item["solver_status"] == "optimal_slsqp"
                         and item["minimum_g"] >= p["metric_problem"]["minimum_g"] - 1e-10
                         and item["minimum_recomputed_euclidean_margin"] > tol
                         and item["minimum_generalized_margin"] > tol)
    if r["graph_checks"][name] != expected_pass:
        errors.append(f"graph decision mismatch: {name}")
    for solver in ("CLARABEL", "SCS"):
        independent = sdp["graphs"][name][solver]["margin"]
        if not np.isclose(independent, item["euclidean_margin"], rtol=0, atol=6e-8):
            errors.append(f"independent solver mismatch: {name}/{solver}")
    if sdp["graphs"][name]["CLARABEL"]["status"] not in {"optimal", "optimal_inaccurate"}:
        errors.append(f"Clarabel status: {name}")
    if sdp["graphs"][name]["SCS"]["status"] != "optimal":
        errors.append(f"SCS status: {name}")
supported = bool(all(r["source_checks"].values()) and all(r["graph_checks"].values()))
if r["exploratory_metric_preflight_supported"] != supported:
    errors.append("global decision mismatch")
if r["exploratory_sensitivity_simulation_authorized"] or r["qualified_openmct_network_model"] or r["physical_evidence"]:
    errors.append("prohibited evidence promotion")
if sdp["protocol_sha256"] != r["protocol_sha256"] or sdp["all_graphs_positive"]:
    errors.append("independent SDP decision mismatch")
for key in ("cvxpy_version", "numpy_version", "scipy_version", "clarabel_version", "scs_version"):
    if not sdp.get(key):
        errors.append(f"missing environment version: {key}")
if errors:
    raise SystemExit("FAIL Round46 preflight\n" + "\n".join(errors))
print(f"PASS: {len(p['graphs'])} graphs x {r['rho_corner_count']} corners; "
      f"metric_supported={supported}; simulation_authorized=false; physical=false")
