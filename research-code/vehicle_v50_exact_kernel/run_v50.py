#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "V50_FROZEN_CONTRACT.json").read_text())
OUT = HERE / "results"
CLEARANCE = SPEC["clearance_m"]
B = SPEC["follower_max_braking_mps2"]
D = SPEC["leader_max_braking_mps2"]
NET = B - D
DT = SPEC["sample_time_s"]


def boundary(closing: np.ndarray | float) -> np.ndarray | float:
    return CLEARANCE + np.maximum(closing, 0.0) ** 2 / (2.0 * NET)


def grid_certificate(delta: float) -> dict[str, float | int]:
    g0, g1 = SPEC["gap_range_m"]
    c0, c1 = SPEC["closing_speed_range_mps"]
    gs = np.arange(g0 + delta / 2, g1, delta)
    cs = np.arange(c0 + delta / 2, c1, delta)
    inner = outer = 0
    maximum_cell_uncertainty = 0.0
    for g in gs:
        for c in cs:
            corners_g = np.array([g - delta / 2, g + delta / 2])
            corners_c = np.array([c - delta / 2, c + delta / 2])
            margins = np.array([gg - boundary(cc) for gg in corners_g for cc in corners_c])
            inner += int(np.min(margins) >= 0.0)
            outer += int(np.max(margins) >= 0.0)
            maximum_cell_uncertainty = max(maximum_cell_uncertainty, float(np.max(margins)-np.min(margins)))
    return {"delta": delta, "inner_cells": inner, "outer_cells": outer,
            "ambiguous_cells": outer-inner,
            "ambiguous_area_m_mps": float((outer-inner)*delta*delta),
            "geometric_error_bound": float(np.sqrt(2.0)*delta),
            "maximum_margin_uncertainty": maximum_cell_uncertainty}


def in_kernel(g: float, c: float) -> bool:
    return g + 1e-12 >= boundary(c)


def robust_action(g: float, c: float) -> float | None:
    # Choose the least braking whose one-step worst-case successor remains in
    # the exact continuous-time stopping kernel.  Falling back to B realizes
    # the exact viability policy when the finite action grid is too coarse.
    for b in SPEC["candidate_braking_mps2"]:
        cn = c + DT * (D - b)
        gn = g - DT * c - 0.5 * DT * DT * (D - b)
        if in_kernel(gn, cn):
            return float(b)
    return float(B) if in_kernel(g, c) else None


def simulate(seed: int, policy: str) -> dict[str, float | int | str]:
    rng = np.random.default_rng(seed)
    c = float(rng.uniform(0.2, 6.5))
    g = float(boundary(c) + rng.uniform(0.05, 4.0))
    safe = True
    energy = 0.0
    for _ in range(SPEC["horizon_steps"]):
        d = float(np.clip(rng.normal(1.5, 0.9), 0.0, D))
        if policy == "kernel_supervisor":
            b = robust_action(g, c)
            b = B if b is None else b
        elif policy == "maximum_braking":
            b = B
        elif policy == "half_braking":
            b = B / 2
        elif policy == "no_intervention":
            b = 0.0
        else:
            raise ValueError(policy)
        g = g - DT * c - 0.5 * DT * DT * (d - b)
        c = c + DT * (d - b)
        energy += b*b*DT
        safe &= g >= CLEARANCE - 1e-12
    return {"seed": seed, "policy": policy, "task_success": int(safe), "energy": energy,
            "final_gap_m": g, "final_closing_speed_mps": c}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    grids = [grid_certificate(float(d)) for d in SPEC["grid_resolutions"]]
    policies = ("kernel_supervisor", "maximum_braking", "half_braking", "no_intervention")
    dev = [simulate(s, p) for s in SPEC["development_seeds"] for p in policies]
    frozen = {"contract_sha256": hashlib.sha256((HERE/"V50_FROZEN_CONTRACT.json").read_bytes()).hexdigest(),
              "policy": "kernel_supervisor", "selection_rule": "least candidate braking preserving the exact stopping kernel"}
    frozen_path = OUT / "V50_FROZEN_POLICY.json"
    frozen_path.write_text(json.dumps(frozen, indent=2)+"\n")
    held = [simulate(s, p) for s in SPEC["heldout_seeds"] for p in policies]
    summary = {}
    for p in policies:
        rows = [r for r in held if r["policy"] == p]
        summary[p] = {"success_rate": float(np.mean([r["task_success"] for r in rows])),
                      "median_energy": float(np.median([r["energy"] for r in rows]))}
    # The per-seed oracle minimizes energy among successful declared policies.
    regret = []
    for s in SPEC["heldout_seeds"]:
        rows = [r for r in held if r["seed"] == s]
        feasible = [r for r in rows if r["task_success"]]
        sup = next(r for r in rows if r["policy"] == "kernel_supervisor")
        if feasible and sup["task_success"]:
            regret.append(float(sup["energy"] - min(r["energy"] for r in feasible)))
    report = {"contract_sha256": frozen["contract_sha256"],
              "prediction_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
              "exact_kernel": "gap >= clearance + max(closing_speed,0)^2/(2*(B-D))",
              "grid_certificates": grids, "development_runs": len(dev), "heldout_runs": len(held),
              "heldout": summary, "oracle_comparable_cases": len(regret),
              "median_energy_regret": float(np.median(regret)) if regret else None,
              "maximum_energy_regret": float(np.max(regret)) if regret else None,
              "claim_boundary": SPEC["claim_boundary"]}
    (OUT/"V50_REPORT.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
