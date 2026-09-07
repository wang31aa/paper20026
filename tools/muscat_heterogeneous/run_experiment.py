#!/usr/bin/env python3
"""Frozen MuSCAT-parameter-constrained heterogeneous attitude computation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sat(x: np.ndarray, limit: np.ndarray) -> np.ndarray:
    return np.clip(x, -limit, limit)


def simulate(c: dict, rho: float, seed: int, policy: str) -> dict:
    rng = np.random.default_rng(seed)
    n = len(c["inertia_kg_m2"])
    j = np.asarray(c["inertia_kg_m2"], float)
    tau_max = np.asarray(c["torque_limit_Nm"], float)
    bias = np.asarray(c["neighbour_attitude_bias_rad"], float)
    a = np.asarray(c["physical_adjacency"], float)
    l = np.diag(a.sum(axis=1)) - a
    pin = np.asarray(c["target_visibility"], float)
    dt, steps = c["dt_s"], int(c["horizon_s"] / c["dt_s"])
    theta = rng.normal(0.0, c["initial_attitude_sd_rad"], n)
    omega = rng.normal(0.0, c["initial_rate_sd_rad_s"], n)
    z = rng.normal(c["observer_initial_mean_rad"], c["observer_initial_sd_rad"], n)
    max_err = min_clear = energy = comm = saturation = 0.0
    tail = []
    failed = False
    for k in range(steps):
        t = k * dt
        target = c["target_amplitude_rad"] * np.sin(c["target_frequency_rad_s"] * t)
        z += dt * c["observer_gain"] * rho * (-l @ z + pin * (target - z))
        measured = theta + bias
        coupling = a @ measured - a.sum(axis=1) * theta
        weights = np.ones(n)
        if policy == "two_layer":
            residual = np.abs(measured - z)
            weights = np.where(residual > c["gate_threshold_rad"], c["gate_floor"], 1.0)
            weights[pin > 0] = 1.0
            coupling = a @ (weights * measured) - (a @ weights) * theta
        disturbance = np.asarray(c["persistent_torque_Nm"], float)
        disturbance += rng.normal(0.0, c["torque_noise_sd_Nm"], n)
        requested = (-np.asarray(c["kp_Nm_rad"], float) * (theta - z)
                     - np.asarray(c["kd_Nm_s_rad"], float) * omega
                     + rho * c["physical_coupling_gain_Nm_rad"] * coupling)
        applied = sat(requested, tau_max)
        omega += dt * (applied + disturbance) / j
        theta += dt * omega
        err = np.abs(theta - target)
        max_err = max(max_err, float(err.max()))
        min_clear = max(min_clear, float(err.max()))
        energy += float(np.sum(applied * applied)) * dt
        comm += float(np.count_nonzero(a) * rho) * dt
        saturation += float(np.mean(np.abs(requested) > tau_max)) * dt
        if k >= int(c["tail_fraction"] * steps):
            tail.append(float(err.max()))
    tail_error = float(np.mean(tail))
    failed = tail_error > c["task_tube_rad"] or saturation > c["max_saturation_dwell_s"]
    return {"rho": rho, "seed": seed, "policy": policy, "success": int(not failed),
            "tail_error_rad": tail_error, "peak_error_rad": max_err,
            "control_effort_Nm2_s": energy, "communication_weight_s": comm,
            "saturation_dwell_s": saturation}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--contract", required=True)
    p.add_argument("--split", choices=["development", "heldout"], required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    contract_path = Path(args.contract)
    raw = contract_path.read_bytes()
    c = json.loads(raw)
    expected = c.get("contract_sha256")
    if expected is None:
        if args.split != "development" or not c["contract_version"].endswith("_DEVELOPMENT"):
            raise SystemExit("FAIL: unsigned contracts are restricted to development")
    else:
        unsigned = dict(c); unsigned.pop("contract_sha256")
        canonical = (json.dumps(unsigned, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if hashlib.sha256(canonical).hexdigest() != expected:
            raise SystemExit("FAIL: frozen contract hash mismatch")
    seeds = c[f"{args.split}_seeds"]
    rows = [simulate(c, rho, seed, policy) for rho in c["rho_grid"]
            for seed in seeds for policy in c["policies"]]
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(json.dumps({"split": args.split, "rows": len(rows), "successes": sum(r["success"] for r in rows)}))


if __name__ == "__main__":
    main()
