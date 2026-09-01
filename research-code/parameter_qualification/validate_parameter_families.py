#!/usr/bin/env python3
"""Fail-closed validation of the declared heterogeneous parameter families."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import csv

ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT / "PHYSICAL_PARAMETER_REGISTRY.json"
OUT = ROOT / "PARAMETER_FAMILY_VALIDATION.json"


def within(x: np.ndarray, bounds: list[float]) -> bool:
    return bool(np.all((x >= bounds[0]) & (x <= bounds[1])))


def sample_uav(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    # Payload severity drives mass and available reserve in opposite directions;
    # aerodynamic and attitude variations have smaller independent components.
    payload = rng.uniform(0.0, 1.0, n)
    mass = 0.75 + 0.60 * payload
    reserve = np.clip(1.0 - 0.24 * payload + rng.normal(0, 0.015, n), 0.72, 1.0)
    drag = np.clip(0.625 + 0.75 * payload + rng.uniform(0, 0.375, n), 0.625, 1.75)
    bandwidth = np.clip(1.263 - 0.55 * payload + rng.normal(0, 0.035, n), 0.579, 1.263)
    return {"mass_ratio": mass, "drag_ratio": drag,
            "bandwidth_ratio": bandwidth, "actuator_reserve_ratio": reserve}


def sample_vehicle(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    # A shared driveline severity variable correlates longer lag with reduced
    # authority, avoiding physically arbitrary Cartesian products.
    severity = rng.uniform(0.0, 1.0, n)
    lag = 0.7 + 1.1 * severity
    authority = np.clip(1.05 - 0.40 * severity + rng.normal(0, 0.012, n), 0.65, 1.05)
    drag = np.clip(0.6 + 0.85 * severity + rng.uniform(0, 0.35, n), 0.6, 1.8)
    return {"lag_ratio": lag, "authority_ratio": authority, "drag_ratio": drag}


def sample_motor(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    # Size/load and winding severities generate correlated complete motors.
    size = rng.uniform(0.0, 1.0, n)
    load = rng.uniform(0.0, 1.0, n)
    resistance = np.clip(1.45 - 0.65 * size + rng.normal(0, 0.025, n), 0.7, 1.5)
    inductance = np.clip(0.65 + 0.75 * size + rng.normal(0, 0.035, n), 0.6, 1.6)
    inertia = np.clip(0.625 + 0.65 * size + 0.475 * load, 0.625, 1.75)
    friction = np.clip(0.60 + 0.58 * size + 0.653 * load, 0.6, 1.833)
    torque = np.clip(0.75 + 0.45 * size + rng.normal(0, 0.018, n), 0.75, 1.2)
    return {"resistance_ratio": resistance, "inductance_ratio": inductance,
            "inertia_ratio": inertia, "friction_ratio": friction,
            "torque_constant_ratio": torque}


def main() -> None:
    registry = json.loads(REGISTRY.read_text())
    failures: list[str] = []
    summaries: dict[str, dict] = {}
    samplers = {"uav": sample_uav, "vehicle": sample_vehicle, "motor": sample_motor}
    for domain, sampler in samplers.items():
        draws = {k: [] for k in registry["domains"][domain]["ranges"]}
        for seed in range(31001, 31031):
            values = sampler(np.random.default_rng(seed), 50)
            for key, x in values.items():
                if not within(x, registry["domains"][domain]["ranges"][key]):
                    failures.append(f"{domain}:{key}:range")
                if np.ptp(x) <= 0:
                    failures.append(f"{domain}:{key}:no_heterogeneity")
                draws[key].extend(x.tolist())
        if domain == "uav":
            a = 2.8 * np.asarray(draws["actuator_reserve_ratio"]) / np.asarray(draws["mass_ratio"])
            if np.min(a) < 1.49 or np.max(a) > 3.74:
                failures.append("uav:available_acceleration")
        elif domain == "vehicle":
            braking = 4.2 * np.asarray(draws["authority_ratio"])
            if np.min(braking) < 2.73 or np.max(braking) > 4.41:
                failures.append("vehicle:braking_authority")
        else:
            b = registry["domains"][domain]["baseline"]
            te = b["inductance_h"] * np.asarray(draws["inductance_ratio"]) / (b["resistance_ohm"] * np.asarray(draws["resistance_ratio"]))
            tm = b["equivalent_inertia_kg_m2"] * np.asarray(draws["inertia_ratio"]) / (b["viscous_friction_n_m_s"] * np.asarray(draws["friction_ratio"]))
            if not within(te, registry["domains"][domain]["constraints"]["electrical_time_constant_s"]):
                failures.append("motor:electrical_time_constant")
            if not within(tm, registry["domains"][domain]["constraints"]["mechanical_time_constant_s"]):
                failures.append("motor:mechanical_time_constant")
        summaries[domain] = {k: {"n": len(v), "min": min(v), "median": float(np.median(v)), "max": max(v)} for k, v in draws.items()}
    v27_path = ROOT.parent / "cross_domain_v27" / "results" / "v27_node_parameters.csv"
    if v27_path.exists():
        v27 = list(csv.DictReader(v27_path.open()))
        for domain in ("robot", "microgrid", "circuit", "water", "structure"):
            summaries[domain] = {}
            for key, bounds in registry["domains"][domain]["ranges"].items():
                x = np.asarray([float(r["value"]) for r in v27 if r["domain"] == domain and r["parameter"] == key])
                if x.size == 0 or not within(x, bounds): failures.append(f"{domain}:{key}:range")
                if x.size == 0 or np.ptp(x) <= 0: failures.append(f"{domain}:{key}:no_heterogeneity")
                if x.size: summaries[domain][key] = {"n": int(x.size), "min": float(x.min()), "median": float(np.median(x)), "max": float(x.max())}
    payload = {
        "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
        "status": "PASS" if not failures else "FAIL",
        "claim_boundary": registry["claim_boundary"],
        "independent_node_level_identification": False,
        "draws_per_domain": 1500,
        "failures": sorted(set(failures)),
        "summaries": summaries
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
