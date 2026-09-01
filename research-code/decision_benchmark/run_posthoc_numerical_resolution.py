#!/usr/bin/env python3
"""Frozen post-hoc numerical-resolution audit for Phase B.

This does not alter or supersede the pre-registered Phase B dt gate. See
POSTHOC_NUMERICAL_PROTOCOL.md.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import run_decision_benchmark as phase_b

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "phase_b_raw" / "decision_runs.csv"
OUT = ROOT / "posthoc_numerical_raw"
DTS = (phase_b.DT, phase_b.DT / 2, phase_b.DT / 4, phase_b.DT / 8)
FINE_DT = DTS[-1]


def archived_labels() -> dict[tuple[int, str], bool]:
    rows = list(csv.DictReader(ARCHIVE.open()))
    labels: dict[tuple[int, str], bool] = {}
    for row in rows:
        key = (int(row["seed"]), row["scenario"])
        value = bool(int(row["protection_required"]))
        if key in labels and labels[key] != value:
            raise RuntimeError(f"inconsistent archived label: {key}")
        labels[key] = value
    expected = {(s, k) for k in phase_b.SCENARIOS for s in phase_b.PHASE_B_SEEDS}
    if set(labels) != expected:
        raise RuntimeError("archived Phase B cases are incomplete")
    return labels


def frozen_noise(seed: int, kind: str) -> np.ndarray:
    """Phase-B-seeded iid path on the finest lattice, shared by nested grids."""
    rng = np.random.default_rng(44021 + seed + 1000 * phase_b.SCENARIOS.index(kind))
    n = int(round(phase_b.T_END / FINE_DT)) + 1
    return rng.normal(0.0, phase_b.NOISE_SD, (n, phase_b.N))


def simulate_coupled(seed: int, kind: str, strategy: str, dt: float,
                     required: bool, noise_fine: np.ndarray) -> dict:
    par = phase_b.case_parameters(seed, kind, "B")
    L = phase_b.laplacian(par["bscale"])
    Phi, Gamma = phase_b.discrete_plant(dt, L)
    stride = int(round(dt / FINE_DT))
    if not np.isclose(stride * FINE_DT, dt):
        raise RuntimeError("grids are not nested")
    steps = int(round(phase_b.T_END / dt))
    x = np.zeros(3 * phase_b.N)
    active = False
    scheduled = np.inf
    action_time = np.nan
    alarm_time = np.nan
    max_abs = 0.0
    unsafe_time = 0.0
    max_est = 0.0
    z = np.zeros(phase_b.N)
    z_prev = np.zeros(phase_b.N)
    alpha, beta = 13.0, 8.0

    for k in range(steps + 1):
        t = k * dt
        p = phase_b.disturbance(t, kind, par["amp"], par["node"])
        f = x[phase_b.N:2 * phase_b.N]
        y = f + noise_fine[k * stride]
        z_prev[:] = z
        z += dt * (-alpha * (z - y) - beta * (L @ z))
        est = float(np.max(np.abs(z)))
        trend = float(max(0.0, -(np.mean(z) - np.mean(z_prev)) / dt))
        risk = est + 0.35 * trend
        max_est = max(max_est, risk)

        if not active and not np.isfinite(scheduled):
            trigger = False
            if strategy == "oracle":
                trigger = bool(required and np.any(p))
            elif strategy == "full":
                trigger = risk + phase_b.CERT_MARGIN >= phase_b.SAFETY
            elif strategy == "uncertified":
                trigger = risk >= phase_b.SAFETY
            elif strategy == "no_model":
                trigger = float(np.max(np.abs(y))) >= phase_b.ALARM
            if trigger:
                alarm_time = t
                scheduled = t + phase_b.ACTION_DELAY
        if not active and t + 1e-12 >= scheduled:
            active = True
            action_time = t

        x = Phi @ x + Gamma @ (p * (1 - phase_b.ACTION_FRACTION * float(active)))
        peak = float(np.max(np.abs(x[phase_b.N:2 * phase_b.N])))
        max_abs = max(max_abs, peak)
        unsafe_time += dt * (peak > phase_b.SAFETY)

    return {
        "seed": seed, "scenario": kind, "strategy": strategy, "dt": dt,
        "amplitude": par["amp"], "disturbance_node": par["node"],
        "bscale": par["bscale"], "protection_required": int(required),
        "action": int(active),
        "alarm_time": None if np.isnan(alarm_time) else alarm_time,
        "action_time": None if np.isnan(action_time) else action_time,
        "max_abs_frequency": max_abs, "unsafe_time": unsafe_time,
        "max_risk_estimate": max_est,
    }


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values), q, method="linear"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = archived_labels()
    rows: list[dict] = []
    for kind in phase_b.SCENARIOS:
        for seed in phase_b.PHASE_B_SEEDS:
            noise = frozen_noise(seed, kind)
            required = labels[(seed, kind)]
            for dt in DTS:
                for strategy in phase_b.STRATEGIES:
                    rows.append(simulate_coupled(seed, kind, strategy, dt, required, noise))

    with (OUT / "runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    indexed = {(r["seed"], r["scenario"], r["strategy"], r["dt"]): r for r in rows}
    comparisons: list[dict] = []
    level_summaries: list[dict] = []
    for coarse, fine in zip(DTS[:-1], DTS[1:]):
        level_rows = []
        for kind in phase_b.SCENARIOS:
            for seed in phase_b.PHASE_B_SEEDS:
                for strategy in phase_b.STRATEGIES:
                    a = indexed[(seed, kind, strategy, coarse)]
                    b = indexed[(seed, kind, strategy, fine)]
                    both = bool(a["action"] and b["action"])
                    action_time_diff = (abs(float(a["action_time"]) - float(b["action_time"]))
                                        if both else None)
                    item = {
                        "seed": seed, "scenario": kind, "strategy": strategy,
                        "coarse_dt": coarse, "fine_dt": fine,
                        "coarse_peak": a["max_abs_frequency"],
                        "fine_peak": b["max_abs_frequency"],
                        "peak_abs_difference": abs(a["max_abs_frequency"] - b["max_abs_frequency"]),
                        "coarse_action": a["action"], "fine_action": b["action"],
                        "action_agreement": int(a["action"] == b["action"]),
                        "both_act": int(both), "action_time_abs_difference": action_time_diff,
                    }
                    comparisons.append(item)
                    level_rows.append(item)
        diffs = [x["peak_abs_difference"] for x in level_rows]
        timing = [x["action_time_abs_difference"] for x in level_rows
                  if x["action_time_abs_difference"] is not None]
        level_summaries.append({
            "coarse_dt": coarse, "fine_dt": fine, "n": len(level_rows),
            "max_peak_abs_difference": max(diffs),
            "p99_peak_abs_difference": quantile(diffs, 0.99),
            "median_peak_abs_difference": quantile(diffs, 0.5),
            "action_agreement_rate": float(np.mean([x["action_agreement"] for x in level_rows])),
            "action_disagreements": sum(1 - x["action_agreement"] for x in level_rows),
            "both_act_n": len(timing),
            "max_action_time_abs_difference": max(timing) if timing else None,
            "p99_action_time_abs_difference": quantile(timing, 0.99) if timing else None,
        })

    with (OUT / "adjacent_comparisons.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    prev, finest = level_summaries[-2], level_summaries[-1]
    gates = {
        "finest_max_peak_le_0.005": finest["max_peak_abs_difference"] <= 0.005,
        "finest_p99_peak_le_0.001": finest["p99_peak_abs_difference"] <= 0.001,
        "finest_action_agreement_ge_0.99": finest["action_agreement_rate"] >= 0.99,
        "max_peak_nonincreasing_at_finest": finest["max_peak_abs_difference"] <= prev["max_peak_abs_difference"],
        "p99_peak_nonincreasing_at_finest": finest["p99_peak_abs_difference"] <= prev["p99_peak_abs_difference"],
    }
    summary = {
        "status": "post-hoc numerical-resolution audit; Phase B gate remains FAIL",
        "phase_b_gate_preserved": "FAIL",
        "noise_coupling": "one finest-grid iid path per scenario; coincident samples on nested grids",
        "dt_levels": list(DTS), "n_runs": len(rows), "n_comparisons": len(comparisons),
        "adjacent_levels": level_summaries, "frozen_gates": gates,
        "numerical_resolution_closed": all(gates.values()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
