#!/usr/bin/env python3
"""Frozen robot formation-integrity stress test in source coordinates."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SOURCE = ROOT / "public_cluster_cases/cache/robot_swarm_validation.csv"
FAULTS = ("none", "formation_change", "swinging", "noisy", "brute_force")


def load() -> dict[int, tuple[np.ndarray, np.ndarray]]:
    with SOURCE.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    out = {}
    for run in sorted({int(r["run"]) for r in rows}):
        selected = [r for r in rows if int(r["run"]) == run]
        nt = 1 + max(int(r["step"]) for r in selected)
        pos = np.zeros((nt, 5, 2)); act = np.zeros_like(pos)
        for row in selected:
            t, i = int(row["step"]), int(row["robot_idx"])
            pos[t, i] = (float(row["pos_x"]), float(row["pos_y"]))
            act[t, i] = (float(row["action_x"]), float(row["action_y"]))
        out[run] = pos, act
    return out


def pair_values(pos: np.ndarray, active: np.ndarray) -> np.ndarray:
    values = []
    for i in range(5):
        for j in range(i + 1, 5):
            if active[i] and active[j]:
                values.append(np.linalg.norm(pos[i] - pos[j]))
    return np.asarray(values)


def inject(actions: np.ndarray, fault: str, scale: float, seed: int) -> np.ndarray:
    out = actions.copy(); start, stop = len(out) // 3, 2 * len(out) // 3
    if fault == "none": return out
    rng = np.random.default_rng(seed)
    for k in range(start, stop):
        if fault == "formation_change": out[k, 4] *= 1.8
        elif fault == "swinging": out[k, 4] += scale * np.array([np.sin(k), np.cos(k)])
        elif fault == "noisy": out[k, 4] += rng.normal(0, scale, 2)
        elif fault == "brute_force":
            direction = out[k, 4] / max(np.linalg.norm(out[k, 4]), 1e-12)
            out[k, 4] += 3.0 * scale * direction
    return out


def simulate(initial: np.ndarray, actions: np.ndarray, policy: str,
             expected: np.ndarray, tolerance: float,
             thresholds: tuple[float, float]) -> dict[str, float | int | str]:
    pos = initial.copy(); active = np.ones(5, dtype=bool)
    over = under = 0; errors = []; first = len(actions) * 2.0; readmitted = False
    minimum_active = 5
    for k, action in enumerate(actions):
        score = np.linalg.norm(action - np.median(action, axis=0), axis=1)
        if policy == "gated":
            over = over + 1 if score[4] > thresholds[0] else 0
            under = under + 1 if score[4] < thresholds[1] else 0
            if over >= 2: active[4] = False
            if under >= 2 and not active[4]: active[4] = True; readmitted = True
            active[0] = True
        minimum_active = min(minimum_active, int(active.sum()))
        pos += action
        current = []
        q = 0
        for i in range(5):
            for j in range(i + 1, 5):
                if active[i] and active[j]: current.append(abs(np.linalg.norm(pos[i]-pos[j])-expected[q]))
                q += 1
        error = max(current) if current else float("inf")
        errors.append(error)
        if error > tolerance and first == len(actions) * 2.0: first = k * 2.0
    array = np.asarray(errors)
    return {"policy": policy, "first_loss_s": first,
            "within_tolerance_fraction": float(np.mean(array <= tolerance)),
            "tail_error": float(np.quantile(array, .95)),
            "minimum_active": minimum_active, "readmitted": int(readmitted)}


def main() -> None:
    runs = load(); calibration = [r for r in runs if r <= 29]; test = [r for r in runs if r >= 30]
    pair_rows = []
    action_scores = []
    for run in calibration:
        pos, act = runs[run]
        for p in pos: pair_rows.append(pair_values(p, np.ones(5, bool)))
        action_scores.extend(np.linalg.norm(act - np.median(act, axis=1, keepdims=True), axis=2).ravel())
    pair_matrix = np.vstack(pair_rows); expected = np.median(pair_matrix, axis=0)
    deviations = np.max(np.abs(pair_matrix - expected), axis=1)
    tolerance = float(np.quantile(deviations, .95))
    thresholds = (float(np.quantile(action_scores, .95)), float(np.quantile(action_scores, .60)))
    scale = thresholds[0]
    rows = []
    for run in test:
        pos, act = runs[run]
        for fault in FAULTS:
            forced = inject(act, fault, scale, 1000 + 10 * run + FAULTS.index(fault))
            for policy in ("all_coupled", "gated"):
                result = simulate(pos[0], forced, policy, expected, tolerance, thresholds)
                rows.append({"run": run, "fault": fault, **result})
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "robot_formation_tasks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    summary = {"calibration_runs": calibration, "heldout_runs": test,
               "task_tolerance_source_units": tolerance,
               "gate_high_source_units": thresholds[0], "gate_low_source_units": thresholds[1],
               "conditions": len(rows),
               "claim_boundary": "public-simulation-calibrated formation integrity; not physical collision safety"}
    (RESULTS / "robot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
