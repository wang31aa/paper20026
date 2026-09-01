#!/usr/bin/env python3
"""Frozen public-simulation-calibrated closed-loop formation test."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from robot_formation import load, inject, FAULTS

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
GAIN = 0.08


def edge_set(weight: float) -> list[tuple[int, int, float]]:
    # (receiver, sender, weight): the trusted directed cycle is always retained.
    edges = [(1, 0, 1.0), (2, 1, 1.0), (3, 2, 1.0), (0, 3, 1.0)]
    edges += [(4, 0, weight), (3, 4, weight)]
    return edges


def formation_error(pos: np.ndarray, reference: np.ndarray,
                    nodes: tuple[int, ...]) -> float:
    values = []
    for q, i in enumerate(nodes):
        for j in nodes[q + 1:]:
            target = reference[j] - reference[i]
            values.append(np.linalg.norm((pos[j] - pos[i]) - target))
    return float(max(values, default=0.0))


def clip_rows(values: np.ndarray, limit: float) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1)
    factors = np.minimum(1.0, limit / np.maximum(norms, 1e-12))
    return values * factors[:, None]


def simulate(initial: np.ndarray, actions: np.ndarray, policy: str,
             thresholds: tuple[float, float], correction_limit: float,
             tol_full: float, tol_core: float) -> dict[str, float | int | str]:
    pos = initial.copy()
    high, low = thresholds
    excluded = False
    over = under = 0
    reentered = False
    full, core, effort, weights = [], [], [], []
    first_full = first_core = len(actions) * 2.0
    for k, action in enumerate(actions):
        score = float(np.linalg.norm(action[4] - np.median(action, axis=0)))
        if policy in ("binary_gate", "connectivity_gate"):
            over = over + 1 if score > high else 0
            under = under + 1 if score < low else 0
            if over >= 2:
                excluded = True
            if under >= 2 and excluded:
                excluded = False
                reentered = True
            weight = 0.0 if excluded else 1.0
        elif policy == "continuous_weight":
            weight = float(np.clip((high - score) / max(high - low, 1e-12), 0.0, 1.0))
        else:
            weight = 1.0

        correction = np.zeros_like(pos)
        for receiver, sender, edge_weight in edge_set(weight):
            target = initial[sender] - initial[receiver]
            correction[receiver] += (GAIN * edge_weight
                                     * ((pos[sender] - pos[receiver]) - target))
        correction = clip_rows(correction, correction_limit)
        pos += action + correction
        ef = formation_error(pos, initial, (0, 1, 2, 3, 4))
        ec = formation_error(pos, initial, (0, 1, 2, 3))
        full.append(ef); core.append(ec)
        effort.append(float(np.sum(correction ** 2)))
        weights.append(weight)
        if ef > tol_full and first_full == len(actions) * 2.0:
            first_full = k * 2.0
        if ec > tol_core and first_core == len(actions) * 2.0:
            first_core = k * 2.0
    f, c = np.asarray(full), np.asarray(core)
    return {
        "policy": policy,
        "first_full_loss_s": first_full,
        "first_core_loss_s": first_core,
        "full_within_fraction": float(np.mean(f <= tol_full)),
        "core_within_fraction": float(np.mean(c <= tol_core)),
        "full_tail_error": float(np.quantile(f, .95)),
        "core_tail_error": float(np.quantile(c, .95)),
        "control_effort": float(np.sum(effort)),
        "mean_candidate_weight": float(np.mean(weights)),
        "readmitted": int(reentered),
    }


def calibration_values(runs: dict[int, tuple[np.ndarray, np.ndarray]]):
    full, core, scores, action_norms = [], [], [], []
    for run in range(30):
        pos, act = runs[run]
        for p in pos:
            full.append(formation_error(p, pos[0], (0, 1, 2, 3, 4)))
            core.append(formation_error(p, pos[0], (0, 1, 2, 3)))
        scores.extend(np.linalg.norm(act[:, 4] - np.median(act, axis=1), axis=1))
        action_norms.extend(np.linalg.norm(act.reshape(-1, 2), axis=1))
    return (float(np.quantile(full, .95)), float(np.quantile(core, .95)),
            (float(np.quantile(scores, .95)), float(np.quantile(scores, .60))),
            float(np.quantile(action_norms, .99)))


def main() -> None:
    runs = load()
    tol_full, tol_core, thresholds, correction_limit = calibration_values(runs)
    policies = ("all_coupled", "binary_gate", "continuous_weight", "connectivity_gate")
    rows = []
    for run in range(30, 50):
        pos, act = runs[run]
        for fault in FAULTS:
            forced = inject(act, fault, thresholds[0], 3000 + 10 * run + FAULTS.index(fault))
            for policy in policies:
                rows.append({"run": run, "fault": fault,
                             **simulate(pos[0], forced, policy, thresholds,
                                        correction_limit, tol_full, tol_core)})
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "robot_closed_loop_tasks.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)
    figure_rows = []
    for row in rows:
        figure_rows.append({
            "condition_index": FAULTS.index(str(row["fault"])),
            "policy_index": policies.index(str(row["policy"])),
            "policy": row["policy"],
            "fault": row["fault"],
            "full_within_fraction": row["full_within_fraction"],
            "full_tail_error": row["full_tail_error"],
        })
    with (RESULTS / "robot_closed_loop_figure.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=figure_rows[0])
        writer.writeheader(); writer.writerows(figure_rows)
    summary = {
        "protocol": "ROBOT_CLOSED_LOOP_PROTOCOL.md",
        "calibration_runs": 30,
        "heldout_runs": 20,
        "paired_conditions": 100,
        "evaluations": len(rows),
        "full_tolerance_source_units": tol_full,
        "core_tolerance_source_units": tol_core,
        "gate_high_source_units": thresholds[0],
        "gate_low_source_units": thresholds[1],
        "correction_limit_source_units_per_step": correction_limit,
        "claim_boundary": "closed-loop public-simulation-calibrated formation task; no collision or hardware claim",
    }
    (RESULTS / "robot_closed_loop_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
