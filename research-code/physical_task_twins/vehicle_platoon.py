#!/usr/bin/env python3
"""Frozen public-data-calibrated longitudinal vehicle task test."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SOURCE = ROOT / "public_cluster_cases/cache/adas_two_vehicle_sample.csv"

DT = 0.1
KP, KV = 0.25, 0.8
TIME_HEADWAY, STANDSTILL = 1.5, 5.0
AMIN, AMAX = -3.0, 2.0
BRAKE, RESPONSE, CLEARANCE = 3.0, 0.5, 2.0
STRESS = (0.0, 1.0, 2.0, 3.0)
DELAYS = (0.0, 0.5, 1.0)


def load() -> dict[str, np.ndarray]:
    with SOURCE.open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["id"] == "6"]
    keys = ("time", "distance_av_headway", "speed_av", "speed_sv1",
            "speed_sv2", "acc_av", "acc_sv1", "acc_sv2", "dim_x_av",
            "dim_x_sv1", "dim_x_sv2")
    return {key: np.asarray([float(r[key]) for r in rows]) for key in keys}


def safety_margin(gap: float, follower_speed: float, leader_speed: float,
                  follower_length: float, leader_length: float) -> float:
    bumper_gap = gap - 0.5 * (follower_length + leader_length)
    required = (CLEARANCE + RESPONSE * follower_speed
                + max((follower_speed ** 2 - leader_speed ** 2) / (2 * BRAKE), 0.0))
    return bumper_gap - required


def simulate(data: dict[str, np.ndarray], stress: float, delay: float,
             policy: str, threshold: tuple[float, float], split: int,
             initial_gap: float) -> dict[str, float | int | str]:
    leader_target = data["speed_av"][split:]
    source_acc = np.column_stack((data["acc_av"], data["acc_sv1"], data["acc_sv2"]))
    disagreement = source_acc - np.median(source_acc, axis=1, keepdims=True)
    forcing = disagreement[split:]
    lengths = np.asarray([data["dim_x_av"][split], data["dim_x_sv1"][split],
                          data["dim_x_sv2"][split]])
    velocity = np.asarray([data["speed_av"][split], data["speed_sv1"][split],
                           data["speed_sv2"][split]], dtype=float)
    position = np.asarray([0.0, -initial_gap, -2.0 * initial_gap])
    delay_steps = int(round(delay / DT))
    velocity_history = [velocity.copy() for _ in range(delay_steps + 1)]
    active_source = True
    margins: list[float] = []
    participation: list[int] = []
    first_violation = len(leader_target) * DT
    recovered = False
    violation_seen = False

    for k, target_speed in enumerate(leader_target):
        velocity[0] += np.clip((target_speed - velocity[0]) / DT, AMIN, AMAX) * DT
        delayed = velocity_history[max(0, len(velocity_history) - 1 - delay_steps)]
        residual = abs(forcing[k, 1])
        if policy != "all_coupled":
            if active_source and residual > threshold[0]:
                active_source = False
            elif not active_source and residual < threshold[1]:
                active_source = True
        parent2 = 1 if active_source else 0
        parents = (0, parent2)
        for follower, parent in ((1, parents[0]), (2, parents[1])):
            gap = position[parent] - position[follower]
            desired = STANDSTILL + TIME_HEADWAY * velocity[follower]
            command = KP * (gap - desired) + KV * (delayed[parent] - velocity[follower])
            command += stress * forcing[k, follower]
            if policy == "gated_safety_filter" and follower == 2:
                local_margin = safety_margin(position[1] - position[2], velocity[2],
                                              velocity[1], lengths[2], lengths[1])
                if local_margin < 5.0:
                    command = AMIN
            velocity[follower] += np.clip(command, AMIN, AMAX) * DT
        position += velocity * DT
        velocity_history.append(velocity.copy())
        m1 = safety_margin(position[0] - position[1], velocity[1], velocity[0],
                           lengths[1], lengths[0])
        # The communication parent may switch, but the physical collision pair
        # remains vehicles 1 and 2 in their longitudinal road order.
        m2 = safety_margin(position[1] - position[2], velocity[2],
                           velocity[1], lengths[2], lengths[1])
        margin = min(m1, m2)
        margins.append(margin)
        participation.append(3 if active_source else 2)
        if margin < 0 and not violation_seen:
            first_violation = k * DT
            violation_seen = True
        if violation_seen and margin >= 0:
            recovered = True

    array = np.asarray(margins)
    return {
        "policy": policy,
        "stress": stress,
        "delay_s": delay,
        "first_violation_s": first_violation,
        "minimum_margin_m": float(array.min()),
        "within_margin_fraction": float(np.mean(array >= 0)),
        "recovered_after_violation": int(recovered),
        "minimum_participating": min(participation),
    }


def main() -> None:
    data = load()
    split = int(0.6 * len(data["time"]))
    calibration_acc = np.column_stack((data["acc_av"][:split], data["acc_sv1"][:split],
                                       data["acc_sv2"][:split]))
    residual = np.abs(calibration_acc - np.median(calibration_acc, axis=1, keepdims=True))
    threshold = (float(np.quantile(residual, 0.9)), float(np.quantile(residual, 0.6)))
    initial_gap = 0.5 * float(np.quantile(data["distance_av_headway"][:split], 0.1))
    policies = ("all_coupled", "gated", "gated_safety_filter")
    rows = [simulate(data, stress, delay, policy, threshold, split, initial_gap)
            for stress in STRESS for delay in DELAYS for policy in policies]
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "vehicle_physical_tasks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source_rows": len(data["time"]), "calibration_rows": split,
        "heldout_rows": len(data["time"]) - split,
        "gate_high_mps2": threshold[0], "gate_low_mps2": threshold[1],
        "initial_pair_gap_m": initial_gap,
        "conditions": len(rows),
        "claim_boundary": "public-data-calibrated computational vehicle model",
    }
    (RESULTS / "vehicle_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
