#!/usr/bin/env python3
"""Fail-closed qualification of a native pinned-acados source replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def minimum_pair(positions: np.ndarray) -> float:
    return float(
        min(
            np.linalg.norm(frame[i] - frame[j])
            for frame in positions
            for i in range(frame.shape[0] - 1)
            for j in range(i + 1, frame.shape[0])
        )
    )


def as_time_rows(value: np.ndarray, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=float).squeeze()
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[1] == width:
        return array
    if array.shape[0] == width:
        return array.T
    raise ValueError(f"cannot orient array {array.shape} as time rows of width {width}")


parser = argparse.ArgumentParser()
parser.add_argument("workspace", type=Path)
parser.add_argument("native_replay", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

source = loadmat(args.workspace, squeeze_me=True, struct_as_record=False)
native = loadmat(args.native_replay, squeeze_me=True, struct_as_record=False)
parameters = source["S"]
n_agents = int(parameters.nb_agents)
width = 3 * n_agents
recorded_p = as_time_rows(source["pos_history"], width).reshape(-1, n_agents, 3)
recorded_v = as_time_rows(source["vel_history"], width).reshape(-1, n_agents, 3)
recorded_u = as_time_rows(source["U_history"], width).reshape(-1, n_agents, 3)
native_p = as_time_rows(native["pos_history"], width).reshape(-1, n_agents, 3)
native_v = as_time_rows(native["vel_history"], width).reshape(-1, n_agents, 3)
native_u = as_time_rows(native["U_history"], width).reshape(-1, n_agents, 3)
recorded_t = np.asarray(source["time_history"], dtype=float).reshape(-1)
native_t = np.asarray(native["time_history"], dtype=float).reshape(-1)
dt = float(np.median(np.diff(recorded_t)))

length = min(len(recorded_p), len(native_p))
position_rmse = float(np.sqrt(np.mean((native_p[:length] - recorded_p[:length]) ** 2)))
position_range = float(np.ptp(recorded_p[:length]))
normalized_rmse = position_rmse / max(position_range, 1e-12)
recorded_speed = np.linalg.norm(recorded_v[:length], axis=2)
native_speed = np.linalg.norm(native_v[:length], axis=2)
recorded_pair = minimum_pair(recorded_p[:length])
native_pair = minimum_pair(native_p[:length])
recorded_energy = float(np.sum(recorded_u**2) * dt)
native_energy = float(np.sum(native_u**2) * dt)
collision_radius = float(parameters.r_coll)
recorded_collision = recorded_pair < 2.0 * collision_radius
native_collision = native_pair < 2.0 * collision_radius
direction = np.asarray(parameters.u_migration, dtype=float).reshape(-1)
direction /= np.linalg.norm(direction)
endpoint = float(np.min(recorded_p[-1] @ direction))
completed = bool(np.all(native_p[-1] @ direction >= endpoint))
statuses = np.asarray(native.get("status", []), dtype=float).reshape(-1)

duration_recorded = float(recorded_t[-1] - recorded_t[0])
duration_native = float(native_t[-1] - native_t[0])
criteria = {
    "complete_finite_trajectory": bool(
        len(native_p) >= 2
        and np.isfinite(native_p).all()
        and np.isfinite(native_v).all()
        and np.isfinite(native_u).all()
    ),
    "position_rmse": position_rmse <= 0.10,
    "normalized_position_rmse": normalized_rmse <= 0.05,
    "completion_time": abs(duration_native - duration_recorded)
    <= max(0.5, 0.05 * duration_recorded),
    "minimum_pair_distance": abs(native_pair - recorded_pair)
    <= max(0.05, 0.05 * recorded_pair),
    "mean_speed": abs(float(native_speed.mean() - recorded_speed.mean()))
    <= max(0.05, 0.05 * float(recorded_speed.mean())),
    "p95_speed": abs(
        float(np.percentile(native_speed, 95) - np.percentile(recorded_speed, 95))
    )
    <= max(0.05, 0.05 * float(np.percentile(recorded_speed, 95))),
    "control_integral": abs(native_energy - recorded_energy)
    <= 0.10 * max(recorded_energy, 1e-12),
    "collision_event": native_collision == recorded_collision,
    "completion_classification": completed,
    "all_solver_steps_passed": bool(statuses.size) and bool(np.all(statuses == 0)),
}
report = {
    "qualification": "source_controller_replay",
    "implementation": "official MATLAB/Octave source with pinned acados 91067da",
    "free_running": True,
    "recorded_intermediate_state_used": False,
    "frozen_tolerances_changed_after_execution": False,
    "metrics": {
        "position_rmse": position_rmse,
        "normalized_position_rmse": normalized_rmse,
        "completion_time_recorded": duration_recorded,
        "completion_time_native": duration_native,
        "minimum_pair_recorded": recorded_pair,
        "minimum_pair_native": native_pair,
        "mean_speed_recorded": float(recorded_speed.mean()),
        "mean_speed_native": float(native_speed.mean()),
        "p95_speed_recorded": float(np.percentile(recorded_speed, 95)),
        "p95_speed_native": float(np.percentile(native_speed, 95)),
        "control_integral_recorded": recorded_energy,
        "control_integral_native": native_energy,
        "solver_steps": int(statuses.size),
        "solver_nonzero_statuses": int(np.count_nonzero(statuses)),
    },
    "criteria": criteria,
    "qualified": bool(all(criteria.values())),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["qualified"] else 2)
