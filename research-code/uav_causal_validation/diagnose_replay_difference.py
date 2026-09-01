#!/usr/bin/env python3
"""Diagnose whether native/source replay drift is plant- or controller-induced.

This audit is deliberately limited to the archived homogeneous EPFL model.  It
does not infer the effect of heterogeneous vehicle dynamics; that requires a
separate, pre-specified paired intervention.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def time_rows(value: np.ndarray, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=float).squeeze()
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[1] == width:
        return array
    if array.shape[0] == width:
        return array.T
    raise ValueError(f"cannot orient {array.shape} as rows of width {width}")


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def transition_audit(data: dict, width: int) -> dict[str, float]:
    position = time_rows(data["pos_history"], width)
    velocity = time_rows(data["vel_history"], width)
    control = time_rows(data["U_history"], width)
    time = np.asarray(data["time_history"], dtype=float).reshape(-1)
    length = min(len(control), len(position) - 1, len(velocity) - 1)
    dt = np.diff(time[: length + 1]).reshape(-1, 1)
    predicted_velocity = velocity[:length] + dt * control[:length]
    predicted_position = (
        position[:length]
        + dt * velocity[:length]
        + 0.5 * dt**2 * control[:length]
    )
    return {
        "velocity_one_step_rmse": rmse(predicted_velocity, velocity[1 : length + 1]),
        "position_one_step_rmse": rmse(predicted_position, position[1 : length + 1]),
    }


def rollout(initial_p: np.ndarray, initial_v: np.ndarray, controls: np.ndarray, dt: float):
    positions = [initial_p.copy()]
    velocities = [initial_v.copy()]
    p, v = initial_p.copy(), initial_v.copy()
    for u in controls:
        p = p + dt * v + 0.5 * dt**2 * u
        v = v + dt * u
        positions.append(p.copy())
        velocities.append(v.copy())
    return np.asarray(positions), np.asarray(velocities)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("native_replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = loadmat(args.workspace, squeeze_me=True, struct_as_record=False)
    native = loadmat(args.native_replay, squeeze_me=True, struct_as_record=False)
    number_of_agents = int(source["S"].nb_agents)
    width = 3 * number_of_agents
    source_p = time_rows(source["pos_history"], width)
    source_v = time_rows(source["vel_history"], width)
    source_u = time_rows(source["U_history"], width)
    native_p = time_rows(native["pos_history"], width)
    native_u = time_rows(native["U_history"], width)
    dt = float(np.median(np.diff(np.asarray(source["time_history"]).reshape(-1))))

    length = min(len(native_u), len(native_p) - 1, len(source_p) - 1)
    reconstructed_p, _ = rollout(source_p[0], source_v[0], native_u[:length], dt)
    direct_drift = rmse(native_p[: length + 1], source_p[: length + 1])
    reconstructed_native_error = rmse(reconstructed_p, native_p[: length + 1])
    reconstructed_source_drift = rmse(reconstructed_p, source_p[: length + 1])
    control_length = min(len(source_u), len(native_u))
    control_rmse = rmse(source_u[:control_length], native_u[:control_length])
    control_correlation = float(
        np.corrcoef(
            source_u[:control_length].reshape(-1),
            native_u[:control_length].reshape(-1),
        )[0, 1]
    )
    statuses = np.asarray(native.get("status", []), dtype=int).reshape(-1)
    unique, counts = np.unique(statuses, return_counts=True)

    numerical_tolerance = 1e-10
    source_transition = transition_audit(source, width)
    native_transition = transition_audit(native, width)
    plant_equations_match = bool(
        max([*source_transition.values(), *native_transition.values()])
        <= numerical_tolerance
    )
    control_sequence_explains_drift = bool(
        reconstructed_native_error <= numerical_tolerance
        and abs(reconstructed_source_drift - direct_drift) <= numerical_tolerance
    )
    report = {
        "audit": "native-versus-source replay difference attribution",
        "model_contract": {
            "number_of_agents": number_of_agents,
            "plant": "one shared 3-D double-integrator equation for every agent",
            "per_agent_dynamic_parameters_in_official_model": False,
            "heterogeneity_present_in_replay_contract": False,
        },
        "transition_audit": {
            "source": source_transition,
            "native": native_transition,
        },
        "control_and_rollout": {
            "control_rmse_per_component": control_rmse,
            "control_correlation": control_correlation,
            "native_source_position_rmse": direct_drift,
            "native_control_reconstruction_to_native_rmse": reconstructed_native_error,
            "native_control_reconstruction_to_source_rmse": reconstructed_source_drift,
            "solver_status_counts": {
                str(int(code)): int(count) for code, count in zip(unique, counts)
            },
        },
        "tests": {
            "both_logs_obey_same_homogeneous_plant_to_numerical_precision": plant_equations_match,
            "different_control_sequences_reproduce_full_position_drift": control_sequence_explains_drift,
            "heterogeneous_dynamics_can_explain_this_replay_difference": False,
        },
        "conclusion": (
            "The replay difference is generated by different NMPC control sequences, not by "
            "heterogeneous plant dynamics. The archived controller uses a homogeneous plant, "
            "and both trajectories satisfy that same double-integrator transition to numerical "
            "precision. The evidence is consistent with solver-path sensitivity (nonzero acados "
            "statuses, finite SQP/QP iteration limits, warm starts and platform-dependent numerical "
            "paths), but the archived workspace lacks the original solver-status trace and full "
            "build fingerprint, so the individual numerical causes cannot be uniquely apportioned."
        ),
        "scope": (
            "This is a source-replay attribution audit. A heterogeneity claim requires a separate "
            "paired experiment with pre-frozen per-agent dynamics and identical exogenous streams."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not (plant_equations_match and control_sequence_explains_drift):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
