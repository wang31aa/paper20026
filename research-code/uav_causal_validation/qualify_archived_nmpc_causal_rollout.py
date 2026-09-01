#!/usr/bin/env python3
"""Qualify archived EPFL NMPC state--input trajectories by causal reintegration.

This deliberately does not claim that the missing historical optimiser path has
been reproduced.  It asks the narrower, identifiable question: do the archived
applied controls generate the archived subsequent states under the archived
homogeneous double-integrator plant?
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def rows(value: np.ndarray, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=float).squeeze()
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[1] == width:
        return array
    if array.shape[0] == width:
        return array.T
    raise ValueError(f"cannot orient {array.shape} as time rows of width {width}")


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((left - right) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    comparison_workspace_sha256 = None
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        result = subprocess.run(
            ["unzip", "-q", str(args.archive), "-d", str(root)],
            capture_output=True,
            check=False,
        )
        workspaces = sorted(p for p in root.rglob("workspace.mat") if "/mpc/" in str(p))
        if not workspaces:
            raise RuntimeError(f"no MPC workspace found: {result.stderr.decode(errors='replace')}")

        ineligible: list[dict[str, str]] = []
        for workspace in workspaces:
            if str(workspace).endswith("dataset/data/mpc/comparison/01/workspace.mat"):
                comparison_workspace_sha256 = hashlib.sha256(workspace.read_bytes()).hexdigest()
            data = loadmat(workspace, squeeze_me=True, struct_as_record=False)
            if "U_history" not in data:
                ineligible.append(
                    {
                        "path": str(workspace.relative_to(root)),
                        "reason": "no archived applied-control history",
                    }
                )
                continue
            parameters = data["S"]
            agents = int(parameters.nb_agents)
            width = 3 * agents
            position = rows(data["pos_history"], width)
            velocity = rows(data["vel_history"], width)
            control = rows(data["U_history"], width)
            time = np.asarray(data["time_history"], dtype=float).reshape(-1)
            length = min(len(control), len(position) - 1, len(velocity) - 1)
            dt = np.diff(time[: length + 1]).reshape(-1, 1)
            predicted_velocity = velocity[:length] + dt * control[:length]
            predicted_position = position[:length] + dt * velocity[:length] + 0.5 * dt**2 * control[:length]
            records.append(
                {
                    "path": str(workspace.relative_to(root)),
                    "samples": int(length),
                    "agents": agents,
                    "velocity_transition_rmse": rmse(predicted_velocity, velocity[1 : length + 1]),
                    "position_transition_rmse": rmse(predicted_position, position[1 : length + 1]),
                    "control_integral": float(np.sum(control[:length] ** 2 * dt)),
                    "finite": bool(
                        np.isfinite(position).all()
                        and np.isfinite(velocity).all()
                        and np.isfinite(control).all()
                    ),
                }
            )

    tolerance = 1e-10
    max_velocity = max(float(record["velocity_transition_rmse"]) for record in records)
    max_position = max(float(record["position_transition_rmse"]) for record in records)
    criteria = {
        "reference_workspace_identity": comparison_workspace_sha256
        == "d96c35cb20551e3db3d6f7dbf2f8e1503277a739fb8fbe944a950791fdbf36a4",
        "all_control_eligible_mpc_records_parsed": len(records) + len(ineligible) == 92,
        "all_records_finite": all(bool(record["finite"]) for record in records),
        "all_velocity_transitions_reintegrate": max_velocity <= tolerance,
        "all_position_transitions_reintegrate": max_position <= tolerance,
    }
    report = {
        "qualification": "archived_applied_control_causal_rollout",
        "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        "reference_workspace_sha256": comparison_workspace_sha256,
        "archive_container_note": (
            "The local ZIP has recoverable prefix/offset damage; qualification is bound to the "
            "extracted reference-workspace hash and record-level numerical checks, not to this "
            "temporary container hash."
        ),
        "mpc_records": len(records),
        "ineligible_mpc_records": len(ineligible),
        "transition_tolerance": tolerance,
        "maximum_velocity_transition_rmse": max_velocity,
        "maximum_position_transition_rmse": max_position,
        "criteria": criteria,
        "qualified": all(criteria.values()),
        "scope": (
            "Exact causal reconstruction of archived state transitions from archived applied "
            "controls under the archived homogeneous plant; not source-controller optimisation "
            "reproduction, a counterfactual intervention, flight, HIL or hardware evidence."
        ),
        "records": records,
        "ineligible_records": ineligible,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    raise SystemExit(0 if report["qualified"] else 2)


if __name__ == "__main__":
    main()
