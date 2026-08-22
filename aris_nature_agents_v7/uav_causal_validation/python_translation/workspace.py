from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.io import loadmat

from .model import SwarmParameters, join_state


def _scalar(value, default=None):
    if value is None:
        return default
    array = np.asarray(value).squeeze()
    return array.item() if array.ndim == 0 else array


@dataclass
class OfficialWorkspace:
    path: Path
    parameters: SwarmParameters
    initial_state: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    controls: np.ndarray | None
    times: np.ndarray
    end_line: float


def load_official_workspace(path: Path) -> OfficialWorkspace:
    raw = loadmat(path, simplify_cells=True)
    source = raw.get("S", raw.get("p_swarm"))
    if source is None:
        raise ValueError("workspace has neither S nor p_swarm")
    positions = np.asarray(raw["pos_history"], dtype=float)
    velocities = np.asarray(raw["vel_history"], dtype=float)
    times = np.asarray(raw["time_history"], dtype=float).reshape(-1)
    n = int(_scalar(source.get("nb_agents"), positions.shape[1] // 3))
    cylinders = np.asarray(source.get("cylinders", np.empty((3, 0))), dtype=float)
    if cylinders.size == 1 and float(cylinders.reshape(-1)[0]) == 0:
        cylinders = np.empty((0, 3))
    else:
        cylinders = cylinders.reshape(3, -1).T
    controls = raw.get("U_history", raw.get("u_history"))
    if controls is not None:
        controls = np.asarray(controls, dtype=float)
        if controls.shape[0] == 3 * n and controls.shape[1] != 3 * n:
            controls = controls.T
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.1
    parameters = SwarmParameters(
        n=n, dt=dt, horizon_seconds=4.0,
        max_neighbours=int(_scalar(source.get("max_neig"), min(3, n - 1))),
        communication_radius=float(_scalar(source.get("r"), 150.0)),
        reference_speed=float(_scalar(source.get("v_ref", source.get("v_swarm")), 0.5)),
        reference_direction=np.asarray(source.get("u_ref", source.get("u_migration", [1, 0, 0])), dtype=float).reshape(3),
        reference_distance=float(_scalar(source.get("d_ref", source.get("d")), 0.8)),
        maximum_acceleration=float(_scalar(source.get("max_a"), 2.0)),
        collision_radius=float(_scalar(source.get("r_coll"), 0.1)),
        cylinders=cylinders,
    )
    initial_state = join_state(positions[0].reshape(n, 3), velocities[0].reshape(n, 3))
    map_data = raw.get("map", {})
    if isinstance(map_data, dict) and "end_line" in map_data:
        end_line = float(_scalar(map_data["end_line"]))
    else:
        # It is a recorded observable only, not an invented success threshold.
        end_line = float(np.min(positions[-1].reshape(n, 3)[:, 0]) - 1e-9)
    return OfficialWorkspace(path, parameters, initial_state, positions, velocities,
                             controls, times, end_line)

