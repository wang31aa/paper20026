from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np

from .model import SwarmParameters, constraint_margins, exact_step, split_state
from .nmpc import TranslatedNMPC


@dataclass
class ClosedLoopSummary:
    completed: bool
    steps: int
    duration: float
    solver_failures: int
    minimum_pair_margin: float
    minimum_obstacle_margin: float
    control_integral: float


def run_nmpc(parameters: SwarmParameters, x0: np.ndarray, end_line: float,
             maximum_steps: int, output: Path, solver_iterations: int = 100) -> ClosedLoopSummary:
    controller = TranslatedNMPC(parameters, maximum_iterations=solver_iterations)
    x = np.asarray(x0, dtype=float).reshape(-1)
    rows = []
    pair_min, obstacle_min, energy, failures = np.inf, np.inf, 0.0, 0
    completed = False
    for k in range(maximum_steps):
        solution = controller.solve(x)
        u = solution.controls[0]
        margins = constraint_margins(x, u, parameters)
        if len(margins["pair"]): pair_min = min(pair_min, float(np.min(margins["pair"])))
        if len(margins["obstacle"]): obstacle_min = min(obstacle_min, float(np.min(margins["obstacle"])))
        failures += int(not solution.success)
        energy += float(np.dot(u, u) * parameters.dt)
        position, velocity = split_state(x, parameters.n)
        rows.append({"step": k, "time": k * parameters.dt, "state": x.tolist(),
                     "requested_control": u.tolist(), "applied_control": u.tolist(),
                     "solver_success": solution.success, "solver_status": solution.status,
                     "solver_iterations": solution.iterations,
                     "solver_seconds": solution.solve_seconds,
                     "objective": solution.objective,
                     "minimum_predicted_constraint_margin": solution.minimum_constraint_margin,
                     "minimum_pair_margin": float(np.min(margins["pair"])) if len(margins["pair"]) else None,
                     "minimum_obstacle_margin": float(np.min(margins["obstacle"])) if len(margins["obstacle"]) else None,
                     "mean_speed": float(np.mean(np.linalg.norm(velocity, axis=1)))})
        x = exact_step(x, u, parameters.dt, parameters.n)
        if np.all(split_state(x, parameters.n)[0][:, 0] > end_line):
            completed = True
            break
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n")
    return ClosedLoopSummary(completed, len(rows), len(rows) * parameters.dt, failures,
                             pair_min, obstacle_min, energy)

