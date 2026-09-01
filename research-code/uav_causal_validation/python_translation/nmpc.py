from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import minimize

from .model import SwarmParameters, constraint_margins, exact_step, stage_cost, terminal_cost


@dataclass
class NMPCResult:
    controls: np.ndarray
    states: np.ndarray
    success: bool
    status: int
    message: str
    iterations: int
    solve_seconds: float
    objective: float
    minimum_constraint_margin: float


class TranslatedNMPC:
    """Direct single-shooting equivalent for qualification, not an acados claim."""

    def __init__(self, parameters: SwarmParameters, maximum_iterations: int = 100):
        self.p = parameters
        self.maximum_iterations = maximum_iterations
        self._warm_controls: np.ndarray | None = None

    def rollout(self, x0: np.ndarray, controls: np.ndarray) -> np.ndarray:
        states = [np.asarray(x0, dtype=float).reshape(-1)]
        for control in np.asarray(controls).reshape(self.p.horizon_steps, 3 * self.p.n):
            states.append(exact_step(states[-1], control, self.p.dt, self.p.n))
        return np.asarray(states)

    def initial_controls(self) -> np.ndarray:
        if self._warm_controls is None:
            return np.zeros((self.p.horizon_steps, 3 * self.p.n))
        return self._warm_controls.copy()

    def solve(self, x0: np.ndarray, edge_weights: np.ndarray | None = None) -> NMPCResult:
        shape = (self.p.horizon_steps, 3 * self.p.n)
        initial = self.initial_controls()

        def objective(flat: np.ndarray) -> float:
            controls = flat.reshape(shape)
            states = self.rollout(x0, controls)
            return sum(stage_cost(states[k], controls[k], self.p, edge_weights) * self.p.dt
                       for k in range(self.p.horizon_steps)) + terminal_cost(
                           states[-1], self.p, edge_weights)

        def inequalities(flat: np.ndarray) -> np.ndarray:
            controls = flat.reshape(shape)
            states = self.rollout(x0, controls)
            margins = []
            for k in range(self.p.horizon_steps):
                value = constraint_margins(states[k], controls[k], self.p)
                margins.extend((value["pair"], value["obstacle"]))
            return np.concatenate(margins) if margins else np.ones(1)

        limit = self.p.component_acceleration_limit
        start = time.perf_counter()
        result = minimize(objective, initial.reshape(-1), method="SLSQP",
                          bounds=[(-limit, limit)] * initial.size,
                          constraints={"type": "ineq", "fun": inequalities},
                          options={"maxiter": self.maximum_iterations, "ftol": 1e-6, "disp": False})
        elapsed = time.perf_counter() - start
        controls = result.x.reshape(shape)
        states = self.rollout(x0, controls)
        all_margins = inequalities(result.x)
        # Match the official trajectory-shift warm start exactly at the semantic level.
        self._warm_controls = np.vstack((controls[1:], controls[-1:]))
        return NMPCResult(controls, states, bool(result.success), int(result.status),
                          str(result.message), int(getattr(result, "nit", -1)), elapsed,
                          float(result.fun), float(np.min(all_margins)))
