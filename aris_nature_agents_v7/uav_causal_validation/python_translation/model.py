from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SwarmParameters:
    n: int
    dt: float
    horizon_seconds: float
    max_neighbours: int
    communication_radius: float
    reference_speed: float
    reference_direction: np.ndarray
    reference_distance: float
    maximum_acceleration: float
    collision_radius: float
    cylinders: np.ndarray
    safety_margin: float = 0.2

    @property
    def horizon_steps(self) -> int:
        return int(np.floor(self.horizon_seconds / self.dt))

    @property
    def component_acceleration_limit(self) -> float:
        return self.maximum_acceleration / np.sqrt(3.0)


ACADOS_CONFIG = {
    "prediction_horizon_seconds": 4.0,
    "parameterization": "multiple_shooting_unif_grid",
    "nlp_solver": "sqp",
    "nlp_solver_exact_hessian": False,
    "regularize_method": "no_regularize",
    "nlp_solver_max_iter": 7,
    "nlp_solver_tol_stat": 1.0,
    "nlp_solver_tol_eq": 1e-1,
    "nlp_solver_tol_ineq": 1e-2,
    "nlp_solver_tol_comp": 1e-1,
    "nlp_solver_step_length": 0.05,
    "qp_solver": "partial_condensing_hpipm",
    "qp_solver_iter_max": 10,
    "qp_solver_cond_N_rule": "horizon_steps/2",
    "qp_solver_warm_start": 0,
    "qp_solver_cond_ric_alg": 0,
    "qp_solver_ric_alg": 0,
    "integrator": "irk_gnsf",
    "integrator_stages": 4,
    "integrator_steps": 3,
    "cost_type": "nonlinear_ls",
}


def split_state(x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size != 6 * n:
        raise ValueError(f"expected {6*n} states, got {x.size}")
    # MATLAB Pos0(:) stores each 3-vector contiguously (Fortran order).
    return x[: 3 * n].reshape(n, 3), x[3 * n :].reshape(n, 3)


def join_state(position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    return np.concatenate((np.asarray(position).reshape(-1), np.asarray(velocity).reshape(-1)))


def exact_step(x: np.ndarray, u: np.ndarray, dt: float, n: int) -> np.ndarray:
    position, velocity = split_state(x, n)
    acceleration = np.asarray(u, dtype=float).reshape(n, 3)
    return join_state(position + dt * velocity + 0.5 * dt * dt * acceleration,
                      velocity + dt * acceleration)


def closest_neighbours(position: np.ndarray, radius: float, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce compute_closest_neighbors.m, including deterministic index ties."""
    p = np.asarray(position, dtype=float).reshape(-1, 3)
    n = len(p)
    adjacency = np.zeros((n, n), dtype=int)
    ordered = np.full((maximum, n), -1, dtype=int)
    distances = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2) ** 2
    np.fill_diagonal(distances, np.inf)
    for agent in range(n):
        candidates = [i for i in range(n) if distances[i, agent] < radius * radius]
        candidates.sort(key=lambda i: (distances[i, agent], i))
        for rank, neighbour in enumerate(candidates[:maximum]):
            ordered[rank, agent] = neighbour
            adjacency[neighbour, agent] = 1
    return adjacency, ordered


def residual_vector(x: np.ndarray, u: np.ndarray | None, parameters: SwarmParameters,
                    terminal: bool = False) -> np.ndarray:
    position, velocity = split_state(x, parameters.n)
    _, neighbours = closest_neighbours(position, parameters.communication_radius,
                                       parameters.max_neighbours)
    separation = []
    for agent in range(parameters.n):
        for rank in range(parameters.max_neighbours):
            neighbour = neighbours[rank, agent]
            if neighbour < 0:
                delta = np.array([parameters.reference_distance, 0.0, 0.0])
            else:
                delta = position[neighbour] - position[agent]
            separation.append((delta @ delta - parameters.reference_distance**2)
                              / parameters.max_neighbours)
    speed_sq = np.einsum("ij,ij->i", velocity, velocity)
    projection = velocity @ parameters.reference_direction
    # Official expression is singular at zero speed. Use IEEE-like large penalty,
    # never silently converting an undefined official cost to a favourable zero.
    direction = np.ones_like(speed_sq)
    moving = speed_sq > 1e-14
    direction[moving] = 1.0 - projection[moving]**2 / speed_sq[moving]
    navigation = speed_sq - parameters.reference_speed**2
    parts = [np.asarray(separation), direction, navigation]
    if not terminal:
        if u is None:
            raise ValueError("stage residual requires control")
        parts.append(0.4 * np.asarray(u, dtype=float).reshape(-1))
    return np.concatenate(parts)


def residual_weights(parameters: SwarmParameters, terminal: bool = False) -> np.ndarray:
    n, m = parameters.n, parameters.max_neighbours
    weights = [np.full(n * m, 1.0 / m), np.full(n, 5.0), np.full(n, 5.0)]
    if not terminal:
        weights.append(np.full(3 * n, 0.4))
    return np.concatenate(weights)


def stage_cost(x: np.ndarray, u: np.ndarray, parameters: SwarmParameters) -> float:
    y = residual_vector(x, u, parameters)
    return float(np.dot(residual_weights(parameters) * y, y))


def terminal_cost(x: np.ndarray, parameters: SwarmParameters) -> float:
    y = residual_vector(x, None, parameters, terminal=True)
    return float(np.dot(residual_weights(parameters, terminal=True) * y, y))


def constraint_margins(x: np.ndarray, u: np.ndarray, parameters: SwarmParameters) -> dict[str, np.ndarray]:
    position, _ = split_state(x, parameters.n)
    control = np.asarray(u, dtype=float).reshape(parameters.n, 3)
    input_margin = parameters.component_acceleration_limit - np.abs(control)
    pair = []
    for i in range(parameters.n - 1):
        for j in range(i + 1, parameters.n):
            pair.append(np.sum((position[j] - position[i]) ** 2) - 4 * parameters.collision_radius**2)
    obstacle = []
    cylinders = np.asarray(parameters.cylinders, dtype=float).reshape(-1, 3)
    for agent in range(parameters.n):
        for cx, cy, radius in cylinders:
            obstacle.append(np.sum((position[agent, :2] - [cx, cy]) ** 2)
                            - radius**2 - parameters.safety_margin**2)
    return {"input": input_margin.reshape(-1), "pair": np.asarray(pair),
            "obstacle": np.asarray(obstacle)}
