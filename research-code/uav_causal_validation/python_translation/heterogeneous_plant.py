"""Pre-specified heterogeneous UAV plant for paired causal experiments.

The official source-replay plant remains unchanged and homogeneous.  This
module is used only after replay qualification, with parameters identified on
development records and frozen before held-out intervention runs.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .model import join_state, split_state


@dataclass(frozen=True)
class HeterogeneousPlant:
    acceleration_effectiveness: np.ndarray
    linear_drag: np.ndarray
    actuator_time_constant: np.ndarray

    def validated(self, n: int) -> "HeterogeneousPlant":
        beta = np.asarray(self.acceleration_effectiveness, dtype=float).reshape(-1)
        drag = np.asarray(self.linear_drag, dtype=float).reshape(-1)
        tau = np.asarray(self.actuator_time_constant, dtype=float).reshape(-1)
        if not (len(beta) == len(drag) == len(tau) == n):
            raise ValueError("heterogeneous parameter vectors must have one value per agent")
        if not (np.isfinite(beta).all() and np.isfinite(drag).all() and np.isfinite(tau).all()):
            raise ValueError("heterogeneous parameters must be finite")
        if np.any(beta <= 0) or np.any(drag < 0) or np.any(tau < 0):
            raise ValueError("require beta>0, drag>=0 and tau>=0")
        return HeterogeneousPlant(beta, drag, tau)

    @classmethod
    def homogeneous(cls, n: int) -> "HeterogeneousPlant":
        return cls(np.ones(n), np.zeros(n), np.zeros(n))


def plant_step(
    state: np.ndarray,
    applied_acceleration: np.ndarray,
    requested_acceleration: np.ndarray,
    disturbance: np.ndarray,
    dt: float,
    plant: HeterogeneousPlant,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one interval with an exact held-input linear update.

    Each axis obeys ``p_dot=v``, ``v_dot=beta*a-drag*v+w`` and, when
    ``tau>0``, ``a_dot=(a_requested-a)/tau``.  A matrix exponential gives a
    deterministic update, avoiding integration-step confounding between the
    homogeneous and heterogeneous arms.
    """
    from scipy.linalg import expm

    n = len(np.asarray(plant.acceleration_effectiveness).reshape(-1))
    plant = plant.validated(n)
    position, velocity = split_state(state, n)
    applied = np.asarray(applied_acceleration, dtype=float).reshape(n, 3)
    requested = np.asarray(requested_acceleration, dtype=float).reshape(n, 3)
    forcing = np.asarray(disturbance, dtype=float).reshape(n, 3)
    next_position = np.empty_like(position)
    next_velocity = np.empty_like(velocity)
    next_applied = np.empty_like(applied)
    for i in range(n):
        beta = float(plant.acceleration_effectiveness[i])
        drag = float(plant.linear_drag[i])
        tau = float(plant.actuator_time_constant[i])
        for axis in range(3):
            if tau == 0.0:
                # Requested acceleration is applied throughout this interval.
                a = requested[i, axis]
                matrix = np.array([[0.0, 1.0], [0.0, -drag]])
                offset = np.array([0.0, beta * a + forcing[i, axis]])
                augmented = np.block([[matrix, offset[:, None]], [np.zeros((1, 3))]])
                propagated = expm(augmented * dt) @ np.r_[position[i, axis], velocity[i, axis], 1.0]
                next_position[i, axis], next_velocity[i, axis] = propagated[:2]
                next_applied[i, axis] = a
            else:
                matrix = np.array(
                    [[0.0, 1.0, 0.0], [0.0, -drag, beta], [0.0, 0.0, -1.0 / tau]]
                )
                offset = np.array([0.0, forcing[i, axis], requested[i, axis] / tau])
                augmented = np.block([[matrix, offset[:, None]], [np.zeros((1, 4))]])
                propagated = expm(augmented * dt) @ np.r_[
                    position[i, axis], velocity[i, axis], applied[i, axis], 1.0
                ]
                next_position[i, axis], next_velocity[i, axis], next_applied[i, axis] = propagated[:3]
    return join_state(next_position, next_velocity), next_applied


def heterogeneity_index(plant: HeterogeneousPlant) -> float:
    """Dimensionless dispersion diagnostic; not a fitted task threshold."""
    beta = np.asarray(plant.acceleration_effectiveness, dtype=float)
    drag = np.asarray(plant.linear_drag, dtype=float)
    tau = np.asarray(plant.actuator_time_constant, dtype=float)
    terms = [np.std(beta) / max(np.mean(beta), 1e-12)]
    for value in (drag, tau):
        terms.append(np.std(value) / max(np.mean(value), 1e-12) if np.mean(value) > 0 else 0.0)
    return float(np.sqrt(np.sum(np.square(terms))))
