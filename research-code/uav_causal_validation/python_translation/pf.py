"""Translation of the official SwarmLab v1.0 Vasarhelyi/PF controller."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

OFFICIAL_VASARHELYI_PARAMETERS = {
    "p_rep": 0.29,
    "r0_fric": 6.98,
    "C_fric": 0.06,
    "v_fric": 0.63,
    "p_fric": 3.34,
    "a_fric": 0.05,
    "r0_shill": 0.10,
    "v_shill": 0.81,
    "p_shill": 2.99,
    "a_shill": 1.17,
}


@dataclass(frozen=True)
class PFParameters:
    communication_radius: float
    maximum_neighbours: int
    collision_radius: float
    cylinders: np.ndarray
    migration_direction: np.ndarray
    reference_speed: float
    maximum_acceleration: float
    maximum_speed: float
    r0_rep: float
    p_rep: float
    r0_fric: float
    C_fric: float
    v_fric: float
    p_fric: float
    a_fric: float
    r0_shill: float
    v_shill: float
    p_shill: float
    a_shill: float


def _v_max(v_fric: float, distance: float, acceleration: float, gain: float) -> float:
    if distance < 0:
        value = 0.0
    elif 0 < distance * gain < acceleration / gain:
        value = distance * gain
    else:
        radicand = 2 * acceleration * distance - acceleration**2 / gain**2
        value = np.sqrt(max(0.0, radicand))
    return max(value, v_fric)


def vasarhelyi_velocity(position: np.ndarray, velocity: np.ndarray, p: PFParameters,
                        dt: float) -> tuple[np.ndarray, dict]:
    """Line-by-line numerical counterpart of SwarmLab v1.0 compute_vel_vasarhelyi."""
    position = np.asarray(position, dtype=float).reshape(-1, 3)
    velocity = np.asarray(velocity, dtype=float).reshape(-1, 3)
    n = len(position)
    command = np.zeros_like(position)
    agent_collisions = 0
    obstacle_collisions = 0
    minimum_obstacle_distance = 20.0
    adjacency = np.zeros((n, n), dtype=int)
    for agent in range(n):
        relative_position = position - position[agent]
        distance = np.linalg.norm(relative_position, axis=1)
        candidates = [i for i in range(n) if i != agent and distance[i] < p.communication_radius]
        if len(candidates) > p.maximum_neighbours:
            candidates.sort(key=lambda i: (distance[i], i))
            candidates = candidates[:p.maximum_neighbours]
        adjacency[agent, candidates] = 1
        agent_collisions += int(np.sum(distance < 2 * p.collision_radius) - 1)
        repulsion = np.zeros(3)
        friction = np.zeros(3)
        if candidates:
            relative_velocity = velocity - velocity[agent]
            velocity_norm = np.linalg.norm(relative_velocity, axis=1)
            for neighbour in candidates:
                # p_rel_u in MATLAB is -p_rel/dist, pointing from neighbour to agent.
                p_unit = -relative_position[neighbour] / distance[neighbour]
                if distance[neighbour] < p.r0_rep:
                    repulsion += p.p_rep * (p.r0_rep - distance[neighbour]) * p_unit
                else:
                    repulsion += p.p_rep * (distance[neighbour] - p.r0_rep) * -p_unit
                allowed = _v_max(p.v_fric, distance[neighbour] - p.r0_fric,
                                 p.a_fric, p.p_fric)
                if velocity_norm[neighbour] > allowed:
                    v_unit = -relative_velocity[neighbour] / velocity_norm[neighbour]
                    friction += p.C_fric * (velocity_norm[neighbour] - allowed) * v_unit
        obstacle = np.zeros(3)
        for cx, cy, radius in np.asarray(p.cylinders, dtype=float).reshape(-1, 3):
            planar = position[agent, :2] - [cx, cy]
            center_distance = np.linalg.norm(planar)
            surface_distance = center_distance - radius
            obstacle_collisions += int(surface_distance < p.collision_radius)
            minimum_obstacle_distance = min(minimum_obstacle_distance, surface_distance)
            virtual = planar / center_distance * p.v_shill
            relative_speed = np.linalg.norm(velocity[agent, :2] - virtual)
            allowed = _v_max(0.0, surface_distance - p.r0_shill,
                             p.a_shill, p.p_shill)
            if relative_speed > allowed:
                obstacle[:2] += (relative_speed - allowed) * (virtual - velocity[agent, :2]) / relative_speed
        command[agent] = repulsion + friction + obstacle + p.reference_speed * p.migration_direction
    speed = np.linalg.norm(command, axis=1)
    over_speed = speed > p.maximum_speed
    command[over_speed] *= (p.maximum_speed / speed[over_speed])[:, None]
    acceleration = (command - velocity) / dt
    acceleration_norm = np.linalg.norm(acceleration, axis=1)
    over_acceleration = acceleration_norm > p.maximum_acceleration
    command[over_acceleration] = velocity[over_acceleration] + (
        dt * p.maximum_acceleration * acceleration[over_acceleration]
        / acceleration_norm[over_acceleration, None])
    return command, {"adjacency": adjacency,
                     "agent_collisions": agent_collisions // 2,
                     "obstacle_collisions": obstacle_collisions,
                     "minimum_obstacle_distance": minimum_obstacle_distance}


def free_run(initial_position: np.ndarray, initial_velocity: np.ndarray,
             parameters: PFParameters, dt: float, maximum_steps: int,
             end_line: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    position = np.asarray(initial_position, dtype=float).reshape(-1, 3).copy()
    velocity = np.asarray(initial_velocity, dtype=float).reshape(-1, 3).copy()
    positions, velocities, commands, events = [position.copy()], [velocity.copy()], [], []
    for _ in range(maximum_steps):
        command, event = vasarhelyi_velocity(position, velocity, parameters, dt)
        velocity = command
        position = position + dt * velocity
        commands.append(command.copy()); events.append(event)
        positions.append(position.copy()); velocities.append(velocity.copy())
        if np.all(position[:, 0] > end_line):
            break
    return np.asarray(positions), np.asarray(velocities), np.asarray(commands), events


def build_official_pf(parameters: PFParameters):
    return lambda position, velocity, dt: vasarhelyi_velocity(position, velocity, parameters, dt)
