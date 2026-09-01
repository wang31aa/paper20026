#!/usr/bin/env python3
"""Fail-closed numerical checks for the critical-function theorem."""
from __future__ import annotations

import math
import random


def psi(rho, modes, eps_p, eps_u, h):
    values = []
    for a, d0, db, u, v in modes:
        r = max(d0 + rho * db - u, 0.0) / a
        p = r + v * h / rho
        values.extend((p / eps_p, r / eps_u))
    return max(values)


def main():
    rng = random.Random(20260812)
    checked = 0
    for _ in range(400):
        modes = [
            (
                rng.uniform(0.2, 4.0),
                rng.uniform(0.0, 2.0),
                rng.uniform(0.0, 1.5),
                rng.uniform(0.0, 2.5),
                rng.uniform(0.02, 2.0),
            )
            for _ in range(rng.randint(1, 8))
        ]
        eps_p = rng.uniform(0.1, 4.0)
        eps_u = rng.uniform(0.1, 4.0)
        h = rng.uniform(0.02, 2.0)
        for _ in range(50):
            x = math.exp(rng.uniform(-3.0, 3.0))
            y = math.exp(rng.uniform(-3.0, 3.0))
            theta = rng.random()
            z = theta * x + (1.0 - theta) * y
            lhs = psi(z, modes, eps_p, eps_u, h)
            rhs = theta * psi(x, modes, eps_p, eps_u, h) + (1.0 - theta) * psi(y, modes, eps_p, eps_u, h)
            if lhs > rhs + 2e-12 * max(1.0, abs(rhs)):
                raise AssertionError((lhs, rhs, x, y, theta))
            checked += 1

        grid = [10 ** (-3 + 6 * j / 4000) for j in range(4001)]
        feasible = [psi(x, modes, eps_p, eps_u, h) <= 1.0 for x in grid]
        transitions = sum(a != b for a, b in zip(feasible, feasible[1:]))
        if transitions > 2:
            raise AssertionError(f"non-interval feasible set: {transitions} transitions")

    # Closed-form derivative check in an active scalar branch.
    for _ in range(1000):
        a = rng.uniform(0.2, 5.0)
        db = rng.uniform(0.01, 3.0)
        vh = rng.uniform(0.01, 3.0)
        rho_star = math.sqrt(a * vh / db)
        derivative = db / a - vh / (rho_star * rho_star)
        if abs(derivative) > 1e-12:
            raise AssertionError(derivative)

    print(f"PASS critical-function convexity: {checked} Jensen checks; 400 interval checks; 1000 stationary checks")


if __name__ == "__main__":
    main()
