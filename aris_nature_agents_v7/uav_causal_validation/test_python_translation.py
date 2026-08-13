#!/usr/bin/env python3
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from python_translation.model import SwarmParameters, closest_neighbours, constraint_margins, exact_step, join_state


def main():
    p = SwarmParameters(3, 0.1, 0.2, 1, 2.1, 0.5, np.array([1., 0., 0.]),
                        1.0, 2.0, 0.1, np.array([[5., 0., 0.5]]))
    position = np.array([[0., 0., 0.], [1., 0., 0.], [3., 0., 0.]])
    velocity = np.tile([0.5, 0., 0.], (3, 1))
    x = join_state(position, velocity)
    adjacency, ordered = closest_neighbours(position, p.communication_radius, p.max_neighbours)
    assert ordered.tolist() == [[1, 0, 1]], (adjacency, ordered)
    u = np.tile([1., 0., 0.], (3, 1)).reshape(-1)
    xn = exact_step(x, u, 0.1, 3)
    assert np.allclose(xn[:9].reshape(3, 3), position + .1 * velocity + .005 * u.reshape(3, 3))
    assert np.allclose(xn[9:].reshape(3, 3), velocity + .1 * u.reshape(3, 3))
    margins = constraint_margins(x, u, p)
    assert np.min(margins["input"]) > 0
    report = {"validator": "PASS", "tested": ["exact double integrator", "deterministic closest neighbours",
              "component input, pair collision and cylinder constraints"],
              "scope": "translation unit tests; not source-controller replay"}
    path = HERE / "results" / "python_translation_unit_tests.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
