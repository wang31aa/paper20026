#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np

from python_translation.heterogeneous_plant import HeterogeneousPlant, plant_step
from python_translation.model import exact_step, join_state


def main():
    n, dt = 5, 0.1
    rng = np.random.default_rng(20260814)
    state = join_state(rng.normal(size=(n, 3)), rng.normal(size=(n, 3)))
    control = rng.normal(scale=0.2, size=(n, 3))
    zero = np.zeros_like(control)
    homogeneous = HeterogeneousPlant.homogeneous(n)
    propagated, applied = plant_step(state, zero, control, zero, dt, homogeneous)
    reference = exact_step(state, control, dt, n)
    homogeneous_error = float(np.max(np.abs(propagated - reference)))
    assert homogeneous_error < 1e-12
    assert np.allclose(applied, control)

    heterogeneous = HeterogeneousPlant(
        np.array([0.78, 0.91, 1.00, 1.08, 1.19]),
        np.array([0.02, 0.04, 0.06, 0.08, 0.10]),
        np.array([0.04, 0.07, 0.10, 0.13, 0.16]),
    )
    changed, _ = plant_step(state, zero, control, zero, dt, heterogeneous)
    heterogeneous_delta = float(np.linalg.norm(changed - reference))
    assert heterogeneous_delta > 0
    result = {
        "validator": "PASS",
        "homogeneous_limit_max_abs_error": homogeneous_error,
        "heterogeneous_counterfactual_delta": heterogeneous_delta,
        "scope": "plant semantics only; parameters are a unit-test fixture, not identified experimental values",
    }
    output = Path(__file__).resolve().parent / "results" / "heterogeneous_plant_unit_test.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
