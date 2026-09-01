#!/usr/bin/env python3
"""Independent equation and one-step RK4 identity tests for Phase 2.

The reference implementation below deliberately does not call the production
``derivative``, ``add``, disagreement helpers, or ``rk4_step`` when constructing
expected values.  Explicit follower-by-follower formulae make this a useful
regression guard rather than a second invocation of the code under test.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import intended_continuous as production


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "incoming/2026-07-28_author_matlab/source_tree/code"


def reference_phi(x: np.ndarray) -> np.ndarray:
    """Independent scalar-by-scalar Chua nonlinearity."""
    y = np.zeros(x.shape, dtype=float)
    for index in np.ndindex(x.shape[:-1]):
        u = float(x[index + (0,)])
        y[index + (0,)] = 0.5 * (abs(u + 1.0) - abs(u - 1.0))
    return y


def reference_derivative(state, A, B, H, F, coup, graph):
    """Direct transcription of the frozen continuous equations using loops."""
    xx, xg, MA, MB = state
    dxx = np.zeros_like(xx)
    dxg = np.zeros_like(xg)
    dMA = np.zeros_like(MA)
    dMB = np.zeros_like(MB)
    pxx = reference_phi(xx)
    pxg = reference_phi(xg)

    for i in range(7):
        plant_disagreement = np.zeros(3)
        reconstructed_disagreement = np.zeros(3)
        model_A_disagreement = np.zeros((3, 3))
        model_B_disagreement = np.zeros((3, 3))
        for j in range(7):
            weight = graph[i, j]
            plant_disagreement += weight * xx[j]
            reconstructed_disagreement += weight * xg[j]
            model_A_disagreement += weight * MA[j]
            model_B_disagreement += weight * MB[j]
        leader_weight = graph[i, 7]
        plant_disagreement += leader_weight * xx[7]
        reconstructed_disagreement += leader_weight * xx[7]
        model_A_disagreement += leader_weight * A[7]
        model_B_disagreement += leader_weight * B[7]

        dMA[i] = -coup * model_A_disagreement
        dMB[i] = -coup * model_B_disagreement
        dxx[i] = (
            A[i] @ xx[i]
            + B[i] @ pxx[i]
            - coup * (H[i] @ plant_disagreement)
        )
        dxg[i] = (
            MA[i] @ xg[i]
            + MB[i] @ pxg[i]
            - coup * (F @ reconstructed_disagreement)
        )

    dxx[7] = A[7] @ xx[7] + B[7] @ pxx[7]
    return dxx, dxg, dMA, dMB


def reference_shift(state, slopes, scale):
    return tuple(
        np.array(x, copy=True) + scale * np.array(k, copy=False)
        for x, k in zip(state, slopes)
    )


def reference_rk4_step(state, h, args):
    """Classical RK4 composed only from the independent derivative above."""
    k1 = reference_derivative(state, *args)
    k2 = reference_derivative(reference_shift(state, k1, h / 2.0), *args)
    k3 = reference_derivative(reference_shift(state, k2, h / 2.0), *args)
    k4 = reference_derivative(reference_shift(state, k3, h), *args)
    return tuple(
        np.array(x, copy=True) + h * (a + 2.0 * b + 2.0 * c + d) / 6.0
        for x, a, b, c, d in zip(state, k1, k2, k3, k4)
    )


def archived_case(graph_name: str):
    """Load the graph-specific archive while retaining its supplied gains."""
    path = SOURCE / f"exam1_{graph_name}.mat"
    raw = loadmat(path, variable_names=production.VARIABLES)
    A = np.stack([np.asarray(raw[f"A{i}{i}"], float) for i in range(1, 9)])
    B = np.stack([np.asarray(raw[f"B{i}{i}"], float) for i in range(1, 9)])
    H = np.stack([np.asarray(raw[f"H{i}{i}"], float) for i in range(1, 8)])
    F = np.asarray(raw["F"], float)
    coup = float(raw["Coup"].item())
    key = "L" if graph_name == "graph1" else "L2"
    graph = np.asarray(raw[key], float)[:7]
    return A, B, H, F, coup, graph


def deterministic_state():
    """Non-symmetric state chosen to exercise every equation block."""
    xx = np.arange(24, dtype=float).reshape(8, 3) / 17.0 - 0.55
    xg = np.arange(21, dtype=float).reshape(7, 3) / 13.0 - 0.72
    MA = np.empty((7, 3, 3), dtype=float)
    MB = np.empty_like(MA)
    base = np.arange(9, dtype=float).reshape(3, 3)
    for i in range(7):
        MA[i] = (i + 1.0) * np.eye(3) + (base - 4.0) / (31.0 + i)
        MB[i] = (8.0 - i) * np.eye(3) + (base.T - 3.0) / (37.0 + i)
    return xx, xg, MA, MB


class IntendedContinuousIdentityTests(unittest.TestCase):
    def assert_blocks_close(self, actual, expected, label):
        for name, got, want in zip(("xx", "xg", "MA", "MB"), actual, expected):
            np.testing.assert_allclose(
                got, want, rtol=2e-13, atol=2e-13,
                err_msg=f"{label}: {name} equation identity",
            )

    def test_derivative_identity_for_archived_graph1_and_graph2(self):
        state = deterministic_state()
        for graph_name in ("graph1", "graph2"):
            with self.subTest(graph=graph_name):
                args = archived_case(graph_name)
                expected = reference_derivative(state, *args)
                actual = production.derivative(state, *args)
                self.assert_blocks_close(actual, expected, graph_name)

    def test_complete_rk4_step_and_fixed_archived_gains(self):
        h = 0.001
        state = deterministic_state()
        for graph_name in ("graph1", "graph2"):
            with self.subTest(graph=graph_name):
                args = archived_case(graph_name)
                frozen_args = tuple(
                    np.array(value, copy=True) if isinstance(value, np.ndarray) else value
                    for value in args
                )
                expected = reference_rk4_step(state, h, args)
                actual = production.rk4_step(state, h, args)
                self.assert_blocks_close(actual, expected, f"{graph_name} RK4")

                # Phase 2 evolves MA/MB as states but holds the author-supplied
                # A/B/H/F, coupling gain and graph topology fixed at all stages.
                for position, (after, before) in enumerate(zip(args, frozen_args)):
                    if isinstance(after, np.ndarray):
                        np.testing.assert_array_equal(
                            after, before,
                            err_msg=f"{graph_name}: archived argument {position} mutated",
                        )
                    else:
                        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
