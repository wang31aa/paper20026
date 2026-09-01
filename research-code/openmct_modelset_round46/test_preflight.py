#!/usr/bin/env python3
import unittest
import numpy as np
import run_preflight as rp


class PreflightTests(unittest.TestCase):
    def test_graphs_are_target_rooted(self):
        for w in rp.topologies().values():
            l1 = rp.follower_laplacian(w)
            self.assertGreater(np.min(np.linalg.svd(l1, compute_uv=False)), 1e-8)

    def test_exact_corner_count(self):
        got = rp.corners(0.2, 1.1, rp.PROTOCOL["followers"])
        self.assertEqual(len(got), 32)
        self.assertEqual(len({tuple(x) for x in got}), 32)

    def test_reference_gain_not_used_as_vertex_population(self):
        vertices = rp.PROTOCOL["frozen_parameter_vertices"]
        self.assertEqual([x["id"] for x in vertices],
                         ["pooled", "leave_2ms_out", "leave_10ms_out", "leave_20ms_out"])


if __name__ == "__main__":
    unittest.main()
