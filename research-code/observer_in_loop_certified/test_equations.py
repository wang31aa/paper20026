#!/usr/bin/env python3
"""Equation and information-boundary regression tests."""
import unittest

import numpy as np

import run_benchmark as m


class EquationTests(unittest.TestCase):
    def test_graphs_and_rates(self):
        for W in m.topologies().values():
            L, g, mu, smin = m.graph_constants(W)
            self.assertLess(np.max(np.abs(L.sum(axis=1))), 1e-14)
            self.assertTrue(np.all(g > 0)); self.assertGreater(mu, 0)
            self.assertGreater(smin, 0)
            for level in m.LEVELS:
                c = m.certificate(W, level)
                self.assertGreater(c['tracking_rate'], 0)
                self.assertGreater(c['observer_rate_proved'], 0)
                self.assertLessEqual(c['observer_rate_used'],
                                     c['observer_rate_proved'])

    def test_controller_uses_estimate_and_exact_error_identity(self):
        rng = np.random.default_rng(20260730)
        for W in m.topologies().values():
            c = m.certificate(W, .30); L1 = c['L'][:m.N, :m.N]
            Xf = rng.normal(size=(m.N, m.D)); sh = rng.normal(size=(m.N, m.D))
            s = rng.normal(size=m.D)
            u, oracle = m.controller_values(Xf, sh, L1, c['pin'], s)
            e = Xf - s; eta = sh - s
            expected = -m.ALPHA * (L1 @ e - c['pin'][:, None] * eta)
            self.assertTrue(np.allclose(u, expected, atol=2e-14, rtol=2e-14))
            self.assertTrue(np.allclose(u - oracle,
                                        m.ALPHA * c['pin'][:, None] * eta,
                                        atol=2e-14, rtol=2e-14))
            # Changing the true target while holding Xf and sh fixed cannot
            # change the applied observer-loop control.
            u2, _ = m.controller_values(Xf, sh, L1, c['pin'], s + 7)
            self.assertTrue(np.array_equal(u, u2))
            self.assertTrue(np.array_equal(
                m.policy_control(Xf, s, c, 'observer_loop', sh), u))
            self.assertTrue(np.array_equal(
                m.policy_control(Xf, s, c, 'oracle_target'), oracle))
            Wf = W[:, :m.N]
            follower_L = np.diag(Wf.sum(axis=1)) - Wf
            self.assertTrue(np.allclose(
                m.policy_control(Xf, s, c, 'no_target'),
                -m.ALPHA * follower_L @ Xf, atol=2e-14, rtol=2e-14))

    def test_rhs_tracking_and_observer_error_equations(self):
        rng = np.random.default_rng(937)
        t = .371
        for W in m.topologies().values():
            c = m.certificate(W, .15)
            X = rng.normal(size=(m.N + 1, m.D)); sh = rng.normal(size=(m.N, m.D))
            dy = m.rhs(t, m.pack(X, sh), c); dX, dsh = m.unpack(dy)
            s = X[m.N]; e = X[:m.N] - s; eta = sh - s; q = m.forcing(t)
            w = ((m.A0 - c['a'])[:, None] * s +
                 (c['b'] - m.B0)[:, None] * (np.tanh(s) + q))
            expected_de = (-c['a'][:, None] * e + c['b'][:, None] *
                (np.tanh(X[:m.N]) - np.tanh(s)) - m.ALPHA *
                (c['L'][:m.N, :m.N] @ e) + w + m.ALPHA *
                c['pin'][:, None] * eta)
            self.assertTrue(np.allclose(dX[:m.N] - dX[m.N], expected_de,
                                        atol=5e-14, rtol=5e-14))
            expected_deta = (-m.A0 * sh + m.B0 * (np.tanh(sh) + q) -
                (-m.A0 * s + m.B0 * (np.tanh(s) + q)) -
                m.GAMMA_STATE * (c['L'][:m.N, :m.N] @ eta))
            self.assertTrue(np.allclose(dsh - dX[m.N], expected_deta,
                                        atol=5e-14, rtol=5e-14))

    def test_comparison_components_start_and_remain_nonnegative(self):
        t = np.linspace(0, m.T_END, 401)
        for W in m.topologies().values():
            for level in m.LEVELS:
                c = m.certificate(W, level)
                hom, mismatch, observer, total = m.comparison_components(
                    t, 2.3, 1.7, c)
                self.assertAlmostEqual(total[0], 2.3, places=14)
                self.assertTrue(np.all(hom >= 0)); self.assertTrue(np.all(mismatch >= 0))
                self.assertTrue(np.all(observer >= 0)); self.assertTrue(np.all(total > 0))

    def test_rk4_recomputes_observer_and_controller_at_all_four_stages(self):
        rng = np.random.default_rng(4421); t = .17; dt = 3e-4
        W = m.topologies()['branch']; c = m.certificate(W, .15)
        X = rng.normal(size=(m.N + 1, m.D)); sh = rng.normal(size=(m.N, m.D))
        y = m.pack(X, sh)

        def independent_rhs(tt, yy):
            xx, ss = m.unpack(yy); q = m.forcing(tt); L1 = c['L'][:m.N, :m.N]
            ff = np.empty_like(xx)
            ff[:m.N] = (-c['a'][:, None] * xx[:m.N] + c['b'][:, None] *
                (np.tanh(xx[:m.N]) + q) - m.ALPHA *
                (L1 @ xx[:m.N] - c['pin'][:, None] * ss))
            ff[m.N] = -m.A0 * xx[m.N] + m.B0 * (np.tanh(xx[m.N]) + q)
            s_aug = np.vstack([ss, xx[m.N]])
            dss = (-m.A0 * ss + m.B0 * (np.tanh(ss) + q) -
                   m.GAMMA_STATE * (c['L'][:m.N] @ s_aug))
            return m.pack(ff, dss)

        k1 = independent_rhs(t, y)
        k2 = independent_rhs(t + dt / 2, y + dt * k1 / 2)
        k3 = independent_rhs(t + dt / 2, y + dt * k2 / 2)
        k4 = independent_rhs(t + dt, y + dt * k3)
        expected = y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        calls = []
        original = m.controller_values
        def traced(Xf, estimate, L1, pin, target):
            calls.append((Xf.copy(), estimate.copy(), target.copy()))
            return original(Xf, estimate, L1, pin, target)
        m.controller_values = traced
        try:
            actual = m.rk4(t, y, dt, c)
        finally:
            m.controller_values = original
        self.assertTrue(np.allclose(actual, expected, atol=2e-15, rtol=2e-15))
        self.assertEqual(len(calls), 4)
        self.assertTrue(any(not np.array_equal(calls[0][0], call[0]) for call in calls[1:]))
        self.assertTrue(any(not np.array_equal(calls[0][1], call[1]) for call in calls[1:]))

    def test_matched_initial_conditions_and_plant_only_policy_rhs(self):
        X1, sh1 = m.initial_conditions(3); X2, sh2 = m.initial_conditions(3)
        self.assertTrue(np.array_equal(X1, X2)); self.assertTrue(np.array_equal(sh1, sh2))
        t = .23; W = m.topologies()['cyclic']; c = m.certificate(W, .30)
        for policy in ('oracle_target', 'no_target'):
            got = m.plant_rhs(t, X1.ravel(), c, policy).reshape(m.N + 1, m.D)
            q = m.forcing(t); expected = np.empty_like(X1)
            expected[:m.N] = (-c['a'][:, None] * X1[:m.N] +
                c['b'][:, None] * (np.tanh(X1[:m.N]) + q) +
                m.policy_control(X1[:m.N], X1[m.N], c, policy))
            expected[m.N] = -m.A0 * X1[m.N] + m.B0 * (np.tanh(X1[m.N]) + q)
            self.assertTrue(np.allclose(got, expected, atol=2e-14, rtol=2e-14))

    def test_plant_only_rk4_four_stage_identity(self):
        X, _ = m.initial_conditions(8); y = X.ravel(); t = .11; dt = 4e-4
        c = m.certificate(m.topologies()['chain'], .05)
        for policy in ('oracle_target', 'no_target'):
            def independent_rhs(tt, yy):
                xx = yy.reshape(m.N + 1, m.D); q = m.forcing(tt)
                ff = np.empty_like(xx)
                ff[:m.N] = (-c['a'][:, None] * xx[:m.N] + c['b'][:, None] *
                    (np.tanh(xx[:m.N]) + q) +
                    m.policy_control(xx[:m.N], xx[m.N], c, policy))
                ff[m.N] = -m.A0 * xx[m.N] + m.B0 * (np.tanh(xx[m.N]) + q)
                return ff.ravel()
            k1 = independent_rhs(t, y)
            k2 = independent_rhs(t + dt / 2, y + dt * k1 / 2)
            k3 = independent_rhs(t + dt / 2, y + dt * k2 / 2)
            k4 = independent_rhs(t + dt, y + dt * k3)
            expected = y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            self.assertTrue(np.allclose(
                m.plant_rk4(t, y, dt, c, policy), expected,
                atol=2e-15, rtol=2e-15))


if __name__ == '__main__':
    unittest.main(verbosity=2)
