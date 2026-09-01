#!/usr/bin/env python3
import unittest
import numpy as np
import run_observer_static as m


class GraphAndEquationTests(unittest.TestCase):
    def test_receiver_row_convention_and_root_reachability(self):
        L = m.full_laplacian()
        self.assertTrue(np.allclose(L.sum(axis=1), 0))
        reached = {7}
        while True:
            nxt = {i for i in range(7) if any(L[i,j] < 0 for j in reached)}
            old = len(reached); reached |= nxt
            if len(reached) == old: break
        self.assertEqual(reached, set(range(8)))

    def test_pinned_block_is_nonsingular_M_matrix(self):
        L1 = m.full_laplacian()[:7,:7]
        self.assertTrue(np.all(np.diag(L1) > 0))
        self.assertTrue(np.all((L1 - np.diag(np.diag(L1))) <= 0))
        self.assertTrue(np.all(np.linalg.eigvals(L1).real > 0))

    def test_correct_diagonal_stability_weight(self):
        a = m.graph_audit()
        self.assertGreater(a['S_min_eigenvalue'], 0)
        self.assertGreater(a['generalized_margin_mu'], 0)
        # Regression guard for the source manuscript's reciprocal/transposed formula.
        self.assertLess(a['regression_wrong_inverse_weight_min_eigenvalue'], 0)

    def test_model_observer_error_dynamics_identity(self):
        rng = np.random.default_rng(3); L1 = m.full_laplacian()[:7,:7]
        A0, B0 = m.chua_matrices(8)
        X = rng.normal(size=(8,3)); Ah = rng.normal(size=(7,3,3))
        Bh = rng.normal(size=(7,3,3)); sh = rng.normal(size=(7,3))
        dy = m.rhs(m.pack(X,Ah,Bh,sh)); _,dAh,dBh,_ = m.unpack(dy)
        self.assertTrue(np.allclose(dAh, -m.GAMMA_MODEL*np.einsum('ij,jkl->ikl',L1,Ah-A0)))
        self.assertTrue(np.allclose(dBh, -m.GAMMA_MODEL*np.einsum('ij,jkl->ikl',L1,Bh-B0)))

    def test_controller_induces_exact_static_error_equation(self):
        rng=np.random.default_rng(4); X=rng.normal(size=(8,3)); A0,B0=m.chua_matrices(8)
        Ah=np.repeat(A0[None],7,0); Bh=np.repeat(B0[None],7,0); sh=np.repeat(X[7][None],7,0)
        dX,_,_,_=m.unpack(m.rhs(m.pack(X,Ah,Bh,sh)))
        e=X[:7]-X[7]; As,Bs=zip(*(m.chua_matrices(i) for i in range(1,8)))
        As,Bs=np.stack(As),np.stack(Bs); w=np.einsum('nij,j->ni',As-A0,X[7])+np.einsum('nij,j->ni',Bs-B0,m.phi(X[7]))
        expected=np.einsum('nij,nj->ni',As,e)+np.einsum('nij,nj->ni',Bs,m.phi(X[:7])-m.phi(X[7]))-m.ALPHA*(m.full_laplacian()[:7,:7]@e)@m.H.T+w
        self.assertTrue(np.allclose(dX[:7]-dX[7],expected))


if __name__ == '__main__': unittest.main()
