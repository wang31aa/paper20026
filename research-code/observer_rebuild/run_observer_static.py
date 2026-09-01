#!/usr/bin/env python3
"""Equation-first distributed target observer and common-H static controller.

No historical result, certificate, gain search, clipping, or hidden transient is
used.  Rows of L are receivers: L[i,j] < 0 means j -> i.  Node 7 is the leader.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np

N, D = 7, 3
ALPHA = 16.1
GAMMA_MODEL = 6.0
GAMMA_STATE = 60.0
H = np.diag([12.0, 10.0, 11.0])
DT, T_END, SAVE_DT = 0.001, 4.0, 0.01
SEEDS = tuple(range(20))


def chua_matrices(one_indexed: int) -> tuple[np.ndarray, np.ndarray]:
    a = 9.0 - 0.1 * (one_indexed - 1)
    ell = 14.286 - 0.1 * (one_indexed - 1)
    A = np.array([[-a * 2 / 7, a, 0], [1, -1, 1],
                  [0, -ell, .0005 + .0001 * one_indexed]], float)
    B = np.diag([a * 3 / 7, 0, 0])
    return A, B


def phi(x: np.ndarray) -> np.ndarray:
    y = np.zeros_like(x)
    y[..., 0] = .5 * (np.abs(x[..., 0] + 1) - np.abs(x[..., 0] - 1))
    return y


def full_laplacian() -> np.ndarray:
    """Published graph under the receiver-row convention, leader index 7."""
    return np.array([
        [1, 0, 0, 0, 0, 0, 0, -1],
        [-2, 2, 0, 0, 0, 0, 0, 0],
        [0, -2, 2, 0, 0, 0, 0, 0],
        [-2, 0, -1, 3, 0, 0, 0, 0],
        [0, 0, -3, 0, 3, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, -1],
        [0, 0, 0, 0, 0, 0, 1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0]], float)


def pack(X, Ah, Bh, sh):
    return np.concatenate([X.ravel(), Ah.ravel(), Bh.ravel(), sh.ravel()])


def unpack(y):
    p = 0
    X = y[p:p + 8 * D].reshape(8, D); p += 8 * D
    Ah = y[p:p + N * D * D].reshape(N, D, D); p += N * D * D
    Bh = y[p:p + N * D * D].reshape(N, D, D); p += N * D * D
    sh = y[p:p + N * D].reshape(N, D)
    return X, Ah, Bh, sh


def rhs(y: np.ndarray) -> np.ndarray:
    X, Ah, Bh, sh = unpack(y)
    L = full_laplacian(); L1 = L[:N, :N]
    As, Bs = zip(*(chua_matrices(i) for i in range(1, 8)))
    A0, B0 = chua_matrices(8)
    As, Bs = np.stack(As), np.stack(Bs)

    # Plant/controller: exactly equation C1 in EQUATION_CODE_MAP.md.
    intrinsic = np.einsum('nij,nj->ni', As, X[:N]) + np.einsum(
        'nij,nj->ni', Bs, phi(X[:N]))
    relative = L[:N] @ X
    dX = np.empty_like(X)
    dX[:N] = intrinsic - ALPHA * (relative @ H.T)
    dX[N] = A0 @ X[N] + B0 @ phi(X[N])

    # Model observers: exact linear leader-following consensus (O1--O2).
    A_aug = np.concatenate([Ah, A0[None]], axis=0)
    B_aug = np.concatenate([Bh, B0[None]], axis=0)
    dAh = -GAMMA_MODEL * np.einsum('ij,jkl->ikl', L[:N], A_aug)
    dBh = -GAMMA_MODEL * np.einsum('ij,jkl->ikl', L[:N], B_aug)

    # Target-state observer O3 uses the reconstructed TARGET vector field.
    s_aug = np.vstack([sh, X[N]])
    internal = np.einsum('nij,nj->ni', Ah, sh) + np.einsum(
        'nij,nj->ni', Bh, phi(sh))
    dsh = internal - GAMMA_STATE * (L[:N] @ s_aug)
    return pack(dX, dAh, dBh, dsh)


def rk4(y, dt):
    k1 = rhs(y); k2 = rhs(y + .5 * dt * k1)
    k3 = rhs(y + .5 * dt * k2); k4 = rhs(y + dt * k3)
    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


@dataclass
class Run:
    seed: int; finite: bool; samples: int
    initial_observer_state_error: float; final_observer_state_error: float
    max_observer_state_error: float; tail20_max_observer_state_error: float
    initial_model_error: float; final_model_error: float
    max_model_error: float; tail20_max_model_error: float
    initial_mismatch_estimation_error: float; final_mismatch_estimation_error: float
    max_mismatch_estimation_error: float; tail20_max_mismatch_estimation_error: float
    initial_tracking_error: float; final_tracking_error: float
    max_tracking_error: float; tail20_max_tracking_error: float


def metrics(y):
    X, Ah, Bh, sh = unpack(y); A0, B0 = chua_matrices(8)
    As, Bs = zip(*(chua_matrices(i) for i in range(1, 8)))
    As, Bs = np.stack(As), np.stack(Bs)
    obs = np.linalg.norm((sh - X[7]).ravel())
    model = np.sqrt(np.sum((Ah - A0) ** 2) + np.sum((Bh - B0) ** 2))
    what = np.einsum('nij,nj->ni', As - Ah, sh) + np.einsum(
        'nij,nj->ni', Bs - Bh, phi(sh))
    wtrue = np.einsum('nij,j->ni', As - A0, X[7]) + np.einsum(
        'nij,j->ni', Bs - B0, phi(X[7]))
    mismatch = np.linalg.norm((what - wtrue).ravel())
    tracking = np.linalg.norm((X[:N] - X[7]).ravel())
    return np.array([obs, model, mismatch, tracking])


def simulate(seed: int, dt: float = DT, save_dt: float = SAVE_DT):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-.2, .2, (8, D))
    A0, B0 = chua_matrices(8)
    Ah = A0 + rng.normal(0, .8, (N, D, D))
    Bh = B0 + rng.normal(0, .3, (N, D, D))
    sh = rng.uniform(-1, 1, (N, D))
    y = pack(X, Ah, Bh, sh)
    steps = round(T_END / dt); stride = round(save_dt / dt)
    ts, ys, ms = [], [], []
    finite = True
    for k in range(steps + 1):
        if k % stride == 0:
            ts.append(k * dt); ys.append(y.copy()); ms.append(metrics(y))
        if k < steps:
            y = rk4(y, dt)
            if not np.all(np.isfinite(y)):
                finite = False; break
    ms = np.stack(ms); tail = max(0, int(.8 * len(ms)))
    vals = []
    for j in range(4): vals.extend([ms[0,j], ms[-1,j], ms[:,j].max(), ms[tail:,j].max()])
    row = Run(seed, finite, len(ms), *map(float, vals))
    return row, np.array(ts), np.stack(ys), ms


def graph_audit():
    L = full_laplacian(); L1 = L[:N, :N]
    q = np.linalg.solve(L1.T, np.ones(N)); G = np.diag(q)
    S = G @ L1 + L1.T @ G
    mu = np.linalg.eigvalsh(np.diag(q ** -.5) @ S @ np.diag(q ** -.5)).min()
    wrong_theta = np.linalg.solve(L1.T, np.ones(N))
    wrong_S = np.diag(1 / wrong_theta) @ L1 + L1.T @ np.diag(1 / wrong_theta)
    return dict(convention='row i receives from column j; L[i,j]<0 means j->i',
                leader_index_zero_based=7, row_sum_max_abs=float(abs(L.sum(1)).max()),
                q_L1_transpose_inverse_ones=q.tolist(), G_diagonal=q.tolist(),
                S_min_eigenvalue=float(np.linalg.eigvalsh(S).min()),
                generalized_margin_mu=float(mu),
                regression_wrong_inverse_weight_min_eigenvalue=float(np.linalg.eigvalsh(wrong_S).min()))


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rows, arrays = [], {}
    for seed in SEEDS:
        row, t, y, m = simulate(seed); rows.append(row)
        arrays[f'seed_{seed}_time'] = t; arrays[f'seed_{seed}_state'] = y
        arrays[f'seed_{seed}_metrics'] = m
    with (out / 'raw_runs.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]))); w.writeheader()
        w.writerows(asdict(r) for r in rows)
    np.savez_compressed(out / 'raw_trajectories.npz', **arrays)
    # Declared step-size check, with identical seeded initial conditions.
    conv=[]
    for dt in (.002, .001, .0005):
        for seed in range(5):
            row, _, _, _ = simulate(seed, dt=dt, save_dt=max(.01,dt))
            item=asdict(row); item['dt']=dt; conv.append(item)
    with (out/'dt_convergence.csv').open('w',newline='') as f:
        fields=['dt']+list(asdict(rows[0])); w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader(); w.writerows(conv)
    audit = graph_audit()
    metadata = dict(equations=['O1','O2','O3','C1'], seeds=list(SEEDS), dt=DT,
        t_end=T_END, save_dt=SAVE_DT, alpha=ALPHA, gamma_model=GAMMA_MODEL,
        gamma_state=GAMMA_STATE, common_H=H.tolist(), integrator='fixed-step RK4',
        gain_search=False, clipping=False, hidden_burn_in=False,
        tail_metric='last 20%, declared and reported in addition to full horizon',
        graph=audit, dt_convergence=dict(steps=[.002,.001,.0005],seeds=list(range(5)),
          criterion='compare endpoint metrics for identical seeds; no run discarded'))
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'n_runs':len(rows),'finite':sum(r.finite for r in rows),
        'max_final_observer_error':max(r.final_observer_state_error for r in rows),
        'max_final_model_error':max(r.final_model_error for r in rows),
        'max_final_mismatch_error':max(r.final_mismatch_estimation_error for r in rows),
        'graph_mu':audit['generalized_margin_mu']}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path,
        default=Path(__file__).resolve().parent / 'results')
    main(p.parse_args().out)
