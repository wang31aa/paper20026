#!/usr/bin/env python3
"""Preregistered observer-in-loop benchmark with an analytic comparison bound.

Target-field parameters are common protocol knowledge.  Only the target state
is estimated.  Follower--follower controller terms use measured follower
states, while every pinning term uses the local estimate, never the true target.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

N, D = 5, 3
A0, B0 = 2.0, 0.6
ALPHA, GAMMA_STATE = 4.0, 35.0
DT, T_END, SAVE_DT = 0.002, 4.0, 0.02
SEEDS = tuple(range(10))
LEVELS = (0.05, 0.15, 0.30)
R = B0 / (A0 - B0)
# A deliberately conservative rate below every proved observer decay rate.
# This avoids using near-machine-zero values as a numerical coverage target.
OBSERVER_ENVELOPE_RATE_CAP = 1.0


def forcing(t: float) -> np.ndarray:
    return np.array([np.sin(t), np.cos(np.sqrt(2) * t),
                     np.sin(np.sqrt(3) * t)]) / np.sqrt(3)


def topologies() -> dict[str, np.ndarray]:
    """Return W with W[i,j]>0 meaning that receiver i observes source j."""
    out = {}
    W = np.zeros((N, N + 1)); W[0, N] = 1; W[1, 0] = 1; W[2, 1] = 1
    W[3, 2] = 1; W[4, 3] = 1; out['chain'] = W
    W = np.zeros((N, N + 1)); W[:, N] = 1; out['star'] = W
    W = np.zeros((N, N + 1)); W[0, N] = 1; W[1, 0] = 1; W[2, 0] = 1
    W[3, 1] = 1; W[3, 2] = .5; W[4, 2] = 1; out['branch'] = W
    W = np.zeros((N, N + 1)); W[0, N] = 1; W[1, 0] = 1; W[2, 1] = 1
    W[3, 2] = 1; W[4, 3] = 1; W[0, 4] = .25; W[2, 0] = .4
    out['cyclic'] = W
    return out


def laplacian(W: np.ndarray) -> np.ndarray:
    L = np.zeros((N + 1, N + 1)); L[:N, :] = -W
    L[np.arange(N), np.arange(N)] += W.sum(1)
    return L


def graph_constants(W: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    L = laplacian(W); L1 = L[:N, :N]
    g = np.linalg.solve(L1.T, np.ones(N)); G = np.diag(g)
    S = G @ L1 + L1.T @ G
    mu = np.linalg.eigvalsh(np.diag(g ** -.5) @ S @
                            np.diag(g ** -.5)).min()
    return L, g, float(mu), float(np.linalg.eigvalsh(S).min())


def node_parameters(level: float) -> tuple[np.ndarray, np.ndarray]:
    pattern = np.array([-1., -.5, 0., .5, 1.])
    return (A0 * (1 + .45 * level * pattern),
            B0 * (1 + .70 * level * pattern[::-1]))


def certificate(W: np.ndarray, level: float) -> dict:
    L, g, mu, smin = graph_constants(W); a, b = node_parameters(level)
    mismatch = np.abs(a - A0) * R + np.abs(b - B0) * (R + 1.)
    omega = float(np.sqrt(np.sum(g * mismatch ** 2)))
    r = float(np.max(-2 * (a - b)))
    d = ALPHA * mu - r
    tracking_rate = d / 2
    observer_rate_proved = A0 - B0 + GAMMA_STATE * mu / 2
    observer_rate_used = min(OBSERVER_ENVELOPE_RATE_CAP,
                             observer_rate_proved)
    delta = omega / tracking_rate / np.sqrt(np.min(g))
    follower_W = W[:, :N]
    follower_L = np.diag(follower_W.sum(axis=1)) - follower_W
    return dict(L=L, follower_L=follower_L, g=g, mu=mu, S_min=smin, a=a, b=b,
                pin=W[:, N].copy(), mismatch=mismatch, omega=omega, r=r,
                d=d, tracking_rate=tracking_rate,
                observer_rate_proved=observer_rate_proved,
                observer_rate_used=observer_rate_used, delta=float(delta))


def controller_values(Xf: np.ndarray, sh: np.ndarray, L1: np.ndarray,
                      pin: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Observer-loop and same-state oracle-pinning controls."""
    u_observer = -ALPHA * (L1 @ Xf - pin[:, None] * sh)
    u_oracle = -ALPHA * (L1 @ Xf - pin[:, None] * target)
    return u_observer, u_oracle


def policy_control(Xf: np.ndarray, target: np.ndarray, c: dict,
                   policy: str, sh: np.ndarray | None = None) -> np.ndarray:
    """Applied control for one of the three frozen matched policies."""
    if policy == 'observer_loop':
        if sh is None: raise ValueError('observer_loop requires sh')
        return -ALPHA * (c['L'][:N, :N] @ Xf - c['pin'][:, None] * sh)
    if policy == 'oracle_target':
        return -ALPHA * (c['L'][:N, :N] @ Xf - c['pin'][:, None] * target)
    if policy == 'no_target':
        return -ALPHA * (c['follower_L'] @ Xf)
    raise ValueError(f'unknown policy {policy!r}')


def initial_conditions(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the frozen matched plant and observer initial states."""
    rng = np.random.default_rng(seed)
    s0 = rng.normal(size=D); s0 *= rng.uniform(0, .9 * R) / np.linalg.norm(s0)
    X = np.vstack([rng.uniform(-1, 1, (N, D)), s0])
    sh = rng.uniform(-.5, .5, (N, D))
    return X, sh


def pack(X: np.ndarray, sh: np.ndarray) -> np.ndarray:
    return np.concatenate([X.ravel(), sh.ravel()])


def unpack(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = (N + 1) * D
    return y[:p].reshape(N + 1, D), y[p:].reshape(N, D)


def rhs(t: float, y: np.ndarray, c: dict) -> np.ndarray:
    X, sh = unpack(y); q = forcing(t); L1 = c['L'][:N, :N]
    f = np.empty_like(X)
    f[:N] = -c['a'][:, None] * X[:N] + c['b'][:, None] * (
        np.tanh(X[:N]) + q)
    f[N] = -A0 * X[N] + B0 * (np.tanh(X[N]) + q)
    u_observer, _ = controller_values(X[:N], sh, L1, c['pin'], X[N])
    f[:N] += u_observer
    s_aug = np.vstack([sh, X[N]])
    dsh = -A0 * sh + B0 * (np.tanh(sh) + q) - \
        GAMMA_STATE * (c['L'][:N] @ s_aug)
    return pack(f, dsh)


def rk4(t: float, y: np.ndarray, dt: float, c: dict) -> np.ndarray:
    k1 = rhs(t, y, c); k2 = rhs(t + dt / 2, y + dt * k1 / 2, c)
    k3 = rhs(t + dt / 2, y + dt * k2 / 2, c)
    k4 = rhs(t + dt, y + dt * k3, c)
    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def plant_rhs(t: float, y: np.ndarray, c: dict, policy: str) -> np.ndarray:
    X = y.reshape(N + 1, D); q = forcing(t); f = np.empty_like(X)
    f[:N] = -c['a'][:, None] * X[:N] + c['b'][:, None] * (
        np.tanh(X[:N]) + q)
    f[N] = -A0 * X[N] + B0 * (np.tanh(X[N]) + q)
    f[:N] += policy_control(X[:N], X[N], c, policy)
    return f.ravel()


def plant_rk4(t: float, y: np.ndarray, dt: float, c: dict,
              policy: str) -> np.ndarray:
    k1 = plant_rhs(t, y, c, policy)
    k2 = plant_rhs(t + dt / 2, y + dt * k1 / 2, c, policy)
    k3 = plant_rhs(t + dt / 2, y + dt * k2 / 2, c, policy)
    k4 = plant_rhs(t + dt, y + dt * k3, c, policy)
    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def comparison_components(t: np.ndarray, z0: float, h0: float,
                          c: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = c['tracking_rate']; kappa = c['observer_rate_used']
    homogeneous = np.exp(-a * t) * z0
    mismatch = c['omega'] / a * (1 - np.exp(-a * t))
    amplitude = ALPHA * float(np.max(c['pin'])) * h0
    if abs(a - kappa) < 1e-12:
        observer = amplitude * t * np.exp(-a * t)
    else:
        observer = amplitude * (np.exp(-kappa * t) - np.exp(-a * t)) / (a - kappa)
    return homogeneous, mismatch, observer, homogeneous + mismatch + observer


def diagnostics(times: np.ndarray, plants: np.ndarray, observers: np.ndarray,
                c: dict) -> dict[str, np.ndarray]:
    e = plants[:, :N] - plants[:, N, None, :]
    eta = observers - plants[:, N, None, :]
    z = np.sqrt(np.sum(c['g'][None, :, None] * e ** 2, axis=(1, 2)))
    h = np.sqrt(np.sum(c['g'][None, :, None] * eta ** 2, axis=(1, 2)))
    hom, mismatch, obs, envelope = comparison_components(times, z[0], h[0], c)
    observer_envelope = h[0] * np.exp(-c['observer_rate_used'] * times)
    node_env = envelope[:, None] / np.sqrt(c['g'])[None, :]
    node_error = np.linalg.norm(e, axis=2)
    return dict(
        tracking=np.linalg.norm(e.reshape(len(times), -1), axis=1),
        tracking_weighted=z, observer=np.linalg.norm(
            eta.reshape(len(times), -1), axis=1),
        observer_weighted=h, homogeneous_component=hom,
        mismatch_component=mismatch, observer_component=obs,
        tracking_envelope=envelope, observer_envelope=observer_envelope,
        tracking_envelope_ratio=z / np.maximum(envelope, 1e-300),
        observer_envelope_ratio=h / np.maximum(observer_envelope, 1e-300),
        max_node_ratio=np.max(node_error / np.maximum(node_env, 1e-300), axis=1),
        target_radius=np.linalg.norm(plants[:, N], axis=1))


@dataclass
class Run:
    topology: str; heterogeneity: float; seed: int; dt: float; finite: bool; samples: int
    delta: float; initial_tracking: float; max_tracking: float
    tail20_max_tracking: float; final_tracking: float; tail20_delta_ratio: float
    initial_observer: float; max_observer: float; tail20_max_observer: float
    final_observer: float; max_tracking_envelope_ratio: float
    max_observer_envelope_ratio: float; max_node_ratio: float
    target_radius_max: float; max_control_norm: float
    max_same_state_oracle_control_norm: float
    max_same_state_control_gap: float; final_same_state_control_gap: float


@dataclass
class PolicyRun:
    topology: str; heterogeneity: float; seed: int; policy: str; dt: float
    finite: bool; samples: int; initial_tracking: float; max_tracking: float
    tail20_max_tracking: float; final_tracking: float; target_radius_max: float
    max_control_norm: float


def simulate(name: str, W: np.ndarray, level: float, seed: int,
             dt: float = DT, t_end: float = T_END, store: bool = False):
    c = certificate(W, level); X, sh = initial_conditions(seed); y = pack(X, sh)
    steps = round(t_end / dt); stride = max(1, round(SAVE_DT / dt))
    times, plants, observers, controls, oracle_controls = [], [], [], [], []
    finite = True
    for step in range(steps + 1):
        t = step * dt
        if step % stride == 0 or step == steps:
            xx, ss = unpack(y); u, uo = controller_values(
                xx[:N], ss, c['L'][:N, :N], c['pin'], xx[N])
            times.append(t); plants.append(xx.copy()); observers.append(ss.copy())
            controls.append(u); oracle_controls.append(uo)
        if step < steps:
            y = rk4(t, y, dt, c)
            if not np.all(np.isfinite(y)):
                finite = False; break
    times = np.asarray(times); plants = np.asarray(plants)
    observers = np.asarray(observers); controls = np.asarray(controls)
    oracle_controls = np.asarray(oracle_controls)
    diag = diagnostics(times, plants, observers, c)
    gap = np.linalg.norm((controls - oracle_controls).reshape(len(times), -1), axis=1)
    control_norm = np.linalg.norm(controls.reshape(len(times), -1), axis=1)
    oracle_norm = np.linalg.norm(oracle_controls.reshape(len(times), -1), axis=1)
    tail = times >= .8 * t_end
    row = Run(name, level, seed, dt, finite, len(times), c['delta'],
              float(diag['tracking'][0]), float(diag['tracking'].max()),
              float(diag['tracking'][tail].max()), float(diag['tracking'][-1]),
              float(diag['tracking'][tail].max() / c['delta']),
              float(diag['observer'][0]), float(diag['observer'].max()),
              float(diag['observer'][tail].max()), float(diag['observer'][-1]),
              float(diag['tracking_envelope_ratio'].max()),
              float(diag['observer_envelope_ratio'].max()),
              float(diag['max_node_ratio'].max()),
              float(diag['target_radius'].max()), float(control_norm.max()),
              float(oracle_norm.max()), float(gap.max()), float(gap[-1]))
    raw = None
    if store:
        raw = dict(time=times, plant=plants, observer=observers,
                   control=controls, same_state_oracle_control=oracle_controls,
                   tracking_weighted=diag['tracking_weighted'],
                   observer_weighted=diag['observer_weighted'],
                   homogeneous_component=diag['homogeneous_component'],
                   mismatch_component=diag['mismatch_component'],
                   observer_component=diag['observer_component'],
                   tracking_envelope=diag['tracking_envelope'],
                   observer_envelope=diag['observer_envelope'],
                   tracking_envelope_ratio=diag['tracking_envelope_ratio'],
                   observer_envelope_ratio=diag['observer_envelope_ratio'])
    return row, raw


def policy_row(name: str, level: float, seed: int, policy: str, dt: float,
               finite: bool, times: np.ndarray, plants: np.ndarray,
               controls: np.ndarray) -> PolicyRun:
    e = plants[:, :N] - plants[:, N, None, :]
    tracking = np.linalg.norm(e.reshape(len(times), -1), axis=1)
    target_radius = np.linalg.norm(plants[:, N], axis=1)
    control_norm = np.linalg.norm(controls.reshape(len(times), -1), axis=1)
    tail = times >= .8 * T_END
    return PolicyRun(name, level, seed, policy, dt, finite, len(times),
        float(tracking[0]), float(tracking.max()), float(tracking[tail].max()),
        float(tracking[-1]), float(target_radius.max()), float(control_norm.max()))


def simulate_plant_policy(name: str, W: np.ndarray, level: float, seed: int,
                          policy: str, dt: float = DT,
                          t_end: float = T_END, store: bool = False):
    if policy not in ('oracle_target', 'no_target'):
        raise ValueError('plant-only comparator must be oracle_target or no_target')
    c = certificate(W, level); X, _ = initial_conditions(seed); y = X.ravel()
    steps = round(t_end / dt); stride = max(1, round(SAVE_DT / dt))
    times, plants, controls = [], [], []; finite = True
    for step in range(steps + 1):
        t = step * dt
        if step % stride == 0 or step == steps:
            xx = y.reshape(N + 1, D)
            u = policy_control(xx[:N], xx[N], c, policy)
            times.append(t); plants.append(xx.copy()); controls.append(u)
        if step < steps:
            y = plant_rk4(t, y, dt, c, policy)
            if not np.all(np.isfinite(y)):
                finite = False; break
    times = np.asarray(times); plants = np.asarray(plants); controls = np.asarray(controls)
    row = policy_row(name, level, seed, policy, dt, finite, times, plants, controls)
    return row, (dict(time=times, plant=plants, control=controls) if store else None)


def write_csv(path: Path, rows: list) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0])))
        writer.writeheader(); writer.writerows(asdict(row) for row in rows)


def generate(out: Path, freeze_path: Path) -> None:
    rows = []; arrays = {}
    policy_rows = []; policy_arrays = {}
    for name, W in topologies().items():
        for level in LEVELS:
            for seed in SEEDS:
                row, raw = simulate(name, W, level, seed, store=True); rows.append(row)
                run_id = f'{name}_h{level:.2f}_seed{seed}'
                for key, value in raw.items(): arrays[f'{run_id}__{key}'] = value
                policy_rows.append(policy_row(name, level, seed, 'observer_loop', DT,
                    row.finite, raw['time'], raw['plant'], raw['control']))
                for key in ('time', 'plant', 'control'):
                    policy_arrays[f'{run_id}__observer_loop__{key}'] = raw[key]
                for policy in ('oracle_target', 'no_target'):
                    prow, praw = simulate_plant_policy(name, W, level, seed,
                        policy, store=True); policy_rows.append(prow)
                    for key, value in praw.items():
                        policy_arrays[f'{run_id}__{policy}__{key}'] = value
    write_csv(out / 'raw_runs.csv', rows)
    np.savez_compressed(out / 'raw_trajectories.npz', **arrays)
    write_csv(out / 'matched_policy_runs.csv', policy_rows)
    np.savez_compressed(out / 'matched_policy_trajectories.npz', **policy_arrays)
    dt_rows = []
    policy_dt_rows = []
    for dt in (.004, .002, .001):
        for name, W in topologies().items():
            for seed in range(2):
                orow, oraw = simulate(name, W, .30, seed, dt=dt, store=True)
                dt_rows.append(orow)
                policy_dt_rows.append(policy_row(name, .30, seed, 'observer_loop', dt,
                    orow.finite, oraw['time'], oraw['plant'], oraw['control']))
                for policy in ('oracle_target', 'no_target'):
                    policy_dt_rows.append(simulate_plant_policy(
                        name, W, .30, seed, policy, dt=dt)[0])
    write_csv(out / 'dt_audit.csv', dt_rows)
    write_csv(out / 'matched_policy_dt_audit.csv', policy_dt_rows)
    certs = {}
    for name, W in topologies().items():
        certs[name] = {}
        for level in LEVELS:
            c = certificate(W, level)
            certs[name][str(level)] = {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                                       for key, value in c.items()}
    metadata = dict(
        preregistered=dict(N=N, D=D, levels=LEVELS, seeds=SEEDS, dt=DT,
            t_end=T_END, save_dt=SAVE_DT, alpha=ALPHA,
            gamma_state=GAMMA_STATE, observer_envelope_rate_cap=OBSERVER_ENVELOPE_RATE_CAP,
            A0=A0, B0=B0, forcing_bound=1, R=R, gain_search=False,
            clipping=False, discarded_runs=0,
            matched_policies=['observer_loop', 'oracle_target', 'no_target'],
            primary_matched_cases=120, primary_policy_runs=360,
            dt_matched_cases=24, dt_policy_runs=72),
        protocol_id='CERT-OIL-R3',
        source_freeze_sha256=hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
        information_boundary=(
            'target-field parameters are protocol-known; controller pinning uses local '
            'state estimates only; true target enters only target dynamics and pinned observer edges'),
        certificates=certs)
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(dict(generated_observer_runs=len(rows),
        generated_policy_runs=len(policy_rows), generated_observer_dt_runs=len(dt_rows),
        generated_policy_dt_runs=len(policy_dt_rows)), indent=2))


def main(out: Path) -> None:
    root = Path(__file__).resolve().parent
    freeze_path = root / 'SOURCE_FREEZE_R3.json'
    if not freeze_path.is_file():
        raise FileNotFoundError('CERT-OIL-R3 source freeze is required before execution')
    out = out.resolve(); active_lock = Path(str(out) + '.active.lock')
    completed_lock = Path(str(out) + '.completed.lock')
    staging = Path(str(out) + '.staging')
    if out.exists() or staging.exists() or completed_lock.exists():
        raise FileExistsError('R3 target/staging/completion marker already exists; overwrite refused')
    started = datetime.now(timezone.utc).isoformat()
    lock_payload = json.dumps(dict(protocol_id='CERT-OIL-R3', pid=os.getpid(),
        started_utc=started, target=str(out)), sort_keys=True).encode() + b'\n'
    fd = os.open(active_lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, lock_payload)
    finally:
        os.close(fd)
    try:
        staging.mkdir(parents=True, exist_ok=False)
        generate(staging, freeze_path)
        completion = dict(protocol_id='CERT-OIL-R3', started_utc=started,
            completed_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
            source_freeze_sha256=hashlib.sha256(freeze_path.read_bytes()).hexdigest())
        (staging / 'RUN_COMPLETION.json').write_text(json.dumps(completion, indent=2) + '\n')
        if out.exists():
            raise FileExistsError('R3 target appeared during execution; atomic publish refused')
        staging.rename(out)
        active_lock.rename(completed_lock)
    except BaseException:
        # Preserve the exclusive lock and staging bytes for forensic inspection.
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path,
                        default=Path(__file__).resolve().parent / 'results_r3')
    main(parser.parse_args().out)
