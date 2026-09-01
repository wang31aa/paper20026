#!/usr/bin/env python3
"""Frozen V46 heterogeneous vehicle paired counterfactual computation."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "public_cluster_cases/cache/adas_two_vehicle_sample.csv"
CONTRACT = HERE / "V46_FROZEN_CONTRACT.json"
OUT = HERE / "results"
DT = 0.1
N = 4
LENGTH = np.array([4.5, 5.0, 4.7, 5.2])
TAU0, DRAG0, ACC0, BRAKE0 = 0.5, 0.025, 2.5, 4.2
HEADWAY, STANDSTILL, CLEARANCE, RESPONSE = 1.5, 5.0, 2.0, 0.5
POLICIES = (
    "all_coupled", "global_gain_reduction", "independent_tracking",
    "residual_gate", "connectivity_gate", "one_step_barrier_filter",
    "finite_grid_mpc", "two_layer",
)


def load_source() -> dict[str, np.ndarray]:
    with SOURCE.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["id"] == "6"]
    if len(rows) < 100:
        raise RuntimeError("record id 6 is absent or too short")
    keys = ("time", "speed_av", "acc_av", "acc_sv1", "acc_sv2",
            "distance_av_headway")
    return {k: np.array([float(r[k]) for r in rows], dtype=float) for k in keys}


def margin(gap: float, vf: float, vl: float, lf: float, ll: float) -> float:
    bumper = gap - 0.5 * (lf + ll)
    required = CLEARANCE + RESPONSE * vf + max((vf * vf - vl * vl) / (2 * 2.73), 0.0)
    return bumper - required


def parameters(seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    # Latin-like stratification prevents accidental near-homogeneous draws.
    q = (np.arange(N) + rng.random(N)) / N
    rng.shuffle(q)
    lag = TAU0 * (0.7 + 1.1 * q)
    rng.shuffle(q)
    drag = DRAG0 * (0.6 + 1.2 * q)
    rng.shuffle(q)
    authority = 0.65 + 0.40 * q
    return lag, drag, ACC0 * authority, BRAKE0 * authority


def candidate_mpc(v: np.ndarray, a: np.ndarray, x: np.ndarray, i: int,
                  target: float, umax: float, bmax: float, tau: float,
                  drag: float) -> float:
    """Finite-grid receding-horizon control under the same actuator limits."""
    best = (-bmax, float("inf"))
    for u in np.linspace(-bmax, umax, 9):
        xx, vv, aa = x.copy(), v.copy(), a.copy()
        feasible, cost = True, 0.0
        for _ in range(4):
            aa[i] += DT * ((np.clip(u, -bmax, umax) - aa[i]) / tau - drag * vv[i])
            vv[i] = max(0.0, vv[i] + DT * aa[i])
            xx[i] += DT * vv[i]
            m = margin(xx[i-1] - xx[i], vv[i], vv[i-1], LENGTH[i], LENGTH[i-1])
            feasible &= m >= 0.0
            cost += (vv[i] - target) ** 2 + 0.04 * u * u + 50.0 * max(-m, 0.0) ** 2
        if feasible and cost < best[1]:
            best = (float(u), float(cost))
    return best[0]


def simulate(source: dict[str, np.ndarray], stress: float, delay: float,
             rho: float, seed: int, policy: str, high: float, low: float,
             initial_gap: float) -> dict[str, float | int | str]:
    split = int(0.6 * len(source["time"]))
    target = source["speed_av"][split:split + 200]
    recorded = np.column_stack((source["acc_av"], source["acc_sv1"], source["acc_sv2"]))
    mismatch = recorded - np.median(recorded, axis=1, keepdims=True)
    mismatch = mismatch[split:split + len(target)]
    rng = np.random.default_rng(seed + int(100 * stress) + int(10 * delay))
    lag, drag, accel_max, brake_max = parameters(seed)
    v = np.full(N, target[0]) + rng.normal(0, 0.25, N)
    a = np.zeros(N)
    x = -np.arange(N, dtype=float) * initial_gap
    z = np.full(N, target[0])
    active = np.ones(N - 1, dtype=bool)
    delay_steps = int(round(delay / DT))
    zhist = [z.copy() for _ in range(delay_steps + 1)]
    margins, errors, controls, observers = [], [], [], []
    comm, saturated = 0, 0
    bad_run = good_after_bad = 0
    first_failure = len(target) * DT
    recovery_dwell = 0.0
    failed = False
    mpc_hold = np.zeros(N)

    for k, ref in enumerate(target):
        z[0] = ref
        delayed_z = zhist[max(0, len(zhist) - 1 - delay_steps)]
        # Information ancestry is a fixed chain; gates change physical influence,
        # while connectivity_gate/two_layer preserve the observer backbone.
        for i in range(1, N):
            received = rng.random() < rho
            if received:
                z[i] += DT * 2.2 * rho * (delayed_z[i-1] - z[i])
                comm += 1
        forcing = np.r_[mismatch[k, 0], mismatch[k],][0:N]
        req = np.zeros(N)
        req[0] = np.clip((ref - v[0]) * 1.8, -brake_max[0], accel_max[0])
        for i in range(1, N):
            gap = x[i-1] - x[i]
            desired = STANDSTILL + HEADWAY * v[i]
            residual = abs(v[i-1] - v[i]) + 0.35 * abs(z[i-1] - z[i]) + stress * abs(forcing[i])
            if policy in ("residual_gate", "connectivity_gate", "two_layer"):
                if active[i-1] and residual > high:
                    active[i-1] = False
                elif (not active[i-1]) and residual < low:
                    active[i-1] = True
            coupled = policy not in ("independent_tracking",) and active[i-1]
            gain = 0.42 if policy == "global_gain_reduction" else 1.0
            physical_term = (0.22 * (gap - desired) + 0.72 * (v[i-1] - v[i])) if coupled else 0.0
            req[i] = gain * physical_term + 0.65 * (z[i] - v[i]) + stress * forcing[i]
            local_m = margin(gap, v[i], v[i-1], LENGTH[i], LENGTH[i-1])
            if policy in ("one_step_barrier_filter", "two_layer") and local_m < 3.0:
                req[i] = min(req[i], -brake_max[i] * min(1.0, (3.0 - local_m) / 3.0))
            if policy == "finite_grid_mpc":
                if k % 10 == 0:
                    mpc_hold[i] = candidate_mpc(v, a, x, i, z[i], accel_max[i], brake_max[i], lag[i], drag[i])
                req[i] = mpc_hold[i]
            if policy == "connectivity_gate" and not active[i-1]:
                # Preserve information backbone but remove physical predecessor injection.
                req[i] = 0.65 * (z[i] - v[i])
        applied = np.clip(req, -brake_max, accel_max)
        saturated += int(np.count_nonzero(abs(applied - req) > 1e-10))
        a += DT * ((applied - a) / lag - drag * v)
        v = np.maximum(0.0, v + DT * a)
        x += DT * v
        zhist.append(z.copy())
        current = min(margin(x[i-1] - x[i], v[i], v[i-1], LENGTH[i], LENGTH[i-1]) for i in range(1, N))
        margins.append(current)
        errors.append(float(np.sqrt(np.mean((v[1:] - ref) ** 2))))
        controls.append(float(np.sum(applied * applied) * DT))
        observers.append(float(np.sqrt(np.mean((z[1:] - ref) ** 2))))
        if current < 0:
            bad_run += 1
            good_after_bad = 0
        else:
            if bad_run >= 5 and not failed:
                failed = True
                first_failure = (k - bad_run + 1) * DT
            if failed:
                good_after_bad += 1
                recovery_dwell = max(recovery_dwell, good_after_bad * DT)
            bad_run = 0
    if bad_run >= 5 and not failed:
        failed = True
        first_failure = (len(target) - bad_run) * DT
    tail = max(1, len(errors) // 5)
    return {
        "seed": seed, "stress": stress, "delay_s": delay, "rho": rho, "policy": policy,
        "task_success": int(not failed), "first_failure_s": first_failure,
        "minimum_margin_m": float(min(margins)),
        "failure_dwell_s": float(sum(m < 0 for m in margins) * DT),
        "recovery_dwell_s": recovery_dwell,
        "tail_tracking_rmse_mps": float(np.mean(errors[-tail:])),
        "control_energy": float(sum(controls)), "communication_messages": comm,
        "saturation_fraction": saturated / (N * len(target)),
        "observer_rmse_mps": float(np.mean(observers)),
        "lag_spread": float(max(lag) / min(lag)), "drag_spread": float(max(drag) / min(drag)),
        "authority_spread": float(max(accel_max) / min(accel_max)),
    }


def main() -> None:
    spec = json.loads(CONTRACT.read_text())
    source = load_source()
    split = int(spec["calibration_fraction"] * len(source["time"]))
    cal = np.column_stack((source["acc_av"][:split], source["acc_sv1"][:split], source["acc_sv2"][:split]))
    disagreement = np.abs(cal - np.median(cal, axis=1, keepdims=True))
    high, low = float(np.quantile(disagreement, .9)), float(np.quantile(disagreement, .6))
    initial_gap = max(25.0, 0.5 * float(np.quantile(source["distance_av_headway"][:split], .1)))
    grid = spec["grid"]
    rows = [simulate(source, s, d, r, seed, p, high, low, initial_gap)
            for seed in grid["seeds"] for s in grid["stress"] for d in grid["delay_s"]
            for r in grid["participation"] for p in POLICIES]
    OUT.mkdir(exist_ok=True)
    with (OUT / "V46_PAIRED_RESULTS.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summary = {
        "contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "rows": len(rows), "paired_conditions": len(rows) // len(POLICIES),
        "source_rows": len(source["time"]), "calibration_rows": split,
        "heldout_rows": len(source["time"]) - split, "gate_high": high, "gate_low": low,
        "initial_gap_m": initial_gap,
        "success_by_policy": {p: int(sum(r["task_success"] for r in rows if r["policy"] == p)) for p in POLICIES},
        "claim_boundary": spec["claim_boundary"],
    }
    (OUT / "V46_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
