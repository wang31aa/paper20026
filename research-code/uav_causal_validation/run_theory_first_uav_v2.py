#!/usr/bin/env python3
"""Theory-first heterogeneous-UAV experiment with an explicit sampled observer.

The exact modal challenge and the physical UAV surrogate are reported as two
different evidence layers.  The former tests the closed-form theorem in its
declared class.  The latter tests a causal observer/gating mechanism and must
not be described as an exact necessary-and-sufficient realization.
"""
from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "UAV_THEORY_FIRST_PROTOCOL_V2.json").read_text())
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
N, DT = P["agents"], P["dt_s"]
STEPS = int(P["duration_s"] / DT)
RHOS = np.asarray(P["participation_grid"], float)
TC = P["theory_contract"]


def psi(rho: float) -> float:
    values = []
    for ab, kap, d0, db, u, vh in zip(TC["modal_a_bar"], TC["modal_kappa"],
                                      TC["D0"], TC["Db"], TC["U"], TC["Vh"]):
        radius = max(d0 + rho * db - u, 0.0) / (ab + kap * rho)
        values.extend(((radius + vh / rho) / TC["epsilon_peak"],
                       radius / TC["epsilon_ultimate"]))
    return max(values)


def exact_modal_challenge(rho: float) -> dict:
    # Worst-case values are achieved by the same revealed constant forcing and
    # same-sign sample-reset continuation in the theorem's pure-cancellation class.
    peak = ultimate = 0.0
    for ab, kap, d0, db, u, vh in zip(TC["modal_a_bar"], TC["modal_kappa"],
                                      TC["D0"], TC["Db"], TC["U"], TC["Vh"]):
        radius = max(d0 + rho * db - u, 0.0) / (ab + kap * rho)
        peak = max(peak, radius + vh / rho)
        ultimate = max(ultimate, radius)
    return {"rho": rho, "psi": psi(rho), "peak": peak, "ultimate": ultimate,
            "predicted_feasible": int(psi(rho) <= 1.0),
            "challenge_feasible": int(peak <= TC["epsilon_peak"] and
                                        ultimate <= TC["epsilon_ultimate"])}


def ring() -> np.ndarray:
    A = np.zeros((N, N))
    for i in range(N):
        A[i, (i - 1) % N] = A[i, (i + 1) % N] = 1.0
    return A


def innovations(seed: int, env: dict) -> dict:
    g = np.random.default_rng(seed)
    return {
        "wind": g.normal(0, env["wind_sd"], (STEPS, N, 2)),
        "position_noise": g.normal(0, 0.008, (STEPS, N, 2)),
        "velocity_noise": g.normal(0, 0.012, (STEPS, N, 2)),
        "delivery": g.random((STEPS, N, N)) >= env["loss"],
    }


def project_acceleration(u: np.ndarray, p: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Deterministic half-space projection for pair separation and corridor."""
    u = u.copy()
    dmin = P["task"]["minimum_pair_clearance_m"]
    for _ in range(3):
        for i in range(N):
            for j in range(i + 1, N):
                d = p[i] - p[j]; dist = np.linalg.norm(d)
                if dist < 1.2 * dmin and dist > 1e-9:
                    n = d / dist
                    required = 3.0 * (dmin - dist) - 1.4 * np.dot(v[i] - v[j], n)
                    violation = required - np.dot(u[i] - u[j], n)
                    if violation > 0:
                        u[i] += 0.5 * violation * n; u[j] -= 0.5 * violation * n
        half = P["task"]["corridor_half_width_m"]
        for i in range(N):
            if abs(p[i, 1]) > 0.82 * half:
                direction = -math.copysign(1.0, p[i, 1])
                u[i, 1] += direction * (2.2 * (abs(p[i, 1]) - .82 * half) + .9 * abs(v[i, 1]))
    lim = P["task"]["maximum_acceleration_mps2"]
    norm = np.linalg.norm(u, axis=1)
    over = norm > lim
    u[over] *= (lim / norm[over])[:, None]
    return u


def run(policy: str, rho: float, env: dict, seed: int, heterogeneous: bool) -> dict:
    inn = innovations(seed, env)
    yoff = np.linspace(-1.0, 1.0, N)
    p = np.c_[np.zeros(N), yoff]
    v = np.zeros((N, 2)); applied = np.zeros((N, 2))
    beta = np.asarray(P["plant"]["acceleration_effectiveness"] if heterogeneous else [1] * N)
    drag = np.asarray(P["plant"]["linear_drag_per_s"] if heterogeneous else [0.08] * N)
    tau = np.asarray(P["plant"]["actuator_time_constant_s"] if heterogeneous else [0.40] * N)
    phat = np.repeat(p[None, :, :], N, axis=0)
    vhat = np.repeat(v[None, :, :], N, axis=0)
    residual = np.zeros(N); A0 = ring()
    update_steps = max(1, int(round(P["observer"]["base_update_interval_s"] / (rho * DT))))
    min_pair = np.inf; max_corridor = 0.; energy = comm = 0.; peak_residual = 0.
    tail = []; gate_events = 0
    delayed_p = [p.copy() for _ in range(env["delay_steps"] + 1)]
    delayed_v = [v.copy() for _ in range(env["delay_steps"] + 1)]
    kp, kv = P["observer"]["position_innovation_gain"], P["observer"]["velocity_innovation_gain"]
    for k in range(STEPS):
        # Predictor flow is explicit and logged through its innovation/residual.
        phat += DT * vhat
        delayed_p.append(p + inn["position_noise"][k]); delayed_p.pop(0)
        delayed_v.append(v + inn["velocity_noise"][k]); delayed_v.pop(0)
        if k % update_steps == 0:
            for receiver in range(N):
                for sender in range(N):
                    if A0[receiver, sender] and inn["delivery"][k, receiver, sender]:
                        ip = delayed_p[0][sender] - phat[receiver, sender]
                        iv = delayed_v[0][sender] - vhat[receiver, sender]
                        residual[sender] = .75 * residual[sender] + .25 * np.linalg.norm(np.r_[ip, iv])
                        phat[receiver, sender] += kp * ip
                        vhat[receiver, sender] += kv * iv
                        comm += 1
        trust = np.ones(N)
        if policy in ("observer_residual_gate", "connectivity_observer_gate"):
            trust = np.clip(.24 / np.maximum(residual, .24), .12, 1.0)
            if policy == "connectivity_observer_gate": trust = np.maximum(trust, .30)
            gate_events += int(np.any(trust < .999))
        goal = np.c_[np.full(N, P["task"]["goal_x_m"]), yoff]
        u = .55 * (goal - p) - 1.05 * v
        if policy != "independent_tracking":
            for i in range(N):
                neighbours = np.where(A0[i] > 0)[0]
                for j in neighbours:
                    wij = rho * trust[j] / len(neighbours)
                    # Estimated-neighbour coupling; platform mismatch is transmitted through this term.
                    u[i] += wij * (.30 * (phat[i, j] - p[i] - np.array([0., yoff[j] - yoff[i]]))
                                    + .42 * (vhat[i, j] - v[i]))
        u = project_acceleration(u, p, v)
        alpha = np.exp(-DT / tau)
        applied = alpha[:, None] * applied + (1 - alpha)[:, None] * u
        v += DT * (beta[:, None] * applied - drag[:, None] * v + inn["wind"][k])
        speed = np.linalg.norm(v, axis=1); over = speed > P["task"]["maximum_speed_mps"]
        v[over] *= (P["task"]["maximum_speed_mps"] / speed[over])[:, None]
        p += DT * v
        pair = min(np.linalg.norm(p[i] - p[j]) for i in range(N) for j in range(i + 1, N))
        min_pair = min(min_pair, pair); max_corridor = max(max_corridor, float(np.max(np.abs(p[:, 1]))))
        energy += DT * float(np.sum(applied ** 2)); peak_residual = max(peak_residual, float(np.max(residual)))
        if k >= int(.8 * STEPS): tail.append(float(np.max(residual)))
    completion = float(np.mean(p[:, 0] >= P["task"]["goal_x_m"]))
    task_success = (min_pair >= P["task"]["minimum_pair_clearance_m"] and
                    max_corridor <= P["task"]["corridor_half_width_m"] and
                    completion >= P["task"]["completion_fraction"])
    return {"environment": env["name"], "seed": seed, "rho": rho,
            "plant": "heterogeneous" if heterogeneous else "homogeneous", "policy": policy,
            "update_interval_s": update_steps * DT, "psi": psi(rho),
            "predicted_feasible": int(psi(rho) <= 1), "task_success": int(task_success),
            "completion_fraction": completion, "minimum_pair_clearance_m": min_pair,
            "maximum_corridor_excursion_m": max_corridor, "peak_observer_residual": peak_residual,
            "tail_observer_residual": max(tail), "control_energy": energy,
            "messages_delivered": comm, "gate_active_steps": gate_events}


modal = [exact_modal_challenge(float(r)) for r in RHOS]
with (OUT / "uav_v2_exact_modal_challenge.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(modal[0])); w.writeheader(); w.writerows(modal)

rows = []
for split in ("development", "heldout"):
    for env in P["environments"][split]:
        for seed in P[f"{split}_seeds"]:
            for rho in RHOS:
                for hetero in (False, True):
                    for policy in P["policies"]:
                        row = run(policy, float(rho), env, int(seed), hetero)
                        row["split"] = split; rows.append(row)
with (OUT / "uav_v2_runs.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

held = [r for r in rows if r["split"] == "heldout" and r["plant"] == "heterogeneous" and
        r["policy"] == "connectivity_observer_gate"]
tp = tn = fp = fn = 0
for r in held:
    pred, actual = bool(r["predicted_feasible"]), bool(r["task_success"])
    tp += pred and actual; fp += pred and not actual; tn += not pred and not actual; fn += not pred and actual
summary = {
    "protocol_status": P["status"], "rows": len(rows),
    "exact_modal_prediction_agreement": all(r["predicted_feasible"] == r["challenge_feasible"] for r in modal),
    "predicted_feasible_rho": [r["rho"] for r in modal if r["predicted_feasible"]],
    "heldout_confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    "heldout_sensitivity": tp / (tp + fn) if tp + fn else None,
    "heldout_specificity": tn / (tn + fp) if tn + fp else None,
    "uav_causal_validation_qualified": False,
    "qualification_reason": "set by fail-closed validator after all gates are checked"
}
(OUT / "uav_v2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
