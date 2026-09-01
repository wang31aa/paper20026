#!/usr/bin/env python3
"""Reproducible proof-of-concept for residual-gated cluster participation."""
from pathlib import Path
import csv, json
import numpy as np

ROOT = Path(__file__).resolve().parent
DT, T = 0.01, 30.0
N, ALPHA = 12, 2.8
TASK_TOL = 0.65


def graph():
    a = np.zeros((N, N))
    for i in range(N):
        for k in (1, 2, 4):
            a[i, (i-k) % N] = 1.0 / k
    return a


def forcing(t, phase):
    d = 0.10*np.sin(0.7*t + phase)
    # Three heterogeneous units become persistently incompatible in succession.
    for node, start, amp in ((7, 6.0, 3.4), (9, 13.0, -4.0), (4, 20.0, 4.5)):
        if t >= start:
            d[node] += amp + 0.25*np.sin(1.3*t + phase[node])
    return d


def simulate(seed, policy):
    a = graph()
    decay = np.linspace(0.45, 1.15, N)
    x = 0.08*np.random.default_rng(seed + 100).standard_normal(N)
    phase = np.random.default_rng(seed).uniform(0, 2*np.pi, N)
    trusted = np.ones(N, dtype=bool)
    over = np.zeros(N); under = np.zeros(N)
    rows, safe = [], []
    bad = {4, 7, 9}
    for k in range(int(T/DT)+1):
        t = k*DT
        centre = np.median(x[trusted]) if trusted.any() else 0.0
        residual = np.abs(x-centre)
        if policy == "gated":
            over = np.where(residual > 0.75, over+DT, 0.0)
            under = np.where(residual < 0.45, under+DT, 0.0)
            trusted[(over >= 0.18) & (np.arange(N) != 0)] = False
            trusted[(under >= 0.75)] = True
            trusted[0] = True  # preserve the target-rooted backbone
        influence = trusted.astype(float) if policy == "gated" else np.ones(N)
        coupling = np.sum(a * influence[None, :] * (x[:, None]-x[None, :]), axis=1)
        pin = np.zeros(N); pin[0] = 2.0*x[0]
        dx = -decay*x - ALPHA*coupling - pin + forcing(t, phase)
        x += DT*dx
        good_error = np.max(np.abs(x[[i for i in range(N) if i not in bad]]))
        safe.append(good_error <= TASK_TOL)
        if k % 5 == 0:
            rows.append((t, policy, seed, good_error, int(trusted.sum()),
                         int(trusted[4]), int(trusted[7]), int(trusted[9])))
    safe = np.asarray(safe)
    first = np.flatnonzero(~safe)
    return rows, {
        "seed": seed, "policy": policy,
        "survival_time": float(first[0]*DT if first.size else T),
        "safe_time_fraction": float(safe.mean()),
        "final_trusted_nodes": int(trusted.sum()),
    }


def main():
    raw = ROOT/"results"; raw.mkdir(exist_ok=True)
    trajectories, summary = [], []
    for seed in range(12):
        for policy in ("all_coupled", "gated"):
            r, s = simulate(seed, policy); trajectories.extend(r); summary.append(s)
    with (raw/"trajectories.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["time","policy","seed","good_node_max_error",
            "trusted_count","node4_trusted","node7_trusted","node9_trusted"]); w.writerows(trajectories)
    with (raw/"summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary[0]); w.writeheader(); w.writerows(summary)
    for policy in ("all_coupled", "gated"):
        with (raw/f"figure_seed0_{policy}.csv").open("w", newline="") as f:
            w=csv.writer(f); w.writerow(["time","error","trusted"])
            w.writerows((r[0],r[3],r[4]) for r in trajectories if r[1]==policy and r[2]==0)
    with (raw/"figure_survival.csv").open("w", newline="") as f:
        w=csv.writer(f); w.writerow(["seed","all_coupled","gated"])
        for seed in range(12):
            vals={r["policy"]:r["survival_time"] for r in summary if r["seed"]==seed}
            w.writerow([seed,vals["all_coupled"],vals["gated"]])
    out = {}
    for p in ("all_coupled", "gated"):
        q = [r for r in summary if r["policy"] == p]
        out[p] = {k: float(np.median([r[k] for r in q])) for k in
                  ("survival_time", "safe_time_fraction", "final_trusted_nodes")}
    (raw/"summary.json").write_text(json.dumps(out, indent=2)+"\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__": main()
