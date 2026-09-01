#!/usr/bin/env python3
"""Dataset-informed cluster simulations for robots, UAVs and road vehicles."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"svg.fonttype": "none", "font.size": 8})


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
RESULTS = HERE / "results"
FIGURES = ROOT / "figures"
SOURCE = FIGURES / "source_data"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def robust_scale(x: np.ndarray) -> float:
    q25, q75 = np.quantile(x, [0.25, 0.75])
    return max(float((q75 - q25) / 1.349), 1e-9)


def robot_forcing() -> tuple[np.ndarray, dict]:
    data = [r for r in rows(CACHE / "robot_swarm_validation.csv") if r["run"] == "0"]
    agents = sorted({int(r["robot_idx"]) for r in data})
    steps = sorted({int(r["step"]) for r in data})
    action = np.zeros((len(steps), len(agents), 2))
    for r in data:
        action[int(r["step"]), int(r["robot_idx"])] = [float(r["action_x"]), float(r["action_y"])]
    centre = np.median(action, axis=1, keepdims=True)
    residual = action - centre
    scale = robust_scale(residual.reshape(-1))
    return residual / scale, {"rows": len(data), "agents": len(agents), "run": 0,
                              "normalization": scale, "source_kind": "public simulated formation-control validation"}


def uav_forcing() -> tuple[np.ndarray, dict]:
    data = rows(CACHE / "uav_swarm_synthetic.csv")
    agents = sorted({int(r["drone_id"]) for r in data})
    times = sorted({int(r["timestamp"]) for r in data})
    velocity = np.zeros((len(times), len(agents), 1))
    signal = np.zeros_like(velocity)
    for r in data:
        t, i = int(r["timestamp"]), int(r["drone_id"])
        velocity[t, i, 0] = float(r["velocity"])
        signal[t, i, 0] = float(r["signal_strength"])
    residual = velocity - np.median(velocity, axis=1, keepdims=True)
    scale = robust_scale(residual.reshape(-1))
    return residual / scale, {"rows": len(data), "agents": len(agents),
                              "normalization": scale,
                              "median_signal_strength": float(np.median(signal)),
                              "source_kind": "public synthetic UAV-swarm trajectories"}


def vehicle_forcing() -> tuple[np.ndarray, dict]:
    data_all = rows(CACHE / "adas_two_vehicle_sample.csv")
    data = [r for r in data_all if r["id"] == "6"]
    speed = np.array([[float(r[k]) for k in ("speed_av", "speed_sv1", "speed_sv2")]
                      for r in data], dtype=float)[:, :, None]
    residual = speed - np.median(speed, axis=1, keepdims=True)
    scale = robust_scale(residual.reshape(-1))
    return residual / scale, {"rows": len(data), "agents": 3, "record_id": 6,
                              "normalization": scale,
                              "source_kind": "public naturalistic ADAS multi-vehicle trajectories"}


def pinned_chain(n: int) -> np.ndarray:
    lap = np.zeros((n, n))
    for i in range(n - 1):
        lap[i, i] += 1
        lap[i + 1, i + 1] += 1
        lap[i, i + 1] -= 1
        lap[i + 1, i] -= 1
    lap[0, 0] += 1
    return lap


def simulate(name: str, forcing: np.ndarray, metadata: dict) -> tuple[dict, np.ndarray]:
    nt, n, dim = forcing.shape
    lap = pinned_chain(n)
    eig_min = float(np.linalg.eigvalsh(lap)[0])
    alpha, a, gamma = 4.0, 2.0, 8.0
    mu = 2 * eig_min
    dmargin = alpha * mu + 2 * a
    beta = dmargin / 2
    kappa = gamma * eig_min
    w = np.max(np.linalg.norm(forcing, axis=2), axis=0)
    omega = float(np.linalg.norm(w))
    delta = 2 * omega / dmargin

    dt, horizon = 0.002, 8.0
    count = int(round(horizon / dt)) + 1
    t = np.linspace(0, horizon, count)
    rng = np.random.default_rng({"robot": 1201, "uav": 1202, "vehicle": 1203}[name])
    e = rng.normal(0, 0.25, size=(n, dim))
    eta = rng.normal(0, 0.20, size=(n, dim))
    z0, h0 = float(np.linalg.norm(e)), float(np.linalg.norm(eta))
    tracking = np.empty(count)
    observer = np.empty(count)
    envelope = np.empty(count)

    def disturbance(time: float) -> np.ndarray:
        idx = min(int(time / horizon * nt), nt - 1)
        return forcing[idx]

    def rhs(time: float, ee: np.ndarray, hh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        de = -a * ee - alpha * (lap @ ee) + disturbance(time)
        de += alpha * np.eye(n)[:, [0]] @ hh[[0], :]
        dh = -gamma * (lap @ hh)
        return de, dh

    for k, time in enumerate(t):
        tracking[k] = np.linalg.norm(e)
        observer[k] = np.linalg.norm(eta)
        if abs(beta - kappa) < 1e-12:
            conv = alpha * h0 * time * np.exp(-beta * time)
        else:
            conv = alpha * h0 * (np.exp(-kappa * time) - np.exp(-beta * time)) / (beta - kappa)
        envelope[k] = np.exp(-beta * time) * z0 + omega / beta * (1 - np.exp(-beta * time)) + conv
        if k == count - 1:
            break
        k1e, k1h = rhs(time, e, eta)
        k2e, k2h = rhs(time + dt / 2, e + dt * k1e / 2, eta + dt * k1h / 2)
        k3e, k3h = rhs(time + dt / 2, e + dt * k2e / 2, eta + dt * k2h / 2)
        k4e, k4h = rhs(time + dt, e + dt * k3e, eta + dt * k3h)
        e += dt * (k1e + 2 * k2e + 2 * k3e + k4e) / 6
        eta += dt * (k1h + 2 * k2h + 2 * k3h + k4h) / 6

    positive = t > 0
    ratio = float(np.max(tracking[positive] / envelope[positive]))
    result = dict(case=name, n_agents=n, state_dimension=dim, alpha=alpha,
                  local_decay=a, observer_gain=gamma, graph_mu=mu,
                  disturbance_envelope=omega, analytic_radius=delta,
                  tail_tracking_max=float(np.max(tracking[int(0.8 * count):])),
                  final_tracking=float(tracking[-1]),
                  max_positive_time_envelope_ratio=ratio,
                  envelope_pass=bool(ratio <= 1 + 1e-10), **metadata)
    trace = np.column_stack([t, tracking, envelope, observer])
    return result, trace


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)
    cases = {"robot": robot_forcing(), "uav": uav_forcing(), "vehicle": vehicle_forcing()}
    results, traces = [], {}
    for name, (forcing, metadata) in cases.items():
        result, trace = simulate(name, forcing, metadata)
        results.append(result)
        traces[name] = trace

    with (RESULTS / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(dict.fromkeys(key for result in results for key in result))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(results)
    for name, trace in traces.items():
        np.savetxt(RESULTS / f"{name}_trace.csv", trace, delimiter=",",
                   header="time,tracking_error,analytic_envelope,observer_error", comments="")

    provenance = {
        "protocol": "PUBLIC-CLUSTER-CASES-R1",
        "classification": "dataset-informed computational validation",
        "sources": {
            "robot": {"doi": "10.18419/DARUS-6135", "license": "CC BY 4.0",
                      "sha256": sha256(CACHE / "robot_swarm_validation.csv")},
            "uav": {"url": "https://huggingface.co/datasets/jason1966/ahsanneural_drone-swarm-coordination-dataset",
                    "license": "upstream license not stated; raw file excluded from release",
                    "sha256": sha256(CACHE / "uav_swarm_synthetic.csv")},
            "vehicle": {"url": "https://data.transportation.gov/d/vhz2-exyi",
                        "query": "first 5000 rows from Socrata resource API on 2026-08-11",
                        "access": "public US Department of Transportation dataset",
                        "sha256": sha256(CACHE / "adas_two_vehicle_sample.csv")},
        },
        "results": results,
    }
    (RESULTS / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    colors = {"robot": "#0072B2", "uav": "#D55E00", "vehicle": "#009E73"}
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.25), constrained_layout=True)
    titles = {"robot": "Heterogeneous robot formation", "uav": "UAV swarm",
              "vehicle": "ADAS vehicle cluster"}
    for ax, name in zip(axes, ("robot", "uav", "vehicle")):
        trace = traces[name]
        ax.plot(trace[:, 0], trace[:, 1], color=colors[name], lw=1.6, label="tracking error")
        ax.plot(trace[:, 0], trace[:, 2], color="#333333", lw=1.3, ls="--", label="analytic envelope")
        ax.set_title(titles[name], fontsize=9)
        ax.set_xlabel("normalized time")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("stacked normalized error")
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.savefig(FIGURES / "Fig15_public_cluster_cases.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "Fig15_public_cluster_cases.svg", bbox_inches="tight")
    fig.savefig(FIGURES / "Fig15_public_cluster_cases.png", dpi=300, bbox_inches="tight")

    for src in (RESULTS / "summary.csv", RESULTS / "provenance.json"):
        (SOURCE / ("Extended_Data_Fig4_" + src.name)).write_bytes(src.read_bytes())
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
