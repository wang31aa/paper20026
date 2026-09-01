#!/usr/bin/env python3
"""Non-trivial network lower bounds and domain-dynamics intervention tests.

The public records set forcing profiles and data-referenced task-proxy
thresholds.  They do not constitute executions of the proposed controller.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SOURCE = ROOT / "figures" / "source_data"
FIGURES = ROOT / "figures"
CACHE = ROOT / "public_cluster_cases" / "cache"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def robust_scale(x: np.ndarray) -> float:
    q1, q3 = np.quantile(x, [0.25, 0.75])
    return max(float((q3 - q1) / 1.349), 1e-9)


def pinned_laplacians(n: int) -> dict[str, np.ndarray]:
    chain = np.zeros((n, n))
    chain[0, 0] = 1
    for i in range(1, n):
        chain[i, i] = 1
        chain[i, i - 1] = -1
    star = np.eye(n)
    cyclic = chain.copy()
    cyclic[0, 0] += 0.25
    cyclic[0, -1] = -0.25
    if n > 2:
        cyclic[2, 2] += 0.4
        cyclic[2, 0] = -0.4
    return {"directed_chain": chain, "direct_pinning": star,
            "directed_cyclic": cyclic}


def graph_metric(lap: np.ndarray) -> tuple[np.ndarray, float]:
    g = np.linalg.solve(lap.T, np.ones(lap.shape[0]))
    s = np.diag(g) @ lap + lap.T @ np.diag(g)
    mu = np.linalg.eigvalsh(np.diag(g ** -0.5) @ s @
                            np.diag(g ** -0.5)).min()
    return g, float(mu)


def directed_lower_bounds() -> list[dict[str, float | str]]:
    """Exact worst-direction steady residual for linear N=5 members."""
    out: list[dict[str, float | str]] = []
    lam = 0.2
    for topology, lap in pinned_laplacians(5).items():
        g, mu = graph_metric(lap)
        for alpha in np.geomspace(0.1, 30.0, 80):
            system = lam * np.eye(5) + alpha * lap
            # weighted-forcing unit ball: d=G^{-1/2}q, ||q||_2<=1
            transfer = np.linalg.solve(system, np.diag(g ** -0.5))
            u, singular, vh = np.linalg.svd(transfer)
            exact = float(singular[0])
            certificate = 2 / ((alpha * mu + 2 * lam) * np.sqrt(g.min()))
            out.append({"topology": topology, "alpha": float(alpha),
                        "exact_worst_residual": exact,
                        "certificate_radius": float(certificate),
                        "attainment_ratio": exact / certificate,
                        "forcing_direction_1": float(vh[0, 0])})
    return out


def public_domain_arrays() -> dict[str, dict[str, np.ndarray | str]]:
    robot_rows = [r for r in read(CACHE / "robot_swarm_validation.csv") if r["run"] == "0"]
    nt = max(int(r["step"]) for r in robot_rows) + 1
    nr = max(int(r["robot_idx"]) for r in robot_rows) + 1
    rpos = np.zeros((nt, nr, 2)); ract = np.zeros_like(rpos)
    for row in robot_rows:
        t, i = int(row["step"]), int(row["robot_idx"])
        rpos[t, i] = [float(row["pos_x"]), float(row["pos_y"])]
        ract[t, i] = [float(row["action_x"]), float(row["action_y"])]

    uav_rows = read(CACHE / "uav_swarm_synthetic.csv")
    nt = max(int(r["timestamp"]) for r in uav_rows) + 1
    nu = max(int(r["drone_id"]) for r in uav_rows) + 1
    upos = np.zeros((nt, nu, 3)); uvel = np.zeros((nt, nu, 1))
    for row in uav_rows:
        t, i = int(row["timestamp"]), int(row["drone_id"])
        upos[t, i] = [float(row["x"]), float(row["y"]), float(row["z"])]
        uvel[t, i, 0] = float(row["velocity"])

    vehicle_rows = [r for r in read(CACHE / "adas_two_vehicle_sample.csv") if r["id"] == "6"]
    vvel = np.asarray([[float(r[k]) for k in ("speed_av", "speed_sv1", "speed_sv2")]
                       for r in vehicle_rows])[:, :, None]
    vacc = np.asarray([[float(r[k]) for k in ("acc_av", "acc_sv1", "acc_sv2")]
                       for r in vehicle_rows])[:, :, None]
    headway = np.asarray([float(r["distance_av_headway"]) for r in vehicle_rows])
    return {
        "robot": {"state": rpos, "forcing": ract, "model": "first-order formation kinematics"},
        "uav": {"state": upos, "forcing": uvel, "model": "damped second-order translational surrogate"},
        "vehicle": {"state": vvel, "forcing": vacc, "headway": headway,
                    "model": "damped longitudinal speed-error surrogate"},
    }


def disagreement(x: np.ndarray) -> np.ndarray:
    centred = x - np.mean(x, axis=1, keepdims=True)
    return np.linalg.norm(centred.reshape(len(x), -1), axis=1)


def resample_profile(x: np.ndarray, count: int) -> np.ndarray:
    old = np.linspace(0, 1, len(x)); new = np.linspace(0, 1, count)
    flat = x.reshape(len(x), -1)
    return np.column_stack([np.interp(new, old, flat[:, j])
                            for j in range(flat.shape[1])]).reshape((count,) + x.shape[1:])


def simulate_domain(name: str, state: np.ndarray, forcing: np.ndarray,
                    intervention: str) -> dict[str, float | str | bool]:
    n, dim = forcing.shape[1:]
    split = max(3, int(0.6 * len(state)))
    observed = disagreement(state)
    source_threshold = float(np.quantile(observed[:split], 0.95))
    # State error is reported in units of this calibration-only 95th
    # percentile, so the data-referenced task-proxy threshold is exactly one.
    epsilon = 1.0
    scale = robust_scale((forcing[:split] - np.median(forcing[:split], axis=1,
                                                      keepdims=True)).ravel())
    profile = (forcing - np.median(forcing, axis=1, keepdims=True)) / scale
    count, dt = 2401, 0.004
    profile = resample_profile(profile[split:], count)
    lap = pinned_laplacians(n)["directed_chain"]
    alpha, gamma, hetero = 1.8, 4.0, 1.0
    if intervention == "information": gamma *= 4
    elif intervention == "coupling": alpha *= 2
    elif intervention == "heterogeneity": hetero *= 0.5
    elif intervention == "topology": lap = pinned_laplacians(n)["direct_pinning"]
    rng = np.random.default_rng({"robot": 401, "uav": 402, "vehicle": 403}[name])
    e = rng.normal(0, 0.05, (n, dim))
    v = np.zeros_like(e)
    eta = rng.normal(0, 0.05, (n, dim))
    trace = np.empty(count)
    second_order = name in {"uav", "vehicle"}
    kp, kd = (1.2, 1.1) if name == "uav" else (0.8, 1.4)
    for k in range(count):
        trace[k] = np.linalg.norm(e)
        d = hetero * profile[k]
        pin_eta = np.zeros_like(e); pin_eta[0] = eta[0]
        if second_order:
            ae = -kp * e - kd * v - alpha * (lap @ e) + d + alpha * pin_eta
            e += dt * v
            v += dt * ae
        else:
            e += dt * (-1.0 * e - alpha * (lap @ e) + d + alpha * pin_eta)
        eta += dt * (-gamma * (lap @ eta))
    held = trace[int(0.6 * count):]
    task_proxy = float(np.quantile(held, 0.95))
    return {"case": name, "intervention": intervention, "model_dimension": int(n * dim * (2 if second_order else 1)),
            "source_disagreement_p95": source_threshold,
            "calibration_task_proxy_threshold": epsilon,
            "heldout_p95_error": task_proxy,
            "normalized_task_margin": task_proxy / max(epsilon, 1e-12),
            "proxy_threshold_met": bool(task_proxy <= epsilon),
            "alpha": alpha, "observer_gain": gamma, "forcing_scale": hetero}


def write_csv(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader(); writer.writerows(data)


def make_figure(lower: list[dict], domains: list[dict]) -> None:
    plt.rcParams.update({"font.size": 7.3, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axs = plt.subplots(2, 2, figsize=(7.08, 5.25), constrained_layout=True)
    colors = {"robot": "#0072B2", "uav": "#D55E00", "vehicle": "#009E73"}
    ax = axs[0, 0]
    for topology, color in zip(("directed_chain", "directed_cyclic", "direct_pinning"),
                               ("#0072B2", "#CC79A7", "#666666")):
        rr = [r for r in lower if r["topology"] == topology]
        ax.plot([r["alpha"] for r in rr], [r["attainment_ratio"] for r in rr],
                lw=1.4, color=color, label=topology.replace("_", " "))
    ax.set(xscale="log", ylim=(0.48, 1.03), xlabel="Coupling gain, α",
           ylabel="Worst residual / certificate")
    ax.axhline(1, ls="--", color="black", lw=.8); ax.legend(frameon=False, fontsize=7)

    ax = axs[0, 1]
    baseline = [r for r in domains if r["intervention"] == "baseline"]
    ax.bar(range(3), [r["normalized_task_margin"] for r in baseline],
           color=[colors[r["case"]] for r in baseline], width=.62)
    ax.axhline(1, color="black", ls="--", lw=.8)
    ax.set(xticks=range(3), xticklabels=[r["case"] for r in baseline],
           ylabel="Held-out error / calibrated proxy")

    ax = axs[1, 0]
    labels = ["information", "coupling", "heterogeneity", "topology"]
    x = np.arange(4); width = .24
    for j, case in enumerate(("robot", "uav", "vehicle")):
        base = next(float(r["heldout_p95_error"]) for r in baseline if r["case"] == case)
        vals = [next(float(r["heldout_p95_error"]) for r in domains
                     if r["case"] == case and r["intervention"] == label) / base for label in labels]
        ax.bar(x + (j - 1) * width, vals, width=width, color=colors[case], label=case)
    ax.axhline(1, color="black", lw=.8)
    ax.set_yscale("log")
    ax.set(xticks=x, xticklabels=["faster\nobserver", "stronger\ncoupling",
                                 "half\nspread", "direct\npinning"],
           ylabel="Intervention error / baseline")
    ax.legend(frameon=False, fontsize=7, ncol=3)

    ax = axs[1, 1]
    effects = []
    for case in ("robot", "uav", "vehicle"):
        base = next(float(r["heldout_p95_error"]) for r in baseline if r["case"] == case)
        for label in labels:
            val = next(float(r["heldout_p95_error"]) for r in domains
                       if r["case"] == case and r["intervention"] == label)
            effects.append((label, 1 - val / base))
    for i, label in enumerate(labels):
        vals = [v for l, v in effects if l == label]
        ax.scatter([i] * len(vals), vals, s=22,
                   c=[colors[c] for c in ("robot", "uav", "vehicle")])
        ax.plot([i - .18, i + .18], [np.median(vals)] * 2, color="black", lw=1.2)
    ax.axhline(0, color="black", lw=.8)
    ax.set_yscale("symlog", linthresh=.1)
    ax.set(xticks=range(4), xticklabels=["information", "coupling", "spread", "topology"],
           ylabel="Fractional reduction from baseline")
    for label, ax in zip("abcd", axs.flat):
        ax.text(-.15, 1.07, label, transform=ax.transAxes, fontweight="bold", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=.15, axis="y")
    for ext, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        fig.savefig(FIGURES / f"Fig16_discovery_extension.{ext}", **kwargs)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True); SOURCE.mkdir(parents=True, exist_ok=True)
    lower = directed_lower_bounds()
    domains_raw = public_domain_arrays(); domains: list[dict] = []
    for case, payload in domains_raw.items():
        for intervention in ("baseline", "information", "coupling", "heterogeneity", "topology"):
            row = simulate_domain(case, payload["state"], payload["forcing"], intervention)  # type: ignore[arg-type]
            row["model"] = payload["model"]
            if case == "vehicle":
                headway = payload["headway"]  # type: ignore[assignment]
                row["recorded_headway_p10"] = float(np.quantile(headway, .10))
            else:
                row["recorded_headway_p10"] = ""
            domains.append(row)
    write_csv(RESULTS / "directed_network_lower_bound.csv", lower)
    write_csv(RESULTS / "domain_interventions.csv", domains)
    write_csv(SOURCE / "Fig16_directed_network_lower_bound.csv", lower)
    write_csv(SOURCE / "Fig16_domain_interventions.csv", domains)
    summary = {
        "directed_chain_max_attainment": max(r["attainment_ratio"] for r in lower if r["topology"] == "directed_chain"),
        "directed_cyclic_max_attainment": max(r["attainment_ratio"] for r in lower if r["topology"] == "directed_cyclic"),
        "domain_rows": len(domains),
        "classification": "public-data-constrained domain-surrogate computational extension",
        "task_threshold_boundary": "calibration-data proxy, not collision or deployment safety",
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (SOURCE / "Fig16_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_figure(lower, domains)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
