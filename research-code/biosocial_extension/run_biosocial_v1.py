#!/usr/bin/env python3
"""Frozen synthetic mechanism tests for biological and social extensions."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = ROOT / "BIOSOCIAL_V1_FROZEN_CONTRACT.json"
RESULTS = ROOT / "results"


def load_contract():
    raw = CONTRACT_PATH.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def psi_curve(domain, rho):
    a = np.asarray(domain["a"], float)
    d0 = np.asarray(domain["D0"], float)
    db = np.asarray(domain["Db"], float)
    u = np.asarray(domain["U"], float)
    vh = np.asarray(domain["Vh"], float)
    residual = np.maximum(d0 + rho * db - u, 0.0) / a
    peak = residual + vh / rho
    return float(max(np.max(peak / domain["epsilon_peak"]),
                     np.max(residual / domain["epsilon_tail"])))


def simulate(domain, rho, seed, cfg):
    """Evaluate the exact response to bounded, non-fitted sinusoidal forcing."""
    rng = np.random.default_rng(seed)
    a = np.asarray(domain["a"], float)
    d0 = np.asarray(domain["D0"], float)
    db = np.asarray(domain["Db"], float)
    u = np.asarray(domain["U"], float)
    vh = np.asarray(domain["Vh"], float)
    dt, horizon, burn = cfg["dt"], cfg["horizon"], cfg["burn_in"]
    t = np.arange(0.0, horizon, dt)[:, None]
    # Random phases and amplitudes are fixed by held-out seed and remain within
    # the contract's persistent-mismatch envelope.
    phase = rng.uniform(0, 2 * np.pi, len(a))
    amp = rng.uniform(0.72, 1.0, len(a))
    omega = 0.7
    envelope = amp * (d0 + rho * db)
    base = envelope * 0.88 - u
    harmonic = envelope * 0.12
    aa = a[None, :]
    ph = phase[None, :]
    steady_sine = harmonic[None, :] / (aa*aa + omega*omega) * (
        aa*np.sin(omega*t + ph) - omega*np.cos(omega*t + ph))
    initial_sine = harmonic / (a*a + omega*omega) * (
        a*np.sin(phase) - omega*np.cos(phase))
    y = base[None, :] / aa * (1.0 - np.exp(-aa*t))
    y += steady_sine - initial_sine[None, :] * np.exp(-aa*t)
    close = np.isclose(a, rho)
    observer = np.empty_like(y)
    observer[:, ~close] = (vh[~close] / rho * a[~close]) * (
        np.exp(-rho*t) - np.exp(-a[~close][None, :]*t)) / (a[~close] - rho)
    observer[:, close] = (vh[close] / rho * a[close]) * t * np.exp(-rho*t)
    y = np.maximum(0.0, y + observer)
    peak = float(np.max(y[t[:, 0] >= burn]))
    tail = float(np.max(y[t[:, 0] >= 0.8*horizon]))
    success = peak <= domain["epsilon_peak"] and tail <= domain["epsilon_tail"]
    return peak, tail, success


def main():
    contract, digest = load_contract()
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for name, domain in contract["domains"].items():
        for rho in contract["participation_grid"]:
            psi = psi_curve(domain, rho)
            for split, seeds in (("development", contract["development_seeds"]),
                                 ("heldout", contract["heldout_seeds"])):
                for seed in seeds:
                    peak, tail, success = simulate(domain, rho, seed, contract["integration"])
                    rows.append({"domain": name, "rho": rho, "split": split,
                                 "seed": seed, "psi": psi,
                                 "certificate_safe": int(psi <= 1.0),
                                 "peak": peak, "tail": tail,
                                 "success": int(success)})
    out_csv = RESULTS / "biosocial_v1_runs.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    held = [r for r in rows if r["split"] == "heldout"]
    safe = [r for r in held if r["certificate_safe"]]
    summary = {
        "contract_sha256": digest,
        "status": contract["status"],
        "heldout_runs": len(held),
        "certificate_safe_runs": len(safe),
        "false_safe": sum(1 for r in safe if not r["success"]),
        "domain_summary": {}
    }
    for name in contract["domains"]:
        rr = [r for r in held if r["domain"] == name]
        summary["domain_summary"][name] = {
            "runs": len(rr),
            "success_rate": sum(r["success"] for r in rr) / len(rr),
            "safe_rho": [rho for rho in contract["participation_grid"]
                         if psi_curve(contract["domains"][name], rho) <= 1.0]
        }
    (RESULTS / "biosocial_v1_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")

    colors = {"biological_group_hunt": "#0072B2",
              "social_collective_choice": "#D55E00",
              "monotone_counterexample": "#009E73"}
    labels = {"biological_group_hunt": "Group-hunt model",
              "social_collective_choice": "Collective-choice model",
              "monotone_counterexample": "No-exposure counterexample"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    for name, domain in contract["domains"].items():
        rho = np.asarray(contract["participation_grid"])
        psi = np.asarray([psi_curve(domain, x) for x in rho])
        axes[0].plot(rho, psi, marker="o", ms=3.5, color=colors[name], label=labels[name])
        rates = []
        for x in rho:
            rr = [r for r in held if r["domain"] == name and r["rho"] == x]
            rates.append(np.mean([r["success"] for r in rr]))
        axes[1].plot(rho, rates, marker="o", ms=3.5, color=colors[name], label=labels[name])
    axes[0].axhline(1, color="0.25", lw=1, ls="--")
    axes[0].set(xlabel="Participation, $\\rho$", ylabel="Frozen certificate, $\\Psi_s(\\rho)$",
                title="a  Theory-first prediction")
    axes[1].set(xlabel="Participation, $\\rho$", ylabel="Held-out task success",
                title="b  Synthetic held-out test", ylim=(-0.04, 1.04))
    axes[0].legend(frameon=False, fontsize=7, loc="upper center")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.18, lw=0.6)
    fig.tight_layout()
    fig.savefig(RESULTS / "Fig_Biosocial_V1.pdf")
    fig.savefig(RESULTS / "Fig_Biosocial_V1.png", dpi=300)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
