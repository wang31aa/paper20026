#!/usr/bin/env python3
"""Derive and plot the frozen dimensionless residual-certificate boundary."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "observer_in_loop_certified/results_r4/raw_runs.csv"
OUT = ROOT / "boundary_law/results"
FIG = ROOT / "figures"
SOURCE = ROOT / "figures/source_data"
TOLERANCES = (0.05, 0.10, 0.20, 0.40)


def load_rows() -> list[dict[str, str]]:
    with INPUT.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 576:
        raise ValueError(f"Expected 576 frozen R4 rows, found {len(rows)}")
    return rows


def expanded_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        delta = float(row["delta"])
        tail = float(row["tail20_max_tracking"])
        utilization = tail / delta
        for epsilon in TOLERANCES:
            number = epsilon / delta
            residual = tail / epsilon
            output.append({
                "topology": row["topology"],
                "heterogeneity": float(row["heterogeneity"]),
                "alpha": float(row["alpha"]),
                "gamma_state": float(row["gamma_state"]),
                "seed": int(row["seed"]),
                "epsilon": epsilon,
                "delta": delta,
                "tail_error": tail,
                "certificate_number": number,
                "normalized_tail_residual": residual,
                "certificate_utilization": utilization,
                "certified": int(number >= 1.0),
                "observed_within_tolerance": int(residual <= 1.0),
            })
    return output


def scalar_sharpness() -> list[dict[str, float]]:
    rows = []
    lam, ell, omega = 1.3, 0.7, 0.8
    for alpha in np.geomspace(0.15, 30.0, 120):
        delta = abs(omega) / (lam + alpha * ell)
        steady = abs(omega) / (lam + alpha * ell)
        rows.append({"alpha": float(alpha), "delta": delta,
                     "steady_error": steady, "ratio": steady / delta})
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(expanded: list[dict[str, object]], sharp: list[dict[str, float]]) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.2,
        "axes.linewidth": 0.65, "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axs = plt.subplots(2, 2, figsize=(7.08, 5.35), constrained_layout=True)
    palette = {0.05: "#0072B2", 0.10: "#009E73", 0.20: "#E69F00", 0.40: "#D55E00"}

    ax = axs[0, 0]
    for eps in TOLERANCES:
        rr = [r for r in expanded if r["epsilon"] == eps]
        x = np.asarray([r["certificate_number"] for r in rr])
        y = np.asarray([r["normalized_tail_residual"] for r in rr])
        ax.scatter(x, y, s=8, alpha=0.40, linewidth=0, color=palette[eps],
                   label=rf"$\varepsilon={eps:.2f}$")
    ax.axvline(1, color="black", lw=0.8, ls="--")
    ax.axhline(1, color="black", lw=0.8, ls=":")
    ax.set(xscale="log", yscale="log", xlabel=r"Certificate number, $\mathcal{C}=\varepsilon/\Delta$",
           ylabel=r"Observed residual, $E_{\rm tail}/\varepsilon$")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
    ax.text(1.08, 1.18, "not observed\nin this scan", fontsize=7, va="bottom")

    ax = axs[0, 1]
    base = [r for r in expanded if r["epsilon"] == TOLERANCES[0]]
    topo_order = ["chain", "star", "branch", "cyclic"]
    topo_colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    for i, (topology, color) in enumerate(zip(topo_order, topo_colors), 1):
        u = np.asarray([r["certificate_utilization"] for r in base if r["topology"] == topology])
        jitter = np.linspace(-0.13, 0.13, len(u))
        ax.scatter(i + jitter, np.sort(u), s=7, alpha=0.42, linewidth=0, color=color)
        ax.plot([i - .18, i + .18], [np.median(u)] * 2, color="black", lw=1.1)
    ax.axhline(1, color="black", lw=0.8, ls="--")
    ax.set(xticks=range(1, 5), xticklabels=topo_order,
           ylabel=r"Certificate utilization, $E_{\rm tail}/\Delta$")
    ax.set_ylim(0, 1.04)

    ax = axs[1, 0]
    x = np.asarray([r["alpha"] for r in sharp])
    delta = np.asarray([r["delta"] for r in sharp])
    steady = np.asarray([r["steady_error"] for r in sharp])
    ax.plot(x, delta, color="#0072B2", lw=1.6, label="Analytic radius")
    ax.plot(x, steady, color="#D55E00", lw=1.0, ls="--", label="Exact steady error")
    ax.set(xscale="log", yscale="log", xlabel=r"Coupling gain, $\alpha$",
           ylabel="Scalar residual")
    ax.legend(frameon=False, fontsize=7)
    ax.text(.03, .07, "ratio = 1 for all gains", transform=ax.transAxes, fontsize=7)

    ax = axs[1, 1]
    base_rows = load_rows()
    gammas = (2.0, 5.0, 15.0, 35.0)
    hs = (0.05, 0.15, 0.30)
    xpos = np.arange(len(gammas))
    for h, color in zip(hs, ["#56B4E9", "#0072B2", "#D55E00"]):
        med = []
        for gamma in gammas:
            vals = [float(r["tail20_max_tracking"]) for r in base_rows
                    if float(r["heterogeneity"]) == h and float(r["gamma_state"]) == gamma]
            med.append(float(np.median(vals)))
        ax.plot(xpos, med, marker="o", ms=3.2, lw=1.2, color=color, label=rf"$h={h:g}$")
    ax.set(xlabel=r"Observer gain, $\gamma_s$",
           ylabel="Median tail tracking error")
    ax.set_xticks(xpos, [f"{g:g}" for g in gammas])
    ax.legend(frameon=False, fontsize=7)

    for label, ax in zip("abcd", axs.flat):
        ax.text(-0.15, 1.06, label, transform=ax.transAxes, fontweight="bold",
                fontsize=9, va="top")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(width=.65, length=3)
    for ext, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600}),
                        ("tiff", {"dpi": 600})):
        fig.savefig(FIG / f"Fig1_dimensionless_boundary.{ext}", **kwargs)
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    expanded = expanded_rows(rows)
    sharp = scalar_sharpness()
    write_csv(OUT / "dimensionless_boundary_rows.csv", expanded)
    write_csv(OUT / "scalar_sharpness.csv", sharp)
    write_csv(SOURCE / "Fig1_dimensionless_boundary_rows.csv", expanded)
    write_csv(SOURCE / "Fig1_scalar_sharpness.csv", sharp)

    certified = [r for r in expanded if r["certified"]]
    violations = [r for r in certified if not r["observed_within_tolerance"]]
    max_u = max(float(r["certificate_utilization"]) for r in expanded)
    sharp_error = max(abs(float(r["delta"]) - float(r["steady_error"])) for r in sharp)
    summary = {
        "input_rows": len(rows), "expanded_rows": len(expanded),
        "ultimate_certified_rows": len(certified),
        "finite_tail_exceedances_among_ultimate_certified_rows": len(violations),
        "finite_tail_within_tolerance_rows": sum(
            int(r["observed_within_tolerance"]) for r in expanded),
        "max_certificate_utilization": max_u,
        "scalar_sharpness_max_abs_error": sharp_error,
        "tolerances": list(TOLERANCES),
        "interpretation_boundary": "ultimate normalized sufficient certificate; finite tail is an empirical consistency check, not a physical safety limit",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    make_figure(expanded, sharp)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
