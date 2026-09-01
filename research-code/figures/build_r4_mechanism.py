#!/usr/bin/env python3
"""Build the editable CERT-OIL-R4 mechanism figure from frozen run rows."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "observer_in_loop_certified/results_r4/raw_runs.csv"
OUT = ROOT / "figures"


def values(rows, key):
    return np.asarray([float(r[key]) for r in rows])


def subset(rows, **terms):
    return [r for r in rows if all(r[k] == str(v) for k, v in terms.items())]


def median_range(ax, x, groups, key, labels, colors):
    for group, label, color in zip(groups, labels, colors):
        med, lo, hi = [], [], []
        for term in group:
            y = values(term, key)
            med.append(np.median(y)); lo.append(np.min(y)); hi.append(np.max(y))
        med, lo, hi = map(np.asarray, (med, lo, hi))
        ax.plot(x, med, marker="o", ms=3.2, lw=1.25, label=label, color=color)
        ax.fill_between(x, lo, hi, color=color, alpha=.12, linewidth=0)


def main():
    with SOURCE.open() as handle:
        rows = list(csv.DictReader(handle))
    alphas = [1.5, 2.0, 4.0, 8.0]
    gammas = [2.0, 5.0, 15.0, 35.0]
    hs = [0.05, 0.15, 0.3]
    colors = ["#0072B2", "#D55E00", "#009E73"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "axes.linewidth": .65, "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    fig, axs = plt.subplots(2, 2, figsize=(7.08, 5.25), constrained_layout=True)
    ax = axs[0, 0]
    groups = [[subset(rows, alpha=a, heterogeneity=h) for h in hs] for a in alphas]
    median_range(ax, hs, groups, "delta", [rf"$\alpha={a:g}$" for a in alphas],
                 ["#56B4E9", "#0072B2", "#E69F00", "#D55E00"])
    ax.set(xlabel="Heterogeneity, $h$", ylabel=r"Analytic radius, $\Delta$")
    ax.legend(frameon=False, ncol=2, fontsize=7.0)

    ax = axs[0, 1]
    groups = [[subset(rows, alpha=a, heterogeneity=h) for a in alphas] for h in hs]
    median_range(ax, alphas, groups, "tail20_max_tracking",
                 [rf"$h={h:g}$" for h in hs], colors)
    ax.set(xlabel=r"Coupling gain, $\alpha$",
           ylabel="Tail maximum tracking error")
    ax.set_xticks(alphas, [f"{a:g}" for a in alphas])
    ax.legend(frameon=False, fontsize=7.0)

    ax = axs[1, 0]
    topologies = ["chain", "star", "branch", "cyclic"]
    topo_colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    groups = [[subset(rows, gamma_state=g, topology=t) for g in gammas]
              for t in topologies]
    median_range(ax, gammas, groups, "final_observer", topologies, topo_colors)
    ax.set(yscale="log", xlabel=r"Observer gain, $\gamma_s$",
           ylabel="Final observer error")
    ax.set_xticks(gammas, [f"{g:g}" for g in gammas])
    ax.set_ylim(bottom=1e-18)
    ax.legend(frameon=False, ncol=2, fontsize=7.0)

    ax = axs[1, 1]
    for a, color in zip(alphas, ["#56B4E9", "#0072B2", "#E69F00", "#D55E00"]):
        ss = subset(rows, alpha=a)
        x, y = values(ss, "max_control_norm"), values(ss, "tail20_max_tracking")
        ax.scatter(np.median(x), np.median(y), s=25, color=color,
                   edgecolor="white", linewidth=.4, label=rf"$\alpha={a:g}$", zorder=3)
        ax.plot([np.min(x), np.max(x)], [np.median(y), np.median(y)], color=color, lw=.8)
        ax.plot([np.median(x), np.median(x)], [np.min(y), np.max(y)], color=color, lw=.8)
    ax.set(xlabel="Maximum control norm", ylabel="Tail maximum tracking error")
    ax.legend(frameon=False, ncol=2, fontsize=7.0)

    for label, ax in zip("abcd", axs.flat):
        ax.text(-.16, 1.06, label, transform=ax.transAxes, fontweight="bold",
                fontsize=9, va="top")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(width=.65, length=3)
    for ext, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600}),
                        ("tiff", {"dpi": 600})):
        fig.savefig(OUT / f"Fig13_mechanism_scan.{ext}", **kwargs)


if __name__ == "__main__":
    main()
