#!/usr/bin/env python3
"""Build the editable main discovery figure from validated result tables."""
from pathlib import Path
import csv
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))

bounds = rows(ROOT / "finite_resource_boundary/results/finite_resource_bounds.csv")
vehicle = rows(ROOT / "physical_task_twins/results/vehicle_physical_tasks.csv")
robot = rows(ROOT / "physical_task_twins/results/robot_closed_loop_tasks.csv")

blue, orange, green, red, grey = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#666666"
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)

# a, logically distinct capacity zones.
ax = axs[0, 0]
ax.axhspan(1.0, 1.55, color="#DCEFE8", zorder=0)
ax.axhspan(0.0, 1.0, color="#F3E8C9", zorder=0)
ax.axhline(1, color="black", lw=0.8)
ax.annotate("guaranteed by upper bound", (0.03, 1.28), color=green)
ax.annotate("unresolved gap", (0.03, 0.56), color="#8A6D1D")
ax.annotate(r"impossible if $\Pi_L<1$", (0.03, 0.12), color=red)
ax.plot([0.68, 0.68], [0.18, 1.35], color=blue, lw=5, solid_capstyle="butt")
ax.plot([0.68], [1.35], marker="v", color=blue, ms=6)
ax.plot([0.68], [0.18], marker="^", color=red, ms=6)
ax.text(0.72, 1.34, r"$\Pi_U$", va="center")
ax.text(0.72, 0.18, r"$\Pi_L$", va="center")
ax.set(xlim=(0, 1), ylim=(0, 1.55), xticks=[], ylabel="task margin", title="Two bounds define three capability zones")

# b, topology-dependent upper/lower gaps.
ax = axs[0, 1]
names = ["direct pin", "directed chain", "directed cycle"]
keys = ["direct_pinning", "directed_chain", "directed_cyclic"]
data = [[float(r["actuation_upper_lower_gap"]) for r in bounds
         if r["topology"] == k and r["actuation_upper_lower_gap"]] for k in keys]
parts = ax.boxplot(data, patch_artist=True, widths=.55, showfliers=True,
                   medianprops={"color":"black", "lw":1})
for box, col in zip(parts["boxes"], [green, blue, orange]):
    box.set(facecolor=col, alpha=.55, edgecolor=col)
ax.axhline(1, color=grey, lw=.8, ls="--")
ax.set_yscale("log")
ax.set_xticklabels(names, rotation=15, ha="right")
ax.set(ylabel="upper/lower radius gap", title="No topology-independent critical constant")

# c, vehicle task feasibility across frozen and amended policies.
ax = axs[1, 0]
policies = ["all_coupled", "gated", "gated_safety_filter"]
labels = ["all coupled", "residual gate", "gate + range filter"]
colors = [blue, red, green]
for i, (p, label, col) in enumerate(zip(policies, labels, colors)):
    vals = [float(r["minimum_margin_m"]) for r in vehicle if r["policy"] == p]
    x = np.full(len(vals), i) + np.linspace(-.13, .13, len(vals))
    ax.scatter(x, vals, s=16, color=col, alpha=.75, label=label, edgecolor="none")
    ax.plot([i-.18, i+.18], [np.median(vals)]*2, color="black", lw=1.2)
ax.axhline(0, color="black", lw=.8)
ax.set_xticks(range(3), labels, rotation=15, ha="right")
ax.set(ylabel="minimum dynamic-spacing margin (m)", title="Information-safe switching can be physically unsafe")

# d, held-out robot formation integrity.
ax = axs[1, 1]
faults = ["none", "formation_change", "swinging", "noisy", "brute_force"]
short = ["none", "formation", "swinging", "noise", "attack"]
shown = [("all_coupled", "all coupled", blue),
         ("continuous_weight", "continuous", orange),
         ("connectivity_gate", "connectivity gate", green)]
for j, (p, label, colour) in enumerate(shown):
    med = []
    lo, hi = [], []
    for f in faults:
        vals = sorted(float(r["full_within_fraction"]) for r in robot if r["policy"] == p and r["fault"] == f)
        med.append(float(np.median(vals)))
        lo.append(float(np.percentile(vals, 25)))
        hi.append(float(np.percentile(vals, 75)))
    x = np.arange(len(faults)) + (-.18 + .18*j)
    yerr = np.array([np.array(med)-np.array(lo), np.array(hi)-np.array(med)])
    ax.errorbar(x, med, yerr=yerr, fmt="o", ms=4, capsize=2,
                color=colour, label=label)
ax.set_xticks(range(len(faults)), short, rotation=20, ha="right")
ax.set_ylim(.90, 1.005)
ax.set(ylabel="full-formation fraction within tolerance", title="Closed-loop selection does not improve the full formation")
ax.legend(frameon=False, ncol=1, loc="lower left")

for label, ax in zip("abcd", axs.flat):
    ax.text(-.13, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

for ext in ("pdf", "svg", "png"):
    fig.savefig(ROOT / "figures" / f"Fig21_two_margin_discovery.{ext}", dpi=400)
plt.close(fig)
