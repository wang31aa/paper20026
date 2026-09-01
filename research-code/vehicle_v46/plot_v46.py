#!/usr/bin/env python3
from pathlib import Path
import plistlib
import subprocess
_check_output = subprocess.check_output
def _bounded_check_output(cmd, *args, **kwargs):
    if isinstance(cmd, (list, tuple)) and cmd and cmd[0] == "system_profiler":
        return plistlib.dumps([{"_items": []}])
    return _check_output(cmd, *args, **kwargs)
subprocess.check_output = _bounded_check_output
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent
d = pd.read_csv(R / "results/V46_PAIRED_RESULTS.csv")
order = ["all_coupled", "global_gain_reduction", "independent_tracking", "residual_gate",
         "connectivity_gate", "one_step_barrier_filter", "finite_grid_mpc", "two_layer"]
labels = ["All", "Low gain", "Independent", "Residual", "Connectivity", "Barrier", "Grid MPC", "Two-layer"]
colors = ["#6B7280", "#A78BFA", "#94A3B8", "#E07A5F", "#3D5A80", "#2A9D8F", "#F4A261", "#264653"]
fig, ax = plt.subplots(2, 2, figsize=(11.2, 6.5), constrained_layout=True)
x = np.arange(len(order))
summary = d.groupby("policy")
success = np.array([summary.get_group(p).task_success.mean() for p in order])
ax[0,0].bar(x, success, color=colors); ax[0,0].set_ylim(0, 1.05); ax[0,0].set_ylabel("Task success fraction")
for i, y in enumerate(success): ax[0,0].text(i, y + .025, f"{y:.2f}", ha="center", fontsize=7)

for p, c, lab in zip(order, colors, labels):
    g = summary.get_group(p).groupby("rho").minimum_margin_m.median()
    ax[0,1].plot(g.index, g.values, "o-", color=c, label=lab, lw=1.5, ms=4)
ax[0,1].axhline(0, color="black", lw=.8, ls="--"); ax[0,1].set_xlabel("Participation, ρ"); ax[0,1].set_ylabel("Median minimum margin (m)")

energy = [summary.get_group(p).control_energy.median() for p in order]
ax[1,0].bar(x, energy, color=colors); ax[1,0].set_ylabel("Median braking-command-squared cost\n(model units)")
tail = [summary.get_group(p).tail_tracking_rmse_mps.median() for p in order]
ax[1,1].scatter(tail, success, c=colors, s=55)
offsets = [(5,5),(5,6),(5,-11),(5,-2),(5,5),(5,5),(5,-11),(5,5)]
for xx, yy, lab, off in zip(tail, success, labels, offsets):
    ax[1,1].annotate(lab, (xx, yy), xytext=off, textcoords="offset points", fontsize=7)
ax[1,1].set_xlabel("Median tail RMSE (m s$^{-1}$)"); ax[1,1].set_ylabel("Task success fraction"); ax[1,1].set_ylim(0, 1.05)

for a, letter in zip(ax.flat, "abcd"):
    a.text(-.14, 1.06, letter, transform=a.transAxes, fontweight="bold", fontsize=12)
    a.spines[["top", "right"]].set_visible(False)
for a in (ax[0,0], ax[1,0]):
    a.set_xticks(x, labels, rotation=32, ha="right", fontsize=8)
ax[0,1].legend(ncol=2, fontsize=7, frameon=False, loc="best")
fig.suptitle("Heterogeneous vehicle counterfactuals expose a domain-specific recovery hierarchy", fontsize=13)
out = R / "results/Fig_V46_vehicle_counterfactual"
fig.savefig(str(out) + ".pdf", bbox_inches="tight")
fig.savefig(str(out) + ".svg", bbox_inches="tight")
fig.savefig(str(out) + ".png", dpi=240, bbox_inches="tight")
