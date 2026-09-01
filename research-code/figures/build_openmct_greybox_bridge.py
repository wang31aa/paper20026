#!/usr/bin/env python3
"""Build an editable figure for the frozen OpenMCT grey-box bridge."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "openmct_greybox_round42/results"
OUT = ROOT / "figures"
BLUE, ORANGE, TEAL, GREY = "#0072B2", "#D55E00", "#009E73", "#666666"


def main():
    rows = list(csv.DictReader((RES / "holdout_metrics.csv").open()))
    ident = json.loads((RES / "identifiability.json").read_text())
    labels = ["PI 10", "PI 20", "PI 50", "PI 5", "Discrete A", "Discrete B", "Discrete C"]
    grey = np.array([float(r["grey_recursive_nrmse"]) for r in rows])
    base = np.array([float(r["persistence_recursive_nrmse"]) for r in rows])
    arx = np.array([float(r["arx_recursive_nrmse"]) for r in rows])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                         "axes.labelsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7,
                         "axes.linewidth": .7, "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    fig, axs = plt.subplots(2, 2, figsize=(7.08, 5.15), constrained_layout=True)

    ax = axs[0, 0]; ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    boxes = [(0.02, "3 APRBS\nidentification\nrecords", "#E6F2FA"),
             (0.39, "Dissipative\nmotor model", "#E6F4EF"),
             (0.76, "7 complete\ncontroller\nholdouts", "#F8EBDD")]
    for x, text, colour in boxes:
        ax.add_patch(plt.Rectangle((x, .34), .24, .34, facecolor=colour,
                                   edgecolor=GREY, linewidth=.8))
        ax.text(x + .12, .54, text, ha="center", va="center", fontsize=7)
    ax.text(.51, .39, r"$\dot y=-ay+bu+c$", ha="center", va="center", fontsize=7)
    for start, end in ((.27, .38), (.64, .75)):
        ax.annotate("", (end, .51), (start, .51),
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=1))
    ax.text(.03, .82, "Public measurements constrain a dissipative motor model",
            weight="bold")

    ax = axs[0, 1]; x = np.arange(len(rows)); w = .35
    ax.bar(x - w/2, grey, w, color=BLUE, label="Grey-box")
    ax.bar(x + w/2, base, w, color="#B9B9B9", label="Constant-state baseline")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Recursive NRMSE")
    ax.legend(frameon=False)
    ax.set_title("Grey-box recursion generalizes to every holdout", loc="left", fontsize=7.5)

    ax = axs[1, 0]
    names = ["$a$", "$b$", "$c$"]; pooled = np.array([ident["theta"][k] for k in "abc"])
    ci = np.array([ident["ci95"][k] for k in "abc"])
    yp = np.arange(3)
    ax.errorbar(pooled, yp, xerr=np.vstack([pooled-ci[:, 0], ci[:, 1]-pooled]),
                fmt="o", color=BLUE, capsize=3, label="Pooled estimate and 95% interval")
    for j, item in enumerate(ident["leave_one_out"]):
        ax.scatter(item["theta"], yp + (j-1)*.10, s=15,
                   color=[ORANGE, TEAL, "#CC79A7"][j],
                   label=f"Leave out {['2 ms','10 ms','20 ms'][j]}")
    ax.set_yticks(yp, names); ax.set_xlabel("Parameter value")
    ax.set_title("Record removal exposes parameter transport uncertainty", loc="left", fontsize=7.5)
    ax.legend(frameon=False, fontsize=7.0)

    ax = axs[1, 1]
    ax.scatter(grey[:4], arx[:4], color=TEAL, marker="o", s=28,
               edgecolor="white", linewidth=.4, label="Continuous PI")
    ax.scatter(grey[4:], arx[4:], color=ORANGE, marker="s", s=28,
               edgecolor="white", linewidth=.4, label="Discrete control")
    lim = max(grey.max(), arx.max()) * 1.08
    ax.plot([0, lim], [0, lim], ls="--", color=GREY, lw=.8)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Grey-box recursive NRMSE",
           ylabel="ARX recursive NRMSE")
    ax.set_title("Continuous structure improves recursive prediction", loc="left", fontsize=7.5)
    ax.legend(frameon=False, loc="lower right")

    for label, ax in zip("abcd", axs.flat):
        ax.text(-.15, 1.07, label, transform=ax.transAxes, weight="bold",
                fontsize=9, va="top")
        if ax.axison:
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(width=.7, length=3)
    for ext, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600}),
                        ("tiff", {"dpi": 600})):
        fig.savefig(OUT / f"Fig14_openmct_greybox_bridge.{ext}", **kwargs)


if __name__ == "__main__":
    main()
