#!/usr/bin/env python3
"""Build Figure 9 from frozen MATLAB-reproduction audit artifacts.

This script is visualization-only. It reads existing CSV/JSON artifacts and
does not execute, import, or modify either the legacy or intended simulators.
All failures are retained and labelled.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import string

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "matlab_reproduction"
OUT = Path(__file__).resolve().parent

# Match Figures 1--8: compact sans-serif type and Okabe--Ito accents.
INK = "#202124"
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#8A8F98"
LIGHT = "#F3F5F7"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.7,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": "nature-tac2023-v7",
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def clean(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color="#D9DDE2", lw=0.5, alpha=0.65)
    ax.set_axisbelow(True)


def panel_letters(axes: np.ndarray) -> None:
    for letter, ax in zip(string.ascii_lowercase, np.ravel(axes)):
        ax.text(-0.12, 1.13, letter, transform=ax.transAxes,
                fontweight="bold", fontsize=10, va="top", ha="left")


def read_frozen_tolerances() -> tuple[float, float]:
    text = (AUDIT / "PREREGISTRATION.md").read_text(encoding="utf-8")
    early = re.search(r"First 100 steps: maximum absolute error `<=([0-9.eE+-]+)`", text)
    full = re.search(r"maximum absolute error `<=([0-9.eE+-]+)`", text[text.index("Full bounded trajectory"):])
    if early is None or full is None:
        raise RuntimeError("Could not recover frozen absolute-error tolerances")
    return float(early.group(1)), float(full.group(1))


def validate_inputs() -> dict[str, object]:
    paths = {
        "g1_checkpoints": AUDIT / "results/graph1_full/raw_checkpoints.csv",
        "g1_summary": AUDIT / "results/graph1_full/parity_summary.json",
        "g2_heldout": AUDIT / "results/graph2_heldout/pointwise_errors.csv",
        "g2_heldout_summary": AUDIT / "results/graph2_heldout/parity_summary.json",
        "g2_posthoc": AUDIT / "posthoc_correction/graph2_full/pointwise_errors.csv",
        "g2_posthoc_summary": AUDIT / "posthoc_correction/graph2_full/parity_summary.json",
        "g1_ab": AUDIT / "phase2/graph1_summary/legacy_ab.csv",
        "g2_ab": AUDIT / "phase2/graph2_summary/legacy_ab.csv",
        "g1_metrics": AUDIT / "phase2/graph1_summary/run_metrics.csv",
        "g2_metrics": AUDIT / "phase2/graph2_summary/run_metrics.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen Figure 9 inputs: " + ", ".join(missing))

    g1_checkpoints = pd.read_csv(paths["g1_checkpoints"])
    heldout = pd.read_csv(paths["g2_heldout"])
    posthoc = pd.read_csv(paths["g2_posthoc"])
    ab = pd.concat([pd.read_csv(paths["g1_ab"]), pd.read_csv(paths["g2_ab"])], ignore_index=True)
    metrics = pd.concat([pd.read_csv(paths["g1_metrics"]), pd.read_csv(paths["g2_metrics"])], ignore_index=True)
    with open(paths["g1_summary"], encoding="utf-8") as f:
        g1_summary = json.load(f)
    with open(paths["g2_heldout_summary"], encoding="utf-8") as f:
        heldout_summary = json.load(f)
    with open(paths["g2_posthoc_summary"], encoding="utf-8") as f:
        posthoc_summary = json.load(f)

    required = {
        "checkpoints": ({"variable", "time_index", "max_abs"}, g1_checkpoints.columns),
        "pointwise heldout": ({"variable", "max_abs", "full_pass"}, heldout.columns),
        "pointwise posthoc": ({"variable", "max_abs", "full_pass"}, posthoc.columns),
        "legacy A/B": ({"graph", "variable", "max_abs", "rms"}, ab.columns),
        "grid metrics": ({"graph", "h", "finite", "recon_tail_rms"}, metrics.columns),
    }
    for name, (needed, columns) in required.items():
        absent = needed - set(columns)
        if absent:
            raise ValueError(f"{name} lacks columns: {sorted(absent)}")
    if set(heldout.variable) != set(posthoc.variable):
        raise ValueError("Held-out and post-hoc variable groups do not match")
    if len(heldout) != 37 or len(posthoc) != 37:
        raise ValueError("Expected exactly 37 Graph 2 variable groups")
    if set(metrics.graph) != {"graph1", "graph2"} or set(metrics.h) != {0.001, 0.0005, 0.00025}:
        raise ValueError("Expected both graphs on all three frozen grids")
    if not metrics.finite.all():
        raise ValueError("A frozen Phase 2 run is non-finite; preserve it explicitly before plotting")

    return {
        "g1_checkpoints": g1_checkpoints,
        "g1_summary": g1_summary,
        "heldout": heldout,
        "heldout_summary": heldout_summary,
        "posthoc": posthoc,
        "posthoc_summary": posthoc_summary,
        "ab": ab,
        "metrics": metrics,
    }


def build() -> None:
    data = validate_inputs()
    early_tol, full_tol = read_frozen_tolerances()
    g1 = data["g1_checkpoints"]
    heldout = data["heldout"].set_index("variable")
    posthoc = data["posthoc"].set_index("variable").loc[heldout.index]
    ab = data["ab"]
    metrics = data["metrics"]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.45))

    # a | Graph 1 checkpoint parity. Zeros receive a data-derived display floor only.
    ax = axes[0, 0]
    checkpoint = g1.groupby("time_index", as_index=False).max_abs.max().sort_values("time_index")
    positive = checkpoint.loc[checkpoint.max_abs > 0, "max_abs"]
    floor = positive.min() / 10
    shown = checkpoint.max_abs.mask(checkpoint.max_abs == 0, floor)
    ax.plot(checkpoint.time_index, shown, color=BLUE, marker="o", ms=3.3, lw=1.25,
            label="maximum across 37 groups")
    ax.axhline(full_tol, color=VERMILION, ls="--", lw=1, label="frozen full-horizon limit")
    ax.plot([checkpoint.time_index.min(), 100], [early_tol, early_tol], color=ORANGE,
            ls=":", lw=1.2, label="frozen first-100-step limit")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("log")
    ax.set_xlabel("Archived checkpoint index")
    ax.set_ylabel("Maximum absolute discrepancy")
    ax.set_title("Graph 1: legacy MATLAB–Python parity")
    ax.legend(frameon=False, loc="upper left")
    clean(ax)
    summary = data["g1_summary"]
    n_groups = int(summary["variables_compared"])
    ax.text(.98, .06, f"{n_groups}/{n_groups} groups pass\nfull frozen gate",
            transform=ax.transAxes, ha="right", va="bottom", color=GREEN, weight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=1.5))
    ax.text(.02, .02, "Zero discrepancies shown at a data-derived display floor.",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.0, color="#5A5E64")

    # b | Held-out failure and separately labelled post-hoc correction.
    ax = axes[0, 1]
    order = np.argsort(np.maximum(heldout.max_abs.to_numpy(), posthoc.max_abs.to_numpy()))
    labels = heldout.index.to_numpy()[order]
    x = np.arange(len(labels))
    hvals = heldout.loc[labels, "max_abs"].to_numpy()
    pvals = posthoc.loc[labels, "max_abs"].to_numpy()
    ax.scatter(x, hvals, s=16, color=VERMILION, marker="o", label="preregistered held-out", zorder=3)
    ax.scatter(x, pvals, s=17, color=BLUE, marker="D", label="post-hoc one-line correction", zorder=3)
    ax.axhline(full_tol, color=INK, ls="--", lw=.9, label="frozen full-horizon limit")
    ax.set_yscale("log")
    ax.set_xlim(-1, len(labels))
    ax.set_xticks([])
    ax.set_xlabel(f"{len(labels)} state, reconstruction and model-matrix groups (sorted)")
    ax.set_ylabel("Full-horizon maximum discrepancy")
    ax.set_title("Graph 2: held-out boundary remains negative")
    clean(ax)
    hfail = int((~heldout.full_pass).sum())
    pfail = int((~posthoc.full_pass).sum())
    ax.text(.02, .96, f"Held-out: {hfail}/{len(labels)} fail\nPost-hoc: {pfail}/{len(labels)} fail",
            transform=ax.transAxes, va="top", ha="left", color=VERMILION, weight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=1.5))
    ax.legend(frameon=False, loc="lower right", fontsize=6.1)
    # Label the five reconstruction groups that still fail after correction.
    offsets = [(0, 7), (-13, 8), (10, 7), (-12, -12), (12, -12)]
    for variable, offset in zip(posthoc.index[~posthoc.full_pass], offsets):
        j = int(np.flatnonzero(labels == variable)[0])
        ax.annotate(variable, (j, posthoc.loc[variable, "max_abs"]), xytext=offset,
                    textcoords="offset points", ha="center", fontsize=6.0, color=BLUE)

    # c | Intended continuous equations are not interchangeable with the legacy run.
    ax = axes[1, 0]
    err = ab.loc[ab.variable == "Error"].set_index("graph").loc[["graph1", "graph2"]]
    x = np.arange(2)
    width = .32
    ax.bar(x - width/2, err.max_abs, width, color=VERMILION, label="maximum absolute difference")
    ax.bar(x + width/2, err.rms, width, color=ORANGE, label="RMS difference")
    ax.set_yscale("log")
    ax.set_xticks(x, ["Graph 1", "Graph 2"])
    ax.set_ylabel("Legacy–intended aggregate-error difference")
    ax.set_title("Legacy and intended formulations diverge")
    clean(ax)
    ax.legend(frameon=False, loc="upper left")
    ax.text(.98, .55, "Not interchangeable evidence",
            transform=ax.transAxes, ha="right", va="center", color=VERMILION, weight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=.88, pad=1.5))

    # d | Three-grid reconstruction audit: Graph 2 worsens under refinement.
    ax = axes[1, 1]
    style = {
        "graph1": (BLUE, "o", "Graph 1"),
        "graph2": (VERMILION, "s", "Graph 2"),
    }
    graph2_sequence = None
    for graph in ["graph1", "graph2"]:
        group = metrics.loc[metrics.graph == graph].sort_values("h", ascending=False)
        color, marker, label = style[graph]
        ax.plot(np.arange(3), group.recon_tail_rms, color=color, marker=marker,
                ms=4, lw=1.35, label=label)
        if graph == "graph1":
            for j, value in enumerate(group.recon_tail_rms):
                ax.text(j, value * 1.35, f"{value:.2g}", ha="center", va="center",
                        fontsize=6.0, color=color)
        else:
            graph2_sequence = group.recon_tail_rms.to_numpy()
    ax.set_yscale("log")
    ax.set_xticks([0, 1, 2], [r"$10^{-3}$", r"$5\times10^{-4}$", r"$2.5\times10^{-4}$"])
    ax.set_xlabel("RK4 step size, h  (refinement →)")
    ax.set_ylabel("Tail RMS reconstruction error")
    ax.set_title("Three-grid stage-consistent audit (archived gains)")
    clean(ax)
    ax.legend(frameon=False, loc="center left")
    graph2_text = " → ".join(f"{value:.3f}" for value in graph2_sequence)
    ax.text(.98, .96, f"Graph 2 worsens:\n{graph2_text}",
            transform=ax.transAxes, ha="right", va="top", color=VERMILION, weight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=1.5))
    ax.text(.98, .34, "Negative/nonconvergent result retained;\nno convergence order claimed.",
            transform=ax.transAxes, ha="right", va="center", fontsize=6, color="#5A5E64",
            bbox=dict(facecolor="white", edgecolor="none", alpha=.86, pad=1.2))

    panel_letters(axes)
    fig.subplots_adjust(left=.09, right=.985, bottom=.105, top=.91, hspace=.45, wspace=.34)
    for item in fig.findobj(match=Text):
        item.set_fontsize(max(item.get_fontsize(), 8.6))
    fig.savefig(OUT / "Fig9_matlab_reproduction.svg", metadata={"Date": None})
    fig.savefig(OUT / "Fig9_matlab_reproduction.pdf",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / "Fig9_matlab_reproduction.png", dpi=600)
    fig.savefig(OUT / "Fig9_matlab_reproduction.tiff", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    build()
