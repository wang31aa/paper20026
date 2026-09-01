#!/usr/bin/env python3
"""Build Figure 10 from frozen cross-domain CPS benchmark outputs."""
from pathlib import Path
import string

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "cps_transfer_benchmark" / "results"
OUT = Path(__file__).resolve().parent
BLUE, ORANGE, GREEN, RED, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#747980"
mpl.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.titlesize": 9,
                     "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 6.7, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "svg.fonttype": "none", "svg.hashsalt": "nature-tac2023-v7",
                     "savefig.dpi": 600,
                     "savefig.bbox": "tight"})


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D9DDE2", lw=.5, alpha=.7)
    ax.set_axisbelow(True)


def build():
    runs = pd.read_csv(RES / "rths_run_metrics.csv")
    faults = pd.read_csv(RES / "rths_fault_metrics.csv")
    water = pd.read_csv(RES / "water_hil_session_metrics.csv")
    if len(runs) != 14 or len(faults) != 224 or len(water) not in (3, 4):
        raise ValueError("Unexpected frozen benchmark dimensions")
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.8))

    a = ax[0, 0]
    a.axis("off")
    boxes = [(0.08, .58, .30, .18, "Domain\nrecords", BLUE),
             (.62, .58, .30, .18, "Domain\npredictor", ORANGE),
             (0.08, .22, .30, .18, "Normal\ncalibration", GREY),
             (.57, .20, .38, .22, "Calibrated\ndimensionless\nresidual", GREEN)]
    for x, y, w, h, label, color in boxes:
        a.add_patch(plt.Rectangle((x, y), w, h, transform=a.transAxes, fc="white", ec=color, lw=1.4))
        a.text(x+w/2, y+h/2, label, transform=a.transAxes, ha="center", va="center", fontsize=7.0)
    for p, q in [((.38,.67),(.62,.67)), ((.77,.58),(.77,.42)), ((.38,.31),(.57,.31))]:
        a.annotate("", xy=q, xytext=p, xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1, color="#30343B"))
    a.text(.5, .96, "Domain-specific residual analysis", transform=a.transAxes,
           ha="center", va="top", weight="bold")
    a.text(.5, .04, "Offline measurement analysis", transform=a.transAxes,
           ha="center", color=RED, fontsize=7.0)

    b = ax[0, 1]
    x = np.arange(len(runs))
    for xi, p, m in zip(x, runs.persistence_nrmse, runs.model_nrmse):
        b.plot([xi-.10, xi+.10], [p, m], color="#C8CDD2", lw=.7, zorder=0)
    b.scatter(x-.10, runs.persistence_nrmse, color=GREY, s=15, label="persistence")
    b.scatter(x+.10, runs.model_nrmse, color=BLUE, s=15, label="lagged ridge")
    b.set_xticks(x, [f"{r.run_id}.{r.actuator}" for r in runs.itertuples()], rotation=55, ha="right")
    b.set_ylabel("Force NRMSE")
    b.set_title("RTHS: cross-configuration prediction")
    b.legend(frameon=False)
    clean(b)

    c = ax[1, 0]
    g = faults.groupby(["fault", "level"]).auprc.agg(["median", lambda x: x.quantile(.25), lambda x: x.quantile(.75)]).reset_index()
    g.columns = ["fault", "level", "median", "q25", "q75"]
    assert (g[["median", "q25", "q75"]] > 0).all().all()
    styles = [("force_bias", ORANGE, "o", "bias"),
              ("force_gain", BLUE, "^", "gain"),
              ("force_dropout", RED, "D", "dropout"),
              ("force_noise", GREEN, "s", "noise")]
    for fault, color, marker, label in styles:
        z = g[g.fault == fault]
        c.plot(z.level, z["median"], marker=marker, color=color, lw=1.4, label=label)
        c.fill_between(z.level, z.q25, z.q75, color=color, alpha=.14, linewidth=0)
    shared_prev = faults.loc[faults.fault != "force_dropout"].positive_prevalence.median()
    drop_prev = faults.loc[faults.fault == "force_dropout"].groupby("level").positive_prevalence.median()
    c.axhline(shared_prev, color=GREY, lw=.8, ls="--", label="prevalence AP: window faults")
    c.plot(drop_prev.index, drop_prev.values, color=RED, lw=.8, ls=":", label="prevalence AP: dropout")
    c.set_xticks([1,2,3,4])
    c.set_xlabel("Synthetic replay severity level")
    c.set_ylabel("Median AUPRC")
    # The asserted AUPRC and residual ratios are strictly positive before log rendering.
    c.set_yscale("log")
    c.set_ylim(.001, 1)
    c.set_title("RTHS: severity response is fault-dependent")
    c.legend(frameon=False, ncol=2, fontsize=6.2, loc="upper center",
             bbox_to_anchor=(.5, -.28))
    clean(c)

    d = ax[1, 1]
    labels = [s.replace("_spoof", "").replace("_", "\n") for s in water.condition]
    colors = [BLUE if v > 1 else GREY for v in water.median_ratio_to_normal]
    assert (water.median_ratio_to_normal > 0).all()
    d.bar(np.arange(len(water)), water.median_ratio_to_normal, color=colors, width=.65)
    d.axhline(1, color=RED, ls="--", lw=1, label="normal-session median")
    d.set_xticks(np.arange(len(water)), labels)
    d.set_ylabel("Median residual / normal median")
    d.set_yscale("log")
    d.set_title("Water HIL: two-output residuals")
    # The dashed reference is defined in the caption; no text is placed over bars.
    clean(d)

    for letter, a0 in zip(string.ascii_lowercase, ax.ravel()):
        a0.text(-.13, 1.10, letter, transform=a0.transAxes, weight="bold", fontsize=10, va="top")
    fig.subplots_adjust(wspace=.34, hspace=.55)
    compact_flow_text = {"Domain\nrecords", "Domain\npredictor", "Normal\ncalibration",
                         "Calibrated\ndimensionless\nresidual",
                         "Offline measurement analysis"}
    for item in fig.findobj(match=Text):
        # Log-axis exponent glyphs are drawn at 70% of the parent size; 8.6 pt
        # keeps those glyphs above the 6-pt final-size gate. Flow labels are
        # ordinary glyphs and retain a verified 7-pt compact layout.
        item.set_fontsize(7.0 if item.get_text() in compact_flow_text
                          else max(item.get_fontsize(), 8.6))
    fig.savefig(OUT / "Fig10_cps_transfer.svg", metadata={"Date": None})
    fig.savefig(OUT / "Fig10_cps_transfer.pdf",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / "Fig10_cps_transfer.png", dpi=600)
    fig.savefig(OUT / "Fig10_cps_transfer.tiff", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    build()
