#!/usr/bin/env python3
"""Build Figure 11 from frozen OpenMCT physical motor results."""
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
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#E69F00", "#009E73", "#7A7F86"
mpl.rcParams.update({"font.family":"sans-serif", "font.size":8, "axes.titlesize":9,
                     "axes.labelsize":8, "xtick.labelsize":7, "ytick.labelsize":7,
                     "legend.fontsize":6.5, "pdf.fonttype":42, "ps.fonttype":42,
                     "svg.fonttype":"none", "svg.hashsalt":"nature-tac2023-v7",
                     "savefig.dpi":600,
                     "savefig.bbox":"tight"})


def clean(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", color="#D9DDE2", lw=.5, alpha=.7)
    ax.set_axisbelow(True)


def build():
    run = pd.read_csv(RES / "openmct_run_metrics.csv")
    rep = pd.read_csv(RES / "openmct_representative_10ms.csv")
    if len(run) != 7 or len(rep) < 400:
        raise ValueError("unexpected frozen OpenMCT dimensions")
    labels = [f"PI {r.condition} ms" if r.family == "continuous_PI" else f"Discrete {r.condition}"
              for r in run.itertuples()]
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.45))

    a = ax[0,0]; y = np.arange(len(run))
    for yi, p, m in zip(y, run.persistence_nrmse, run.model_nrmse):
        a.plot([p,m], [yi,yi], color="#CCD1D6", lw=1.1)
    a.scatter(run.persistence_nrmse, y, color=GREY, s=22, label="speed persistence", zorder=3)
    a.scatter(run.model_nrmse, y, color=BLUE, s=22, label="APRBS-trained ARX", zorder=3)
    a.set_yticks(y, labels); a.invert_yaxis(); a.set_xlabel("Held-out speed-prediction NRMSE")
    a.set_xticks([.05, .10, .15, .20], ["0.05", "0.10", "0.15", "0.20"])
    a.set_title("Seven held-out controller records"); a.legend(frameon=False); clean(a)

    b = ax[0,1]
    b.plot(rep.time_s, rep.reference_rpm, color=GREY, lw=1, ls="--", label="reference")
    b.plot(rep.time_s, rep.measured_rpm, color=BLUE, lw=1, label="measured speed")
    b.set_xlabel("Time (s)"); b.set_ylabel("Speed (r.p.m.)")
    b.set_title("Representative 10-ms PI experiment"); b.legend(frameon=False); clean(b)

    c = ax[1,0]
    ep = np.abs(rep.measured_rpm-rep.persistence_prediction_rpm)
    em = np.abs(rep.measured_rpm-rep.model_prediction_rpm)
    window = 9
    c.plot(rep.time_s, ep.rolling(window, center=True, min_periods=1).median(), color=GREY, lw=1,
           label="persistence error")
    c.plot(rep.time_s, em.rolling(window, center=True, min_periods=1).median(), color=BLUE, lw=1,
           label="ARX error")
    c.set_xlabel("Time (s)"); c.set_ylabel("Absolute prediction error\n(rolling median, r.p.m.)")
    c.set_title("One-step residuals without test refitting")
    c.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.25), ncol=2)
    clean(c)

    d = ax[1,1]
    pi = run[run.family == "continuous_PI"].copy(); pi["ms"] = pi.condition.astype(float); pi = pi.sort_values("ms")
    assert (pi.ms > 0).all(), "log-scale controller intervals must be positive"
    d.plot(pi.ms, pi.tracking_nrmse, color=ORANGE, marker="o", lw=1.3, label="full record")
    d.plot(pi.ms, pi.tail20_tracking_nrmse, color=GREEN, marker="s", lw=1.3, label="last 20%")
    # The asserted controller intervals are strictly positive before log rendering.
    d.set_xscale("log"); d.set_xticks([5,10,20,50], ["5","10","20","50"])
    d.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    d.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    d.set_xlabel("Nominal controller interval (ms)"); d.set_ylabel("Reference–speed NRMSE")
    d.set_title("Tracking varies with controller interval"); d.legend(frameon=False); clean(d)

    for letter, a0 in zip(string.ascii_lowercase, ax.ravel()):
        a0.text(-.18, 1.15, letter, transform=a0.transAxes, weight="bold", fontsize=10, va="top")
    fig.subplots_adjust(wspace=.38, hspace=.68, bottom=.14)
    for item in fig.findobj(match=Text):
        item.set_fontsize(max(item.get_fontsize(), 8.6))
    fig.savefig(OUT / "Fig11_openmct_motor.svg", metadata={"Date": None})
    fig.savefig(OUT / "Fig11_openmct_motor.pdf",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / "Fig11_openmct_motor.png", dpi=600)
    fig.savefig(OUT / "Fig11_openmct_motor.tiff", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    build()
