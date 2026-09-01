from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "observer_rebuild" / "results"
OUT = Path(__file__).resolve().parent

plt.rcParams.update({"font.family":"sans-serif",
                     "font.sans-serif":["Arial","Helvetica","DejaVu Sans","sans-serif"],
                     "font.size":8,"axes.linewidth":0.8,
                     "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none",
                     "svg.hashsalt":"nature-tac2023-v7"})
blue, green, orange, purple = "#0072B2", "#009E73", "#D55E00", "#6A3D9A"
d = pd.read_csv(RES / "raw_runs.csv")
z = np.load(RES / "raw_trajectories.npz")
dt = pd.read_csv(RES / "dt_convergence.csv")
assert len(d) == 20 and d.finite.all()

fig, ax = plt.subplots(1, 3, figsize=(7.2, 3.25), constrained_layout=True)
metrics = ["final_observer_state_error", "final_model_error",
           "final_mismatch_estimation_error", "final_tracking_error"]
labels = ["Target state", "Target model", "Mismatch", "Tracking"]
vals = [d[m].to_numpy() for m in metrics]
assert all(np.all(v > 0) for v in vals), "log-scale errors must be positive"
bp = ax[0].boxplot(vals, tick_labels=labels, patch_artist=True, showfliers=False)
for p,c in zip(bp["boxes"],[blue,green,orange,purple]): p.set_facecolor(c); p.set_alpha(.7)
for i,v in enumerate(vals,1): ax[0].scatter(np.full(len(v),i)+np.linspace(-.08,.08,len(v)),v,s=8,c="k",alpha=.35)
# The asserted source metrics are strictly positive before log-axis rendering.
ax[0].set_yscale("log"); ax[0].set_ylabel("Final error norm")
ax[0].set_title("20 seeds; all transients retained")
ax[0].tick_params(axis="x", labelrotation=22, labelsize=7)

t=z["seed_0_time"]; m=z["seed_0_metrics"]
assert np.all(m > 0), "log-scale trajectories must be positive"
for j,(lab,c) in enumerate(zip(labels,[blue,green,orange,purple])):
    ax[1].plot(t,m[:,j],label=lab,color=c,lw=1.25)
ax[1].set_yscale("log"); ax[1].set_xlabel("Time"); ax[1].set_ylabel("Error norm")
ax[1].set_title("Complete trajectory (seed 0)")
ax[1].legend(frameon=False,fontsize=7,ncol=1,loc="upper right")

for seed,g in dt.groupby("seed"):
    g=g.sort_values("dt",ascending=False)
    ax[2].plot(g.dt,g.final_tracking_error,"o-",ms=3,lw=.8,alpha=.7)
ax[2].set_xscale("log"); ax[2].invert_xaxis(); ax[2].set_xlabel("RK4 step size")
ax[2].set_ylabel("Final tracking error"); ax[2].set_title("Matched-seed step convergence")

for a,l in zip(ax.flat,"abc"):
    a.text(-.18,1.13,l,transform=a.transAxes,fontsize=12,weight="bold",va="top")
for item in fig.findobj(match=Text):
    item.set_fontsize(max(item.get_fontsize(), 8.6))
fig.savefig(OUT/"Fig5_observer_rebuild.svg", bbox_inches="tight", metadata={"Date": None})
fig.savefig(OUT/"Fig5_observer_rebuild.pdf", bbox_inches="tight",
            metadata={"CreationDate": None, "ModDate": None})
fig.savefig(OUT/"Fig5_observer_rebuild.png", dpi=600, bbox_inches="tight")
fig.savefig(OUT/"Fig5_observer_rebuild.tiff", dpi=600, bbox_inches="tight")
plt.close(fig)
