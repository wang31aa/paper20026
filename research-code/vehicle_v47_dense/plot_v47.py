#!/usr/bin/env python3
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
d = json.loads((OUT / "V47_HELDOUT_EVALUATION.json").read_text())
styles = {
    "all_coupled": ("#2878B5", "Permanent coupling"),
    "two_layer": ("#2A9D8F", "Two-layer policy"),
    "one_step_barrier_filter": ("#E76F51", "Physical barrier")
}
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for key, (colour, label) in styles.items():
    curve = d["policies"][key]["heldout_curve"]
    ax.plot([x["rho"] for x in curve], [x["success_rate"] for x in curve],
            marker="o", lw=2, ms=4, color=colour, label=label)
ax.axhline(0.8, color="0.25", ls="--", lw=1, label="Prespecified feasibility threshold")
ax.set(xlabel="Participation strength, $\\rho$", ylabel="Held-out task success fraction",
       xlim=(0.18, 1.02), ylim=(-0.02, 1.03))
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "Fig_V47_dense_vehicle_response.pdf")
fig.savefig(OUT / "Fig_V47_dense_vehicle_response.png", dpi=300)
