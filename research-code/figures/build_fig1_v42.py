#!/usr/bin/env python3
"""Build the theory-first discovery figure from frozen V24 and V42 data."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"


def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return np.nan, np.nan
    p = successes / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


v24 = rows(ROOT / "cross_domain_v24/results/v24_runs.csv")
v42 = rows(ROOT / "cross_domain_v42/results/v42_all_policy_runs.csv")
heldout = [r for r in v42 if r["split"] == "heldout"]
v41_summary = rows(ROOT / "cross_domain_v41/results/v41_heldout_summary.csv")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

domain_colors = {
    "uav6dof": "#0072B2", "vehicle": "#D55E00", "motor": "#009E73",
    "robot": "#56B4E9", "microgrid": "#E69F00", "circuit": "#CC79A7",
    "water": "#7F7F7F", "structure": "#6A3D9A",
}
domain_labels = {
    "uav6dof": "UAV", "vehicle": "Vehicle", "motor": "Motor",
    "robot": "Robot", "microgrid": "Microgrid", "circuit": "Circuit",
    "water": "Water", "structure": "Structure",
}

fig, axs = plt.subplots(2, 2, figsize=(11.2, 6.2), constrained_layout=True)

# a: two-layer principle with distinct qualifications
ax = axs[0, 0]
ax.set_axis_off()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
box = dict(edgecolor="#333333", linewidth=0.8)
ax.add_patch(plt.Rectangle((0.04, 0.64), 0.28, 0.22, facecolor="#DCEAF7", **box))
ax.add_patch(plt.Rectangle((0.36, 0.64), 0.28, 0.22, facecolor="#FBE5D5", **box))
ax.add_patch(plt.Rectangle((0.68, 0.64), 0.28, 0.22, facecolor="#DFF0E6", **box))
ax.text(0.18, 0.75, "Target-rooted\ninformation graph", ha="center", va="center", weight="bold")
ax.text(0.50, 0.75, "Admissible physical\ninfluence graph", ha="center", va="center", weight="bold")
ax.text(0.82, 0.75, "Task-feasible\nviability kernel", ha="center", va="center", weight="bold")
for x0, x1 in ((0.32, 0.36), (0.64, 0.68)):
    ax.annotate("", xy=(x1, 0.75), xytext=(x0, 0.75), arrowprops={"arrowstyle": "->", "lw": 1.1})
ax.text(0.18, 0.42, "Information loss\nwhen participation is weak", ha="center", color="#0072B2")
ax.text(0.50, 0.42, "Mismatch propagation\nwhen exposure dominates", ha="center", color="#D55E00")
ax.text(0.82, 0.42, "Select an action only if\nits capture basin contains x", ha="center", color="#009E73")
ax.text(0.50, 0.13, "Reachability alone is neither physical feasibility nor recovery", ha="center", weight="bold")
ax.set_title("a  Information and physical influence are distinct resources", loc="left", weight="bold")

# b: three observed shapes under the corrected switching experiment
ax = axs[0, 1]
for domain in ("uav6dof", "vehicle", "motor"):
    rhos = sorted({float(r["rho"]) for r in v24 if r["domain"] == domain})
    y, lo, hi, counts = [], [], [], []
    for rho in rhos:
        rr = [r for r in v24 if r["domain"] == domain and r["policy"] == "all_coupled"
              and r["topology"] == "certified_switching" and r["split"] == "heldout"
              and abs(float(r["rho"]) - rho) < 1e-12]
        s = sum(int(r["task_success"]) for r in rr)
        a, b = wilson(s, len(rr))
        y.append(s / len(rr)); lo.append(a); hi.append(b); counts.append(len(rr))
    ax.fill_between(rhos, lo, hi, color=domain_colors[domain], alpha=0.12, linewidth=0)
    ax.plot(rhos, y, marker="o", linewidth=1.5, markersize=3.8,
            color=domain_colors[domain], label=domain_labels[domain])
ax.set(xlabel="Participation strength, ρ", ylabel="Held-out task success", ylim=(-0.04, 1.04))
ax.grid(alpha=0.18)
ax.legend(frameon=False, ncol=3, loc="upper center")
ax.set_title("b  Windowed, monotone and plateau responses are all admissible", loc="left", weight="bold")

# c: actual switching common-metric margins
ax = axs[1, 0]
keys, values, colors = [], [], []
for domain in ("uav6dof", "vehicle", "motor"):
    for n in (5, 20):
        rr = [float(r["common_metric_mu"]) for r in v24 if r["domain"] == domain
              and int(r["n"]) == n and r["topology"] == "certified_switching"]
        keys.append(f"{domain_labels[domain]}\nN={n}")
        values.append(min(rr))
        colors.append(domain_colors[domain])
bars = ax.bar(np.arange(len(values)), values, color=colors, width=0.68)
ax.axhline(0, color="#333333", linewidth=0.8)
ax.set_xticks(np.arange(len(values)), keys)
ax.set_ylabel("Minimum common-metric margin")
ax.grid(axis="y", alpha=0.18)
for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.3f}", ha="center", fontsize=7)
ax.set_title("c  Switching qualification is outcome independent", loc="left", weight="bold")

# d: V42 frozen supervisor versus the best fixed candidate in each domain
ax = axs[1, 1]
domains = list(domain_labels)
fixed = ["all_coupled", "global_gain_reduction", "independent_tracking",
         "connectivity_gate", "physical_filter", "two_layer_gate"]
supervisor_rates, best_rates = [], []
best_names = []
for domain in domains:
    def rate(policy):
        rr = [r for r in v41_summary if r["domain"] == domain and r["policy"] == policy]
        if len(rr) != 1:
            raise ValueError(f"expected one V41 summary row for {domain}/{policy}, found {len(rr)}")
        return float(rr[0]["success_rate"])
    supervisor_rates.append(rate("supervisor"))
    candidates = [(rate(policy), policy) for policy in fixed]
    best_rate, best_name = max(candidates)
    best_rates.append(best_rate)
    best_names.append(best_name)
x = np.arange(len(domains)); width = 0.36
ax.bar(x - width / 2, best_rates, width, color="#B8C2CC", label="Best fixed candidate")
ax.bar(x + width / 2, supervisor_rates, width,
       color=[domain_colors[d] for d in domains], label="Frozen supervisor")
ax.set_xticks(x, [domain_labels[d] for d in domains], rotation=28, ha="right")
ax.set_ylabel("Held-out task success")
ax.set_ylim(0, 1.06)
ax.grid(axis="y", alpha=0.18)
ax.legend(frameon=False, ncol=2, loc="upper center")
ax.text(6, 0.055, "No viable\ncandidate", ha="center", va="bottom", fontsize=7, color="#555555")
ax.set_title("d  Recovery transfers as selection, not as one fixed action", loc="left", weight="bold")

for ext in ("pdf", "svg", "png"):
    fig.savefig(OUT / f"Fig1_v42_capability_discovery.{ext}", dpi=300 if ext == "png" else None,
                bbox_inches="tight")
plt.close(fig)

with (OUT / "Fig1_v42_source_data.csv").open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["panel", "domain", "condition", "value", "n", "source"])
    for domain in ("uav6dof", "vehicle", "motor"):
        for rho in sorted({float(r["rho"]) for r in v24 if r["domain"] == domain}):
            rr = [r for r in v24 if r["domain"] == domain and r["policy"] == "all_coupled"
                  and r["topology"] == "certified_switching" and r["split"] == "heldout"
                  and abs(float(r["rho"]) - rho) < 1e-12]
            writer.writerow(["b", domain, rho, sum(int(r["task_success"]) for r in rr) / len(rr),
                             len(rr), "cross_domain_v24/results/v24_runs.csv"])
    for key, value in zip(keys, values):
        writer.writerow(["c", key, "minimum_common_metric_margin", value, "", "v24_runs.csv"])
    for domain, sup, best, name in zip(domains, supervisor_rates, best_rates, best_names):
        writer.writerow(["d", domain, "supervisor", sup, 70, "cross_domain_v41/results/v41_heldout_summary.csv"])
        writer.writerow(["d", domain, f"best_fixed:{name}", best, 70, "cross_domain_v41/results/v41_heldout_summary.csv"])
