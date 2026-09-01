#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd

R = Path(__file__).resolve().parent
d = pd.read_csv(R / "results/V46_PAIRED_RESULTS.csv")
g = d.groupby("policy", as_index=False).agg(
    runs=("task_success", "size"), successes=("task_success", "sum"),
    success_rate=("task_success", "mean"), median_margin_m=("minimum_margin_m", "median"),
    median_tail_rmse_mps=("tail_tracking_rmse_mps", "median"),
    median_energy=("control_energy", "median"), median_messages=("communication_messages", "median"),
    median_observer_rmse_mps=("observer_rmse_mps", "median"),
    median_saturation_fraction=("saturation_fraction", "median"),
)
g.to_csv(R / "results/V46_POLICY_SUMMARY.csv", index=False)
best = g.sort_values(["success_rate", "median_energy"], ascending=[False, True]).iloc[0]
report = {
    "best_success_policy": best["policy"],
    "best_success_rate": float(best["success_rate"]),
    "two_layer_success_rate": float(g.loc[g.policy == "two_layer", "success_rate"].iloc[0]),
    "all_coupled_success_rate": float(g.loc[g.policy == "all_coupled", "success_rate"].iloc[0]),
    "interpretation": "vehicle-domain physical filtering is stronger than participation gating; no universal gate superiority claim",
}
(R / "results/V46_ANALYSIS.json").write_text(json.dumps(report, indent=2) + "\n")
print(g.to_string(index=False))
