#!/usr/bin/env python3
"""Development-only V3 calibration with clustered percentile bounds."""
from __future__ import annotations
import csv, json, math
from pathlib import Path
import numpy as np
from uav_v3_engine import P, simulate

HERE = Path(__file__).resolve().parent; OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
rhos = np.asarray(P["rho_grid"], float); rows = []
for env in P["development_environments"]:
    for seed in P["development_seeds"]:
        for rho in rhos:
            for policy, arm in (("all_coupled_with_trim_sharing", "all_heterogeneous"),
                                ("all_coupled_without_trim_sharing", "all_heterogeneous_without_feedforward_sharing")):
                rows.append(simulate(policy, float(rho), env, int(seed), arm))
with (OUT / "uav_v3_development_runs.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

sharing = [r for r in rows if r["policy"] == "all_coupled_with_trim_sharing"]
baseline = [r for r in rows if r["policy"] == "all_coupled_without_trim_sharing"]
by_key = {(r["environment"], r["seed"], r["rho"]): r for r in baseline}

# A conservative dominant-mode decay proxy is estimated from the last 20% tube
# relative to the all-time excursion.  It is a computational-model parameter,
# not a physical UAV identification claim.
a_samples = []
for r in sharing:
    ratio = (r["peak_tracking_error_m"] + 1e-6) / (r["tail_tracking_error_m"] + 1e-6)
    a_samples.append(max(.02, math.log(max(ratio, 1.000001)) / (0.8 * P["duration_s"])))
a_lower = float(np.quantile(a_samples, .05))

# Intrinsic and transmitted loads use paired sharing/no-sharing runs.  The
# upper support slope is fitted without an intercept after removing the paired
# no-sharing tail; negative differences are retained as zero evidence.
d0_samples = [a_lower * r["tail_tracking_error_m"] for r in baseline]
db_samples = []
for r in sharing:
    b = by_key[(r["environment"], r["seed"], r["rho"])]
    db_samples.append(max(a_lower * (r["tail_tracking_error_m"] - b["tail_tracking_error_m"]), 0.) / r["rho"])
D0_upper = float(np.quantile(d0_samples, .95)); Db_upper = float(np.quantile(db_samples, .95))

# Information growth is identified from the observed sample-reset innovation:
# peak_q <= Vh/rho implies Vh >= rho*peak_q.  Use a 95% development bound.
vh_samples = [r["rho"] * r["peak_observer_residual"] for r in sharing]
Vh_upper = float(np.quantile(vh_samples, .95))

# Authority is deliberately conservative: only reserve left after the maximum
# declared trim and a 0.35 m/s^2 safety reserve is credited.
trim = np.asarray(P["declared_trim_acceleration_mps2"], float)
U_lower = max(0., 2.0 - float(np.max(np.linalg.norm(trim, axis=1))) * P["trim_sharing_gain"] - .35)

# Four disagreement modes use the same conservative resources and increasing
# graph contraction factors.  This is an inner contract, not fitted precision.
scales = np.array([.70, .90, 1.10, 1.30])
modes = []
for s in scales:
    modes.append({"a_bar_lower": a_lower * float(s), "kappa_lower": 0.0,
                  "D0_upper": D0_upper, "Db_upper": Db_upper,
                  "U_lower": U_lower, "Vh_upper": Vh_upper})
cal = {
    "status": "DEVELOPMENT_ONLY_CALIBRATION_COMPLETE",
    "development_seeds": P["development_seeds"], "heldout_seed_count_read": 0,
    "runs": len(rows), "confidence_rule": "5th percentile decay and authority; 95th percentile load and information bounds",
    "dominant_decay_samples": {"n": len(a_samples), "lower_95": a_lower},
    "D0": {"n": len(d0_samples), "upper_95": D0_upper},
    "Db": {"n": len(db_samples), "upper_95": Db_upper,
           "positive_fraction": float(np.mean(np.asarray(db_samples) > 0))},
    "U": {"lower_bound": U_lower}, "Vh": {"n": len(vh_samples), "upper_95": Vh_upper},
    "modes": modes,
    "epsilon_peak": P["task_contract"]["analytic_tracking_tolerance_m"],
    "epsilon_ultimate": P["task_contract"]["analytic_tracking_tolerance_m"]
}
(OUT / "uav_v3_development_calibration.json").write_text(json.dumps(cal, indent=2) + "\n")
print(json.dumps(cal, indent=2))
