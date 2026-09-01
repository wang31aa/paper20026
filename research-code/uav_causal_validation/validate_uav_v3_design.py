#!/usr/bin/env python3
"""Static, outcome-blind validation of the V3 preregistration."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "UAV_V3_PREREGISTRATION.json").read_text())
dev, test = set(P["development_seeds"]), set(P["heldout_seeds"])
t = P["task_contract"]
derived = min((t["nominal_adjacent_spacing_m"] - t["minimum_pair_clearance_m"]) / 2,
              t["corridor_half_width_m"] - t["nominal_max_lateral_offset_m"],
              t["terminal_reference_overshoot_m"])
checks = {
    "design_not_mislabelled_as_executed": P["status"] == "DESIGN_COMPLETE_NOT_CALIBRATED_NOT_RUN",
    "seed_sets_disjoint": not (dev & test),
    "heldout_larger_than_development": len(test) > len(dev),
    "rho_grid_spans_weak_and_strong": min(P["rho_grid"]) < 0.25 and max(P["rho_grid"]) > 1.25,
    "observer_contains_trim_state": "constant_trim_acceleration" in P["observer"]["augmented_state"],
    "mechanism_ablation_present": "all_heterogeneous_without_feedforward_sharing" in P["heterogeneity_sources"]["required_ablation_arms"],
    "task_tolerance_derived_correctly": abs(derived - t["analytic_tracking_tolerance_m"]) < 1e-12,
    "frozen_hash_required": P["parameter_identification"]["freeze_hash_required"],
    "adverse_branches_retained": all(x in P["pre_registered_branches"] for x in
                                     ("monotone_information", "empty_window", "disconnected_window")),
    "positive_result_not_required": P["promotion_gates"]["positive_result_not_required_for_reporting"]
}
report = {"checks": checks, "design_valid": all(checks.values()),
          "execution_status": "NOT_RUN", "derived_epsilon_task_m": derived}
(HERE / "results" / "uav_v3_design_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["design_valid"] else 2)
