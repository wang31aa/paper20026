#!/usr/bin/env python3
"""Fail-closed validation for the theory-first UAV v2 experiment."""
import csv, json, math
from pathlib import Path

HERE = Path(__file__).resolve().parent; OUT = HERE / "results"
P = json.loads((HERE / "UAV_THEORY_FIRST_PROTOCOL_V2.json").read_text())
S = json.loads((OUT / "uav_v2_summary.json").read_text())
rows = list(csv.DictReader((OUT / "uav_v2_runs.csv").open()))
expected = (len(P["development_seeds"]) * len(P["environments"]["development"]) +
            len(P["heldout_seeds"]) * len(P["environments"]["heldout"])) * len(P["participation_grid"]) * 2 * len(P["policies"])
checks = {
    "row_count": len(rows) == expected,
    "all_finite": all(math.isfinite(float(r[k])) for r in rows for k in
                      ("completion_fraction", "minimum_pair_clearance_m", "peak_observer_residual",
                       "tail_observer_residual", "control_energy", "messages_delivered")),
    "paired_keys": len({(r["split"], r["environment"], r["seed"], r["rho"], r["plant"], r["policy"]) for r in rows}) == len(rows),
    "observer_logged": all(float(r["update_interval_s"]) > 0 and float(r["peak_observer_residual"]) >= 0 for r in rows),
    "rho_changes_update_interval": len({r["update_interval_s"] for r in rows}) > 1,
    "exact_modal_agreement": bool(S["exact_modal_prediction_agreement"]),
    "heldout_has_both_outcomes": len({int(r["task_success"]) for r in rows if r["split"] == "heldout" and r["plant"] == "heterogeneous"}) == 2,
    "heldout_sensitivity": S["heldout_sensitivity"] is not None and S["heldout_sensitivity"] >= P["fail_closed_promotion"]["heldout_window_sensitivity_min"],
    "heldout_specificity": S["heldout_specificity"] is not None and S["heldout_specificity"] >= P["fail_closed_promotion"]["heldout_window_specificity_min"]
}
qualified = all(checks.values())
report = {"checks": checks, "all_pass": qualified,
          "uav_causal_validation_qualified": qualified,
          "evidence_label": P["evidence_label"] if qualified else "NOT_QUALIFIED theory-first computational study"}
(OUT / "uav_v2_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if qualified else 2)
