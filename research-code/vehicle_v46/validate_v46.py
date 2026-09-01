#!/usr/bin/env python3
from __future__ import annotations
import csv, json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader((HERE / "results/V46_PAIRED_RESULTS.csv").open()))
spec = json.loads((HERE / "V46_FROZEN_CONTRACT.json").read_text())
expected = len(spec["grid"]["seeds"]) * len(spec["grid"]["stress"]) * len(spec["grid"]["delay_s"]) * len(spec["grid"]["participation"]) * len(spec["policies"])
groups = defaultdict(list)
for r in rows:
    groups[(r["seed"], r["stress"], r["delay_s"], r["rho"])].append(r)
checks = {
    "row_count": len(rows) == expected,
    "complete_paired_blocks": all(len(v) == len(spec["policies"]) for v in groups.values()),
    "all_policies_present": {r["policy"] for r in rows} == set(spec["policies"]),
    "heterogeneity_entered": all(float(r["lag_spread"]) > 1.15 and float(r["drag_spread"]) > 1.15 and float(r["authority_spread"]) > 1.08 for r in rows),
    "finite_metrics": all(float(r[k]) == float(r[k]) for r in rows for k in ("minimum_margin_m", "control_energy", "observer_rmse_mps")),
    "policy_changes_outcome": any(len({round(float(x["minimum_margin_m"]), 6) for x in v}) > 1 for v in groups.values()),
}
status = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "rows": len(rows), "paired_conditions": len(groups)}
(HERE / "results/V46_VALIDATION.json").write_text(json.dumps(status, indent=2) + "\n")
print(json.dumps(status, indent=2))
raise SystemExit(0 if status["status"] == "PASS" else 1)
