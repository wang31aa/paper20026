#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
C = json.loads((HERE / "V56_FROZEN_CONTRACT.json").read_text())
R = json.loads((HERE / "results/V56_REPORT.json").read_text())
rows = list(csv.DictReader((HERE / "results/V56_ALL_POLICY_RUNS.csv").open()))
selected = list(csv.DictReader((HERE / "results/V56_SELECTED_POLICY_EVALUATION.csv").open()))

source_seeds = set()
for path in [ROOT / "uav_v54_sixdof/UAV_V54_FROZEN_CONTRACT.json",
             ROOT / "vehicle_v52_two_layer_platoon/V52_FROZEN_CONTRACT.json",
             ROOT / "cross_domain_v21/V21_MOTOR_FROZEN_CONTRACT.json"]:
    payload = json.loads(path.read_text())
    source_seeds.update(payload.get("development_seeds", []))
    source_seeds.update(payload.get("heldout_seeds", []))
new_seeds = {seed for platform in C["new_streams"].values() for seed in platform["seeds"]}

by_case = defaultdict(list)
for row in rows:
    by_case[(row["platform"], row["case"])].append(row)
expected_policy_counts = {"uav6dof": 4, "vehicle": 4, "motor": 6}
checks = {
    "contract_hash": R["contract_sha256"] == hashlib.sha256((HERE / "V56_FROZEN_CONTRACT.json").read_bytes()).hexdigest(),
    "three_platforms": {x["platform"] for x in rows} == set(C["platforms"]),
    "new_seeds_disjoint": not (new_seeds & source_seeds),
    "complete_policy_pairing": all(len(q) == expected_policy_counts[p] for (p, _), q in by_case.items()),
    "one_selected_row_per_condition": len(selected) == len(by_case),
    "finite": all(math.isfinite(float(row[key])) for row in rows for key in ["physical_margin", "terminal_error", "control_cost", "communication", "loss"]),
    "regret_nonnegative": all(float(row["regret"]) >= -1e-12 for row in selected),
    "failures_reported": all("false_safe_count" in R["summary"][p] for p in C["platforms"]),
    "scope_computational": "computational" in R["claim_boundary"],
    "scope_not_hil": "not leave-physical-domain" in R["claim_boundary"] and "HIL" in R["claim_boundary"],
}
status = "PASS" if all(checks.values()) else "FAIL"
out = {"status": status, "checks": checks, "rows": len(rows), "conditions": len(by_case)}
(HERE / "results/V56_VALIDATION.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
if status != "PASS":
    raise SystemExit(1)
