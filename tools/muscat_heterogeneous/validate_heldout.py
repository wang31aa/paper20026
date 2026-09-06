#!/usr/bin/env python3
"""Evaluate only the acceptance rules frozen in the signed contract."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


contract = json.loads(Path(sys.argv[1]).read_text())
rows = list(csv.DictReader(Path(sys.argv[2]).open()))
grouped: dict[tuple[str, float], list[dict[str, str]]] = defaultdict(list)
for row in rows:
    grouped[(row["policy"], float(row["rho"]))].append(row)

summary = {}
for (policy, rho), values in grouped.items():
    summary.setdefault(policy, {})[rho] = {
        "success_fraction": sum(int(v["success"]) for v in values) / len(values),
        "mean_tail_error_rad": sum(float(v["tail_error_rad"]) for v in values) / len(values),
    }

rule = contract["frozen_prediction"]["acceptance"]
checks = {}
for policy, curve in summary.items():
    checks[f"{policy}_low_middle"] = min(v["success_fraction"] for r, v in curve.items() if r <= 1.35) >= rule["rho_at_or_below_1_35_success_fraction_min"]
    checks[f"{policy}_high_boundary"] = max(v["success_fraction"] for r, v in curve.items() if r >= 2.75) <= rule["rho_at_or_above_2_75_success_fraction_max"]
    ordered = [curve[r]["mean_tail_error_rad"] for r in sorted(curve)]
    nonincrease = sum(b <= a for a, b in zip(ordered, ordered[1:])) / (len(ordered) - 1)
    checks[f"{policy}_tail_nonincrease"] = nonincrease >= rule["tail_error_nonincrease_fraction"]
    checks[f"{policy}_no_aggregate_false_safe_at_3_40"] = curve[3.4]["success_fraction"] <= rule["rho_at_or_above_2_75_success_fraction_max"]

result = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
          "contract_sha256": contract["contract_sha256"], "summary": summary}
print(json.dumps(result, indent=2, sort_keys=True))
if result["status"] != "PASS":
    raise SystemExit(1)
