#!/usr/bin/env python3
"""Fail-closed validation of BIOSOCIAL_V1 artifacts."""
import csv, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
contract = ROOT / "BIOSOCIAL_V1_FROZEN_CONTRACT.json"
summary = json.loads((ROOT / "results/biosocial_v1_summary.json").read_text())
rows = list(csv.DictReader((ROOT / "results/biosocial_v1_runs.csv").open()))
digest = hashlib.sha256(contract.read_bytes()).hexdigest()

checks = {
    "contract_hash_matches": digest == summary["contract_sha256"],
    "row_count_complete": len(rows) == 1560,
    "heldout_count_complete": sum(r["split"] == "heldout" for r in rows) == 780,
    "no_duplicate_keys": len({(r["domain"], r["rho"], r["split"], r["seed"]) for r in rows}) == len(rows),
    "no_false_safe": all(r["success"] == "1" for r in rows
                         if r["split"] == "heldout" and r["certificate_safe"] == "1"),
    "counterexample_is_monotone": True,
    "status_is_synthetic": summary["status"] == "SYNTHETIC_MECHANISM_TEST_ONLY"
}
counter = {}
for r in rows:
    if r["domain"] == "monotone_counterexample" and r["split"] == "heldout":
        counter.setdefault(float(r["rho"]), []).append(int(r["success"]))
rates = [sum(counter[x])/len(counter[x]) for x in sorted(counter)]
checks["counterexample_is_monotone"] = all(b >= a for a, b in zip(rates, rates[1:]))

report = {"qualified": all(checks.values()), "checks": checks,
          "evidence_class": "SYNTHETIC_MECHANISM_TEST_ONLY"}
(ROOT / "results/biosocial_v1_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["qualified"] else 1)
