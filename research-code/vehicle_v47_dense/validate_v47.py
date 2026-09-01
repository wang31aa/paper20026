#!/usr/bin/env python3
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
spec = json.loads((HERE / "V47_STAGE1_CONTRACT.json").read_text())
pred = json.loads((OUT / "V47_FROZEN_PREDICTIONS.json").read_text())
evaluation = json.loads((OUT / "V47_HELDOUT_EVALUATION.json").read_text())

with (OUT / "V47_DEVELOPMENT.csv").open(newline="") as f:
    development = list(csv.DictReader(f))
with (OUT / "V47_HELDOUT.csv").open(newline="") as f:
    heldout = list(csv.DictReader(f))

expected_development = (len(spec["development_seeds"]) * len(spec["stress"]) *
                        len(spec["delay_s"]) * len(spec["participation_grid"]) *
                        len(spec["policies"]))
expected_heldout = (len(spec["heldout_seeds"]) * len(spec["stress"]) *
                    len(spec["delay_s"]) * len(spec["participation_grid"]) *
                    len(spec["policies"]))
pred_sha = hashlib.sha256((OUT / "V47_FROZEN_PREDICTIONS.json").read_bytes()).hexdigest()
checks = {
    "development_and_heldout_seeds_disjoint": not set(spec["development_seeds"]) & set(spec["heldout_seeds"]),
    "development_row_count": len(development) == expected_development,
    "heldout_row_count": len(heldout) == expected_heldout,
    "prediction_hash_matches_evaluation": pred_sha == evaluation["prediction_sha256"],
    "contract_hash_matches_prediction": hashlib.sha256((HERE / "V47_STAGE1_CONTRACT.json").read_bytes()).hexdigest() == pred["contract_sha256"],
    "all_policies_present": set(spec["policies"]) == set(evaluation["policies"]),
    "all_participation_levels_present": all(len(evaluation["policies"][p]["heldout_curve"]) == len(spec["participation_grid"]) for p in spec["policies"])
}
report = {"passed": all(checks.values()), "checks": checks,
          "development_rows": len(development), "heldout_rows": len(heldout),
          "claim_boundary": spec["claim_boundary"]}
(OUT / "V47_VALIDATION.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["passed"] else 1)
