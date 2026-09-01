#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    contract_path = HERE / "V49_STAGE1_CONTRACT.json"
    frozen_path = RESULTS / "V49_FROZEN_WARNING.json"
    heldout_path = RESULTS / "V49_HELDOUT_WARNING.json"
    contract = json.loads(contract_path.read_text())
    frozen = json.loads(frozen_path.read_text())
    heldout = json.loads(heldout_path.read_text())

    eligible = [x for x in frozen["candidate_metrics"] if x["sensitivity"] >= 0.8]
    expected = sorted(
        eligible or frozen["candidate_metrics"],
        key=lambda x: (-x["youden"], x["threshold_m"]),
    )[0]["threshold_m"]
    checks = {
        "contract_declared_pre_execution": contract["created_before_execution"] is True,
        "contract_hash_matches": frozen["contract_sha256"] == sha256(contract_path),
        "prediction_hash_matches": heldout["prediction_sha256"] == sha256(frozen_path),
        "seed_sets_disjoint": not set(contract["development_seeds"]) & set(contract["heldout_seeds"]),
        "selection_rule_reproduced": frozen["chosen_threshold_m"] == expected,
        "threshold_carried_to_holdout": heldout["chosen_threshold_m"] == expected,
        "development_run_count": heldout["development_runs"] == len(contract["development_seeds"]) * 27,
        "heldout_run_count": heldout["heldout_runs"] == len(contract["heldout_seeds"]) * 27,
        "heldout_sensitivity_target": heldout["heldout"]["sensitivity"] >= 0.8,
        "heldout_specificity_nonzero": heldout["heldout"]["specificity"] > 0.0,
    }
    report = {"all_pass": all(checks.values()), "checks": checks}
    (RESULTS / "V49_VALIDATION.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    main()
