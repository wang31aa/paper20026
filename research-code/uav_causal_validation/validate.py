#!/usr/bin/env python3
import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
reg = json.loads((HERE / "qualification_registry.json").read_text())
heterogeneous = json.loads((HERE / "HETEROGENEOUS_EXPERIMENT_REGISTRY.json").read_text())
errors = []
required = ("L0_archive_qualified", "L1_source_replay_qualified",
            "L2_paired_intervention_qualified", "L3_heldout_prediction_qualified")

for key in required:
    if key not in reg["levels"]: errors.append(f"missing level {key}")
expected = all(reg["levels"][key] for key in required)
if reg["causal_validation_qualified"] != expected:
    errors.append("causal flag is not the conjunction of L0-L3")
if reg["status"] == "QUALIFIED" and not expected:
    errors.append("fail-open status")
if not reg["levels"]["L1_source_replay_qualified"]:
    for downstream in required[2:]:
        if reg["levels"][downstream]: errors.append(f"{downstream} true while L1 false")
    if heterogeneous["status"] != "LOCKED_UNTIL_L1":
        errors.append("heterogeneous experiment unlocked while L1 false")
    if heterogeneous["paired_closed_loop_executed"] or heterogeneous["heldout_opened"]:
        errors.append("heterogeneous outcomes opened while L1 false")
if heterogeneous["claim_status"] == "QUALIFIED" and not expected:
    errors.append("heterogeneous causal claim promoted before L0-L3")
for name in ("source_replay_metrics", "native_matlab_replay_metrics", "translation_unit_tests", "frozen_prediction",
             "paired_counterfactual_log", "heldout_metrics"):
    value = reg["artifacts"][name]
    if value is not None and not (HERE / value).is_file(): errors.append(f"missing artifact {value}")
if errors:
    print(json.dumps({"validator": "FAIL", "errors": errors}, indent=2)); sys.exit(1)
print(json.dumps({"validator": "PASS", "status": reg["status"],
                  "causal_validation_qualified": expected,
                  "blocking_reason": reg["blocking_reason"]}, indent=2))
