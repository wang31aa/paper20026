#!/usr/bin/env python3
"""Fail-closed registry for mutually non-substitutable UAV evidence tiers."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


archive = load("archived_nmpc_causal_rollout_qualification.json")
native = load("native_matlab_source_replay_metrics_cloud_31726983422.json")
v8 = load("UAV_V8_QUALIFICATION_REGISTRY.json")

tiers = {
    "archived_state_input_causal_reconstruction": bool(archive["qualified"]),
    "official_source_controller_replay": bool(native["qualified"]),
    "same_simulator_heterogeneous_counterfactual": v8["status"]
    == "V8_SAME_SIMULATOR_CONFIRMATORY_EXTENSION_PASSED",
    "public_flight_counterfactual": False,
    "hil_execution_of_proposed_gate": False,
    "hardware_execution_of_proposed_gate": False,
}
report = {
    "schema_version": "1.0",
    "tiers": tiers,
    "non_substitution_rules": [
        "Archived applied-control reconstruction does not imply optimiser reproduction.",
        "Same-simulator intervention does not imply a public-flight counterfactual.",
        "A public trajectory or HIL record without the proposed gate does not imply HIL execution.",
        "No computational result can be relabelled as hardware execution.",
    ],
    "official_nmpc_gate": {
        "qualified": False,
        "control_integral_recorded": native["metrics"]["control_integral_recorded"],
        "control_integral_replay": native["metrics"]["control_integral_native"],
        "relative_error": abs(
            native["metrics"]["control_integral_native"]
            - native["metrics"]["control_integral_recorded"]
        )
        / native["metrics"]["control_integral_recorded"],
        "nonzero_solver_statuses": native["metrics"]["solver_nonzero_statuses"],
        "identifiability_conclusion": (
            "The archived solver-status trace and complete original numerical build fingerprint "
            "are absent. The historical optimiser path is therefore not identifiable from the "
            "public archive and cannot be made qualified by parameter fitting or tolerance changes."
        ),
    },
    "overall_claim": (
        "The public archive is dynamically self-consistent and the V8 heterogeneous intervention "
        "is causally valid in its declared simulator. Official NMPC source replay, public-flight "
        "counterfactual, HIL and hardware execution remain separate unpassed tiers."
    ),
}
(RESULTS / "UAV_EVIDENCE_TIER_REGISTRY.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))

if not tiers["archived_state_input_causal_reconstruction"]:
    raise SystemExit("archived causal reconstruction failed")
if tiers["official_source_controller_replay"]:
    raise SystemExit("unexpected source replay promotion: inspect frozen evidence")
