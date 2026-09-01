#!/usr/bin/env python3
"""Fail closed on physical evidence status.

Manual eligible states are intentionally unsupported. A future transition to
eligibility requires an independently reviewed promotion output backed by an
out-of-band hardware trust root; that verifier does not exist in this tree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DESIGN_STATUSES = {"design_only_not_executed", "hardware_preparation"}
EXECUTED_STATUS = "executed_pending_independent_confirmation"
ELIGIBLE_STATUS = "eligible_independently_confirmed"
ARMS = (
    "primary_observer_loop_runs",
    "primary_oracle_target_runs",
    "primary_no_target_runs",
)


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate(payload: dict) -> list[str]:
    errors: list[str] = []
    status = payload.get("status")
    eligible = payload.get("physical_evidence_eligible")
    claim = payload.get("manuscript_physical_e2e_claim_allowed")
    credit = payload.get("score_credit_for_physical_e2e")
    apparatus = payload.get("apparatus", {})
    runs = payload.get("executions", {})
    independent = payload.get("independent_confirmation", {})

    require(payload.get("protocol_id") == "PHYS-E2E-OCT-R34", "wrong protocol id", errors)
    require(isinstance(payload.get("blockers"), list), "blockers must be a list", errors)
    require(isinstance(payload.get("prohibited_interpretations"), list),
            "prohibited_interpretations must be a list", errors)
    require(eligible is False,
            "manual physical eligibility is prohibited; use a future independently "
            "reviewed hardware-rooted promotion output", errors)
    require(claim is False,
            "manual manuscript physical-E2E claim permission is prohibited", errors)
    require(credit == 0,
            "manual physical-evidence score credit is prohibited", errors)

    if status in DESIGN_STATUSES:
        require(sum(int(runs.get(key, 0)) for key in ARMS) == 0,
                "design status cannot contain primary physical runs", errors)
        require(not independent.get("completed", False),
                "design status cannot contain independent confirmation", errors)
    elif status == EXECUTED_STATUS:
        minimum = int(apparatus.get("required_minimum_motor_nodes", 3))
        require(int(apparatus.get("registered_motor_nodes", 0)) >= minimum,
                "executed status has too few registered motor nodes", errors)
        require(int(apparatus.get("physical_leaders", 0)) >= 1,
                "executed status needs a physical leader", errors)
        require(int(apparatus.get("physical_followers", 0)) >= 2,
                "executed status needs at least two physical followers", errors)
        for key in ARMS:
            require(int(runs.get(key, 0)) > 0, f"executed status has no {key}", errors)
        for key in ("frozen_parameter_file", "raw_manifest", "firmware_binary_manifest",
                    "certificate_verification"):
            require(bool(payload.get(key)), f"executed status missing {key}", errors)
        require(not independent.get("completed", False),
                "manual independent confirmation cannot promote status; use a future "
                "independently reviewed hardware-rooted promotion output", errors)
    elif status == ELIGIBLE_STATUS:
        errors.append(
            "manual eligible_independently_confirmed status is disabled; this repository "
            "has no independently reviewed hardware-rooted promotion verifier"
        )
    else:
        errors.append(f"unsupported status: {status!r}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("status", nargs="?", type=Path,
                        default=Path(__file__).with_name("EVIDENCE_STATUS.json"))
    args = parser.parse_args()
    payload = json.loads(args.status.read_text(encoding="utf-8"))
    errors = validate(payload)
    if errors:
        raise SystemExit("FAIL physical evidence status\n" + "\n".join(errors))
    print(f"PASS: {payload['status']}; physical_evidence_eligible="
          f"{payload['physical_evidence_eligible']}; score_credit="
          f"{payload['score_credit_for_physical_e2e']}")


if __name__ == "__main__":
    main()
