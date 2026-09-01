#!/usr/bin/env python3
"""Fail-closed eligibility audit for the frozen three-domain protocol."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def headers(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        return set(next(csv.reader(handle)))


def main() -> None:
    robot_path = ROOT / "physical_task_twins/results/robot_closed_loop_tasks.csv"
    vehicle_path = ROOT / "physical_task_twins/results/vehicle_physical_tasks.csv"
    carma_path = ROOT / "external_validation/carma2/results/vehicle_run_metrics.csv"
    epfl_path = ROOT / "external_physical_validation/results/epfl_swarm_archive_audit_fresh.json"

    robot_h = headers(robot_path)
    vehicle_h = headers(vehicle_path)
    carma_h = headers(carma_path)
    epfl = json.loads(epfl_path.read_text())

    required_dynamic = {
        "timestamp", "node_id", "reference", "estimate", "information_error",
        "message_age", "adjacency", "participation_weight", "switch_reason",
        "requested_control", "executed_control", "physical_margin", "failure",
        "recovery", "control_cost", "communication_cost",
    }

    # Existing summary tables are deliberately not mistaken for the required
    # per-step prospective logs.
    checks = {
        "robot_summary_present": robot_path.exists(),
        "vehicle_summary_present": vehicle_path.exists(),
        "carma_response_summary_present": carma_path.exists(),
        "epfl_archive_checksum_qualified": epfl.get("archive_md5") == "d0dbf0bfb4891a3f34fcb2e381971087",
        "epfl_five_agent_state_present": epfl["field_decision"].get("five_agent_position", "").startswith("pos_history"),
        "epfl_hil_low_level_control_present": bool(epfl["field_decision"].get("hitl_low_level_u_t_present")),
        "robot_full_prospective_log": required_dynamic <= robot_h,
        "vehicle_full_prospective_log": required_dynamic <= vehicle_h,
        "carma_gate_executed": {"gate_weight", "gate_action", "post_gate_state"} <= carma_h,
        "three_domain_leave_out_results_present": (ROOT / "cross_domain_prospective/results/leave_domain_out.csv").exists(),
        "two_physical_platform_recovery_present": (ROOT / "cross_domain_prospective/results/physical_platform_recovery.csv").exists(),
    }
    blocking = [name for name, passed in checks.items() if not passed]
    report = {
        "status": "PASS" if not blocking else "NOT_ELIGIBLE",
        "checks": checks,
        "blocking": blocking,
        "interpretation": (
            "Qualification files and computational summaries are present, but the prospective "
            "three-domain and two-platform evidence gates remain closed." if blocking else
            "All declared evidence files are present; substantive statistical and theorem audit is still required."
        ),
    }
    out = ROOT / "cross_domain_prospective/eligibility.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if not blocking else 2)


if __name__ == "__main__":
    main()
