#!/usr/bin/env python3
"""Fail-closed static qualification of the MuSCAT reaction-wheel causal chain.

This validator does not qualify an intervention result.  It only establishes
that a proposed source-model experiment can use a real state -> command ->
actuator -> applied torque -> next-state path and rejects the broken KKT path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SOURCE_COMMIT = "d8a0739426a23b3e25dc66e0fa4b7152972c11df"


def require(text: str, needle: str, label: str, checks: dict[str, bool]) -> None:
    checks[label] = needle in text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.source_root
    main_text = (root / "Main/main_v3.m").read_text()
    adc_text = (root / "True_SC/True_SC_ADC.m").read_text()
    wheel_text = (root / "True_Sensors_Actuators/True_SC_Reaction_Wheel.m").read_text()
    pinv_text = (root / "Supporting_Functions/mission_specific/DART/func_apply_reaction_wheel_commands.m").read_text()
    kkt_text = (root / "Supporting_Functions/mission_specific/DART/func_compute_rw_command_kkt.m").read_text()

    checks: dict[str, bool] = {}
    require(main_text, "func_main_true_SC_attitude", "state_integrator_called", checks)
    require(main_text, "func_main_software_SC_control_attitude", "controller_called", checks)
    require(main_text, "func_main_true_reaction_wheel", "wheel_actuator_called", checks)
    checks["integration_precedes_control"] = (
        main_text.index("func_main_true_SC_attitude")
        < main_text.index("func_main_software_SC_control_attitude")
        < main_text.index("func_main_true_reaction_wheel")
    )
    require(adc_text, "obj.total_torque = obj.control_torque + obj.disturbance_torque", "applied_torque_enters_dynamics", checks)
    require(adc_text, "obj.control_torque =  [0 0 0]", "applied_torque_is_consumed_once", checks)
    require(wheel_text, "obj.maximum_acceleration", "wheel_acceleration_limit_present", checks)
    require(wheel_text, "obj.max_angular_velocity", "wheel_speed_limit_present", checks)
    require(wheel_text, "obj.saturated", "wheel_saturation_telemetry_present", checks)
    require(wheel_text, "mission.true_SC{i_SC}.true_SC_adc.control_torque", "actual_wheel_torque_written_to_adc", checks)
    require(pinv_text, "commanded_angular_acceleration", "valid_wheel_command_field_used", checks)
    require(pinv_text, "flag_executive", "wheel_execution_flag_used", checks)
    checks["kkt_path_rejected"] = "commanded_torque" in kkt_text and "commanded_torque" not in wheel_text

    qualified = all(checks.values())
    result = {
        "schema": "MUSCAT_ACTUATOR_CHAIN_STATIC_V1",
        "source_commit": SOURCE_COMMIT,
        "qualified_for": "source-model causal actuator path construction",
        "not_qualified_for": [
            "intervention outcome",
            "source-controller replay",
            "hardware-in-the-loop",
            "flight or physical experiment",
        ],
        "approved_allocation_path": "DART pseudoinverse -> commanded_angular_acceleration",
        "rejected_allocation_path": "DART KKT -> undeclared commanded_torque field",
        "checks": checks,
        "qualified": qualified,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
