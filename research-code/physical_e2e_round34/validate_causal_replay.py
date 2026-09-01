#!/usr/bin/env python3
"""Independent fail-closed causal replay for observer-loop raw records.

PASS means only that supplied records are algebraically and causally consistent
with a frozen, pre-outcome contract. It does not authenticate hardware origin,
establish plant performance, or create scientific results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from contract_common import PROTOCOL_ID, ContractError, load_json, sha256_file
from validate_parameters import (
    DEFAULT_SCHEMA,
    canonical_parameter_payload_sha256,
    validate_completed,
    validate_schema_document,
)

CLASSIFICATION = "CAUSAL_REPLAY_CONSISTENCY_ONLY_NOT_HARDWARE_EVIDENCE"
REQUIRED_RAW = {
    "protocol_id", "run_id", "arm", "node_id", "local_tick",
    "scheduled_time_us", "local_time_us", "computation_time_us",
    "actuation_time_us", "deadline_missed", "speed_normalised",
    "state_quantised_local", "observer_state_pre", "observer_state_post",
    "observer_innovation", "controller_network_innovation",
    "leader_u_nominal", "follower_u_nominal", "u_command",
    "u_pre_saturation", "follower_u_clipped_float", "u_post_saturation",
    "follower_u_quantised_signed", "direction_bit", "pwm_count", "pwm_register",
    "pwm_magnitude", "saturation_flag", "pin_state",
    "received_leader_state", "received_neighbor_state",
    "received_neighbor_observer", "frozen_parameter_sha256",
}
REQUIRED_PACKET = {
    "protocol_id", "run_id", "arm", "receiver_node", "sender_node",
    "local_tick", "sender_tick", "packet_sequence", "packet_kind",
    "receive_time_us", "sender_time_us", "packet_age_samples",
    "held_packet_flag", "state_value", "observer_value", "forcing_value",
    "frozen_parameter_sha256",
}


def _finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ContractError(f"{label} must be finite")
    return result


def _integer(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} must be an integer") from exc


def _boolean(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ContractError(f"{label} must be boolean")


def _missing(value: Any) -> bool:
    return value is None or str(value).strip() in {"", "NA"}


def _close(actual: Any, expected: float, label: str, tolerance: float) -> None:
    value = _finite(actual, label)
    if abs(value - expected) > tolerance * max(1.0, abs(value), abs(expected)):
        raise ContractError(f"{label} mismatch: logged={value!r}, replay={expected!r}")


def _quantize(value: float, step: float) -> float:
    scaled = value / step
    units = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
    return units * step


def _canonical_file_hash(parameters: dict[str, Any]) -> str:
    encoded = json.dumps(
        parameters, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _models(parameters: dict[str, Any]) -> dict[int, dict[str, float]]:
    try:
        rows = parameters["observer_controller"]["nominal_parameters"]
    except (KeyError, TypeError) as exc:
        raise ContractError("missing frozen nominal parameters") from exc
    result: dict[int, dict[str, float]] = {}
    for row in rows:
        node = _integer(row.get("node_id"), "nominal.node_id")
        if node in result:
            raise ContractError(f"duplicate nominal model for node {node}")
        result[node] = {
            "a": _finite(row.get("a_nominal"), f"nominal[{node}].a"),
            "b": _finite(row.get("b_nominal"), f"nominal[{node}].b"),
            "c": _finite(row.get("c_nominal"), f"nominal[{node}].c"),
        }
    if set(result) != {0, 1, 2} or any(result[i]["b"] <= 0 for i in result):
        raise ContractError("nominal models must cover nodes 0,1,2 with positive b")
    return result


def _verify_certificate_binding(
    parameters: dict[str, Any], certificate_report: dict[str, Any],
    certificate_report_sha256: str,
) -> str:
    if parameters.get("status") != "preregistration_parameters_no_results":
        raise ContractError("parameters are not a completed pre-outcome freeze")
    payload_hash = canonical_parameter_payload_sha256(parameters)
    if parameters.get("freeze", {}).get("canonical_parameter_payload_sha256") != payload_hash:
        raise ContractError("canonical frozen-parameter hash mismatch")
    expected_report_hash = parameters.get("certificate", {}).get(
        "verification_report_sha256"
    )
    if certificate_report_sha256 != expected_report_hash:
        raise ContractError("certificate report bytes do not match frozen report hash")
    required = {
        "status": "PASS",
        "classification": "PRE_OUTCOME_CERTIFICATE_ONLY_NOT_PHYSICAL_RESULT",
        "contains_outcomes": False,
        "physical_result": False,
        "protocol_id": PROTOCOL_ID,
    }
    for key, expected in required.items():
        if certificate_report.get(key) != expected:
            raise ContractError(f"certificate report binding failed at {key}")
    if parameters["certificate"].get("all_vertices_verified") is not True:
        raise ContractError("frozen certificate does not verify all vertices")
    return payload_hash


def validate_replay(
    parameters: dict[str, Any], certificate_report: dict[str, Any],
    certificate_report_sha256: str, raw_rows: Iterable[dict[str, Any]],
    packet_rows: Iterable[dict[str, Any]], *, parameter_file_sha256: str,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Replay supplied records. Returned counts are audit counts, not outcomes."""
    payload_hash = _verify_certificate_binding(
        parameters, certificate_report, certificate_report_sha256
    )
    models = _models(parameters)
    oc = parameters["observer_controller"]
    qa = parameters["quantization_actuation"]
    timing = parameters["timing_network"]
    gamma = _finite(oc["observer_gain_gamma"], "observer_gain_gamma")
    kappa = _finite(oc["controller_gain_kappa"], "controller_gain_kappa")
    dx = _finite(qa["state_quantisation_step"], "state_quantisation_step")
    ds = _finite(qa["observer_quantisation_step"], "observer_quantisation_step")
    du = _finite(qa["pwm_quantisation_step"], "pwm_quantisation_step")
    umin = _finite(qa["signed_pwm_min"], "signed_pwm_min")
    umax = _finite(qa["signed_pwm_max"], "signed_pwm_max")
    ages = {_integer(x, "admissible age") for x in timing["admissible_packet_ages_samples"]}
    deadline = _integer(timing["computation_deadline_us"], "computation_deadline_us")
    max_holds = _integer(timing["max_consecutive_packet_holds"], "max holds")

    packets: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    last_sequence: dict[tuple[str, int, int, str], int] = {}
    hold_count: dict[tuple[str, int, int, str], int] = {}
    packet_count = 0
    for line, packet in enumerate(packet_rows, 2):
        missing = REQUIRED_PACKET - set(packet)
        if missing:
            raise ContractError(f"packet row {line} missing columns: {sorted(missing)}")
        if packet["protocol_id"] != PROTOCOL_ID or packet["arm"] != "observer_loop":
            raise ContractError(f"packet row {line} identity/arm mismatch")
        if packet["frozen_parameter_sha256"] != parameter_file_sha256:
            raise ContractError(f"packet row {line} parameter hash mismatch")
        receiver = _integer(packet["receiver_node"], f"packet row {line}.receiver")
        sender = _integer(packet["sender_node"], f"packet row {line}.sender")
        tick = _integer(packet["local_tick"], f"packet row {line}.tick")
        sender_tick = _integer(packet["sender_tick"], f"packet row {line}.sender_tick")
        age = _integer(packet["packet_age_samples"], f"packet row {line}.age")
        kind = str(packet["packet_kind"])
        allowed = (
            (receiver == 1 and sender == 0 and kind in {"leader_state", "leader_forcing"})
            or (receiver == 2 and sender == 1 and kind == "neighbor_state_observer")
            or (receiver == 2 and sender == 0 and kind == "leader_forcing")
        )
        if not allowed:
            raise ContractError(f"packet row {line} violates observer-loop ACL")
        if receiver == 2 and kind == "leader_state":
            raise ContractError(f"packet row {line} leaks target state to node 2")
        if age not in ages or sender_tick != tick - age:
            raise ContractError(f"packet row {line} age/tick identity fails")
        sent = _integer(packet["sender_time_us"], f"packet row {line}.sender_time")
        received = _integer(packet["receive_time_us"], f"packet row {line}.receive_time")
        if sent > received:
            raise ContractError(f"packet row {line} arrives before it is sent")
        stream = (str(packet["run_id"]), receiver, sender, kind)
        sequence = _integer(packet["packet_sequence"], f"packet row {line}.sequence")
        held = _boolean(packet["held_packet_flag"], f"packet row {line}.held")
        if stream in last_sequence:
            if sequence < last_sequence[stream] or (sequence == last_sequence[stream] and not held):
                raise ContractError(f"packet row {line} reorders/reuses an unmarked packet")
        last_sequence[stream] = sequence
        hold_count[stream] = hold_count.get(stream, 0) + 1 if held else 0
        if hold_count[stream] > max_holds:
            raise ContractError(f"packet row {line} exceeds frozen hold limit")
        key = (str(packet["run_id"]), receiver, tick, kind)
        if key in packets:
            raise ContractError(f"duplicate packet event for {key}")
        packets[key] = packet
        packet_count += 1

    checked = 0
    seen: set[tuple[str, int, int]] = set()
    last_tick: dict[tuple[str, int], int] = {}
    for line, row in enumerate(raw_rows, 2):
        missing = REQUIRED_RAW - set(row)
        if missing:
            raise ContractError(f"raw row {line} missing columns: {sorted(missing)}")
        if row["protocol_id"] != PROTOCOL_ID or row["arm"] != "observer_loop":
            raise ContractError(f"raw row {line} identity/arm mismatch")
        if row["frozen_parameter_sha256"] != parameter_file_sha256:
            raise ContractError(f"raw row {line} parameter hash mismatch")
        run = str(row["run_id"])
        node = _integer(row["node_id"], f"raw row {line}.node")
        tick = _integer(row["local_tick"], f"raw row {line}.tick")
        if node not in {1, 2}:
            raise ContractError(f"raw row {line} is not follower 1 or 2")
        identity = (run, node, tick)
        if identity in seen:
            raise ContractError(f"duplicate raw identity {identity}")
        seen.add(identity)
        stream = (run, node)
        if stream in last_tick and tick != last_tick[stream] + 1:
            raise ContractError(f"raw row {line} has non-consecutive tick")
        last_tick[stream] = tick
        scheduled = _integer(row["scheduled_time_us"], f"raw row {line}.scheduled")
        local = _integer(row["local_time_us"], f"raw row {line}.local")
        computation = _integer(row["computation_time_us"], f"raw row {line}.computation")
        actuation = _integer(row["actuation_time_us"], f"raw row {line}.actuation")
        if local < scheduled or computation < 0 or actuation < local:
            raise ContractError(f"raw row {line} has noncausal local timestamps")
        if computation > deadline or actuation > scheduled + deadline:
            raise ContractError(f"raw row {line} misses frozen computation deadline")
        if _boolean(row["deadline_missed"], f"raw row {line}.deadline_missed"):
            raise ContractError(f"raw row {line} declares a deadline miss")

        forcing = packets.get((run, node, tick, "leader_forcing"))
        source_kind = "leader_state" if node == 1 else "neighbor_state_observer"
        source = packets.get((run, node, tick, source_kind))
        if forcing is None or source is None:
            raise ContractError(f"raw row {line} lacks required capture-derived events")
        for event in (forcing, source):
            if _integer(event["receive_time_us"], "packet.receive_time") > actuation:
                raise ContractError(f"raw row {line} uses a packet received after actuation")
        if node == 2 and not _missing(row["received_leader_state"]):
            raise ContractError(f"raw row {line} exposes target state at node 2")
        if _boolean(row["pin_state"], f"raw row {line}.pin_state") != (node == 1):
            raise ContractError(f"raw row {line} pin state violates frozen topology")

        x = _finite(row["speed_normalised"], f"raw row {line}.x")
        shat = _finite(row["observer_state_pre"], f"raw row {line}.shat")
        qx, qs = _quantize(x, dx), _quantize(shat, ds)
        _close(row["state_quantised_local"], qx, f"raw row {line}.Qx", tolerance)
        u0 = _finite(forcing["forcing_value"], f"raw row {line}.u0")
        leader = models[0]
        prediction = leader["a"] * shat + leader["b"] * u0 + leader["c"]
        if node == 1:
            target = _finite(source["state_value"], f"raw row {line}.leader_state")
            _close(row["received_leader_state"], target, f"raw row {line}.leader log", tolerance)
            observer_z = qs - target
            controller_z = qx - qs
        else:
            nx = _finite(source["state_value"], f"raw row {line}.neighbor_state")
            ns = _finite(source["observer_value"], f"raw row {line}.neighbor_observer")
            _close(row["received_neighbor_state"], nx, f"raw row {line}.neighbor x log", tolerance)
            _close(row["received_neighbor_observer"], ns, f"raw row {line}.neighbor shat log", tolerance)
            observer_z = qs - ns
            controller_z = qx - nx
        shat_next = prediction - gamma * observer_z
        follower = models[node]
        command = (
            prediction - follower["a"] * x - follower["c"] - kappa * controller_z
        ) / follower["b"]
        clipped = min(max(command, umin), umax)
        quantized = min(max(_quantize(clipped, du), umin), umax)
        direction = 0 if quantized == 0 else (1 if quantized > 0 else -1)
        register = abs(quantized / du)
        saturated = command < umin or command > umax
        _close(row["observer_innovation"], observer_z, f"raw row {line}.observer innovation", tolerance)
        _close(row["observer_state_post"], shat_next, f"raw row {line}.shat[k+1]", tolerance)
        _close(row["controller_network_innovation"], controller_z, f"raw row {line}.controller innovation", tolerance)
        for field in ("follower_u_nominal", "u_command", "u_pre_saturation"):
            _close(row[field], command, f"raw row {line}.{field}", tolerance)
        _close(row["follower_u_clipped_float"], clipped, f"raw row {line}.clipped", tolerance)
        for field in ("follower_u_quantised_signed", "u_post_saturation"):
            _close(row[field], quantized, f"raw row {line}.{field}", tolerance)
        _close(row["pwm_register"], register, f"raw row {line}.pwm_register", tolerance)
        _close(row["pwm_count"], register, f"raw row {line}.pwm_count", tolerance)
        _close(row["pwm_magnitude"], abs(quantized), f"raw row {line}.pwm_magnitude", tolerance)
        if _integer(row["direction_bit"], f"raw row {line}.direction") != direction:
            raise ContractError(f"raw row {line} signed direction identity fails")
        if _boolean(row["saturation_flag"], f"raw row {line}.saturation") != saturated:
            raise ContractError(f"raw row {line} saturation flag mismatch")
        checked += 1
    if checked == 0:
        raise ContractError("raw log contains no observer-loop follower records")
    run_nodes: dict[str, set[int]] = {}
    for run, node, _tick in seen:
        run_nodes.setdefault(run, set()).add(node)
    incomplete = sorted(run for run, nodes in run_nodes.items() if nodes != {1, 2})
    if incomplete:
        raise ContractError(f"observer-loop runs lack both followers: {incomplete}")
    used = 2 * checked
    if len(packets) != used:
        raise ContractError("packet capture contains missing or unconsumed events")
    return {
        "status": "PASS",
        "classification": CLASSIFICATION,
        "hardware_origin_authenticated": False,
        "physical_result": False,
        "contains_outcomes": False,
        "parameter_payload_sha256": payload_hash,
        "parameter_file_sha256": parameter_file_sha256,
        "certificate_report_sha256": certificate_report_sha256,
        "checked_raw_record_count": checked,
        "checked_packet_event_count": packet_count,
        "checks": [
            "pre-outcome certificate byte binding",
            "observer-loop ACL and target non-leakage",
            "packet and timestamp causality",
            "observer/controller causal replay using shat[k]",
            "saturation, quantisation and applied-PWM identities",
        ],
    }


def _csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise ContractError(f"cannot read CSV {path}: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--certificate-report", type=Path, required=True)
    parser.add_argument("--raw-log", type=Path, required=True)
    parser.add_argument("--packet-events", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        parameters = load_json(args.parameters)
        schema = load_json(args.schema)
        validate_schema_document(schema)
        validate_completed(parameters, schema, args.parameters)
        certificate = load_json(args.certificate_report)
        report = validate_replay(
            parameters, certificate, sha256_file(args.certificate_report),
            _csv_rows(args.raw_log), _csv_rows(args.packet_events),
            parameter_file_sha256=sha256_file(args.parameters),
        )
        output = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.report:
            args.report.write_text(output, encoding="utf-8")
        print(output, end="")
        return 0
    except (ContractError, ValueError, KeyError, TypeError) as exc:
        print(f"CAUSAL_REPLAY_FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover
        print(f"CAUSAL_REPLAY_INTERNAL_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
