#!/usr/bin/env python3
"""Fail-closed validator for future PHYS-E2E-OCT-R34 hardware evidence."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

from contract_common import (
    ARMS, PROTOCOL_ID, ContractError, load_json, parse_bool, parse_float,
    parse_int, read_csv, require_columns, require_sha256, resolve_beneath,
    sha256_file,
)

SCHEDULE_COLUMNS = [
    "protocol_id", "schedule_status", "block_id", "profile_id", "load_id",
    "repeat_id", "arm", "acquisition_order", "planned_run_id",
    "randomisation_seed",
]
LEDGER_COLUMNS = [
    "protocol_id", "run_id", "block_id", "profile_id", "load_id", "repeat_id",
    "arm", "acquisition_order", "actual_status", "stop_stage", "failure_reason",
    "independent_hardware_reset", "reset_event_id", "started_at_utc", "ended_at_utc",
    "expected_nodes", "observed_nodes", "raw_rows", "deadline_miss_count",
    "saturation_count", "safety_event_count", "exclusion_requested",
    "exclusion_reason", "retain_in_analysis", "operator_id",
    "firmware_manifest_sha256", "parameter_sha256", "raw_file_count",
]
MANIFEST_COLUMNS = [
    "relative_path", "sha256", "bytes", "protocol_id", "run_id", "block_id",
    "arm", "node_id", "role", "row_count", "status", "firmware_sha256",
    "parameter_sha256",
]
RESET_COLUMNS = [
    "protocol_id", "reset_event_id", "run_id", "acquisition_order",
    "event_time_utc", "event_type", "device_log_sha256s", "host_log_sha256",
    "operator_id", "hardware_power_cycle_confirmed", "load_reset_confirmed",
    "cooldown_complete",
]
STATUS = {"completed", "failed", "aborted"}
ROLES = {0: "physical_leader", 1: "physical_follower", 2: "physical_follower"}


def check_schedule(path: Path, manifest_path: Path) -> tuple[dict[str, dict[str, str]], int]:
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "PHYS-E2E-OCT-R34-randomisation-v1":
        raise ContractError("unexpected randomisation manifest schema")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("randomisation manifest protocol mismatch")
    if manifest.get("status") != "future_plan_only_not_executed":
        raise ContractError("randomisation manifest must remain a future plan")
    if manifest.get("contains_measurements") is not False or manifest.get("contains_outcomes") is not False:
        raise ContractError("randomisation manifest may not contain measurements or outcomes")
    require_sha256(str(manifest.get("schedule_sha256", "")), "schedule_sha256")
    if sha256_file(path) != manifest["schedule_sha256"]:
        raise ContractError("randomisation schedule SHA-256 mismatch")

    header, rows = read_csv(path)
    require_columns(header, SCHEDULE_COLUMNS, "randomisation schedule")
    if not rows:
        raise ContractError("randomisation schedule is empty")
    by_run: dict[str, dict[str, str]] = {}
    by_block: dict[str, list[dict[str, str]]] = {}
    orders: list[int] = []
    seeds: set[int] = set()
    for row in rows:
        if row["protocol_id"] != PROTOCOL_ID or row["schedule_status"] != "planned_not_executed":
            raise ContractError("schedule row is not a PHYS-E2E future plan")
        if row["arm"] not in ARMS:
            raise ContractError(f"unknown scheduled arm: {row['arm']!r}")
        run_id = row["planned_run_id"]
        if run_id in by_run:
            raise ContractError(f"duplicate planned_run_id: {run_id}")
        by_run[run_id] = row
        by_block.setdefault(row["block_id"], []).append(row)
        orders.append(parse_int(row["acquisition_order"], f"{run_id}.acquisition_order"))
        seeds.add(parse_int(row["randomisation_seed"], f"{run_id}.randomisation_seed"))
    if sorted(orders) != list(range(1, len(rows) + 1)):
        raise ContractError("acquisition_order must be exactly 1..planned_run_count")
    if len(seeds) != 1 or next(iter(seeds)) != manifest.get("randomisation_seed"):
        raise ContractError("schedule randomisation seed mismatch")
    for block, block_rows in by_block.items():
        if sorted(row["arm"] for row in block_rows) != sorted(ARMS):
            raise ContractError(f"block {block} does not contain all three arms exactly once")
        identities = {(r["profile_id"], r["load_id"], r["repeat_id"]) for r in block_rows}
        if len(identities) != 1:
            raise ContractError(f"block {block} mixes profile/load/repeat identities")
    if manifest.get("planned_run_count") != len(rows) or manifest.get("planned_block_count") != len(by_block):
        raise ContractError("randomisation manifest planned counts mismatch")
    return by_run, len(by_block)


def check_ledger(path: Path, schedule: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    header, rows = read_csv(path)
    require_columns(header, LEDGER_COLUMNS, "run ledger")
    if not rows:
        raise ContractError("run ledger is empty; no execution can be validated")
    by_run: dict[str, dict[str, str]] = {}
    resets: set[str] = set()
    for row in rows:
        run_id = row["run_id"]
        if run_id in by_run:
            raise ContractError(f"duplicate run ledger row: {run_id}")
        if run_id not in schedule:
            raise ContractError(f"unregistered run in ledger: {run_id}")
        planned = schedule[run_id]
        for field, planned_field in (
            ("block_id", "block_id"), ("profile_id", "profile_id"),
            ("load_id", "load_id"), ("repeat_id", "repeat_id"),
            ("arm", "arm"), ("acquisition_order", "acquisition_order"),
        ):
            if row[field] != planned[planned_field]:
                raise ContractError(f"{run_id} ledger/schedule mismatch: {field}")
        if row["protocol_id"] != PROTOCOL_ID:
            raise ContractError(f"{run_id} protocol mismatch")
        if row["actual_status"] not in STATUS:
            raise ContractError(f"{run_id} invalid actual_status")
        if row["actual_status"] != "completed" and not row["failure_reason"].strip():
            raise ContractError(f"{run_id} failed/aborted without failure_reason")
        if not parse_bool(row["independent_hardware_reset"], f"{run_id}.independent_hardware_reset"):
            raise ContractError(f"{run_id} is not an independently reset hardware trial")
        reset = row["reset_event_id"].strip()
        if not reset or reset in resets:
            raise ContractError(f"{run_id} reset_event_id is missing or reused")
        resets.add(reset)
        if not parse_bool(row["retain_in_analysis"], f"{run_id}.retain_in_analysis"):
            raise ContractError(f"{run_id} is deleted from analysis; failed runs must be retained")
        parse_bool(row["exclusion_requested"], f"{run_id}.exclusion_requested")
        if row["expected_nodes"] != "0|1|2":
            raise ContractError(f"{run_id} expected_nodes must be 0|1|2")
        if row["observed_nodes"] != "0|1|2" and row["stop_stage"] != "pre_actuation":
            raise ContractError(f"{run_id} lacks three-node coverage")
        require_sha256(row["firmware_manifest_sha256"], f"{run_id}.firmware_manifest_sha256")
        require_sha256(row["parameter_sha256"], f"{run_id}.parameter_sha256")
        for count in ("raw_rows", "deadline_miss_count", "saturation_count",
                      "safety_event_count", "raw_file_count"):
            if parse_int(row[count], f"{run_id}.{count}") < 0:
                raise ContractError(f"{run_id}.{count} must be nonnegative")
        by_run[run_id] = row
    missing = sorted(set(schedule) - set(by_run))
    if missing:
        raise ContractError(f"scheduled runs missing from ledger, including failures: {missing[:8]}")
    return by_run


def _is_missing(value: str) -> bool:
    return (value or "").strip() in {"", "NA"}


def _utc(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} is not an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ContractError(f"{label} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def check_raw_file(
    path: Path,
    expected: dict[str, str],
    ledger: dict[str, str],
    required_columns: list[str],
    monotonic_columns: list[str],
    sha_columns: list[str],
) -> tuple[int, bool, set[str], set[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        if len(header) != len(set(header)):
            raise ContractError(f"raw log has duplicate columns: {path}")
        require_columns(header, required_columns, f"raw log {path.name}")
        previous: dict[str, int] = {}
        rows = 0
        has_stop_reason = False
        device_log_hashes: set[str] = set()
        host_log_hashes: set[str] = set()
        node_id = parse_int(expected["node_id"], "manifest.node_id")
        role = ROLES.get(node_id)
        if expected["role"] != role:
            raise ContractError(f"manifest role mismatch for node {node_id}: {path}")
        for row in reader:
            rows += 1
            identities = {
                "protocol_id": PROTOCOL_ID,
                "run_id": expected["run_id"],
                "arm": expected["arm"],
                "block_id": expected["block_id"],
                "node_id": str(node_id),
                "role": role,
            }
            for field, value in identities.items():
                if row[field] != value:
                    raise ContractError(f"{path}:{rows} identity mismatch: {field}")
            for field in monotonic_columns:
                value = parse_int(row[field], f"{path}:{rows}.{field}")
                if field in previous and value <= previous[field]:
                    raise ContractError(f"{path}:{rows}.{field} is not strictly increasing")
                previous[field] = value
            for field in sha_columns:
                require_sha256(row[field], f"{path}:{rows}.{field}")
            device_log_hashes.add(row["node_log_sha256"])
            host_log_hashes.add(row["host_log_sha256"])
            if row["firmware_sha256"] != expected["firmware_sha256"]:
                raise ContractError(f"{path}:{rows} firmware identity mismatch")
            if row["frozen_parameter_sha256"] != expected["parameter_sha256"]:
                raise ContractError(f"{path}:{rows} parameter identity mismatch")
            if row["frozen_parameter_sha256"] != ledger["parameter_sha256"]:
                raise ContractError(f"{path}:{rows} parameter/ledger mismatch")

            applied = parse_float(row["u_post_saturation"], f"{path}:{rows}.u_post_saturation")
            magnitude = parse_float(row["pwm_magnitude"], f"{path}:{rows}.pwm_magnitude")
            register = parse_float(row["pwm_register"], f"{path}:{rows}.pwm_register")
            pwm_count = parse_float(row["pwm_count"], f"{path}:{rows}.pwm_count")
            direction = parse_int(row["direction_bit"], f"{path}:{rows}.direction_bit")
            if direction not in {-1, 0, 1}:
                raise ContractError(f"{path}:{rows} direction_bit must be -1, 0 or 1")
            if abs(abs(applied) - magnitude) > 1e-9 or abs(register - magnitude) > 1e-9:
                raise ContractError(f"{path}:{rows} post-saturation/PWM magnitude identity fails")
            if abs(pwm_count - register) > 1e-9:
                raise ContractError(f"{path}:{rows} pwm_count/register identity fails")
            expected_direction = 0 if abs(applied) <= 1e-12 else (1 if applied > 0 else -1)
            if direction != expected_direction:
                raise ContractError(f"{path}:{rows} direction does not match applied input")
            if role == "physical_follower":
                quantized = parse_float(row["follower_u_quantised_signed"],
                                        f"{path}:{rows}.follower_u_quantised_signed")
                if abs(applied - quantized) > 1e-9:
                    raise ContractError(f"{path}:{rows} applied input is not the logged quantised follower input")
            else:
                leader_applied = parse_float(row["leader_u_applied"],
                                             f"{path}:{rows}.leader_u_applied")
                if abs(applied - leader_applied) > 1e-9:
                    raise ContractError(f"{path}:{rows} leader applied-input identity fails")

            arm = expected["arm"]
            if arm == "observer_loop" and role == "physical_follower":
                parse_float(row["observer_state_post"], f"{path}:{rows}.observer_state_post")
                parse_float(row["observer_innovation"], f"{path}:{rows}.observer_innovation")
            if arm == "oracle_target" and role == "physical_follower":
                parse_float(row["received_leader_state"], f"{path}:{rows}.received_leader_state")
            if arm == "no_target" and role == "physical_follower":
                if parse_bool(row["pin_state"], f"{path}:{rows}.pin_state"):
                    raise ContractError(f"{path}:{rows} no_target arm has active pinning")
                if not _is_missing(row["received_leader_state"]):
                    raise ContractError(f"{path}:{rows} no_target arm exposes leader state")
            if row["stop_reason"].strip() not in {"", "NA", "normal_completion"}:
                has_stop_reason = True
    return rows, has_stop_reason, device_log_hashes, host_log_hashes


def check_raw(
    raw_dir: Path,
    manifest_path: Path,
    ledgers: dict[str, dict[str, str]],
    schedule: dict[str, dict[str, str]],
    schema_path: Path,
) -> tuple[int, int, int, dict[str, set[str]], dict[str, set[str]]]:
    schema = load_json(schema_path)
    if schema.get("protocol_id") != PROTOCOL_ID:
        raise ContractError("raw-data schema protocol mismatch")
    required = list(schema.get("required_columns", []))
    monotonic = list(schema.get("strictly_increasing_within_run_node", []))
    sha_columns = list(schema.get("sha256_columns", []))
    if len(required) < 70:
        raise ContractError("raw-data schema is unexpectedly incomplete")

    header, entries = read_csv(manifest_path)
    require_columns(header, MANIFEST_COLUMNS, "raw-file manifest")
    expected_pairs = {(run_id, node) for run_id in schedule for node in ROLES}
    observed_pairs: set[tuple[str, int]] = set()
    total_rows = 0
    failed_raw_stop_evidence: dict[str, bool] = {run: False for run in schedule}
    per_run_rows: dict[str, int] = {run: 0 for run in schedule}
    per_run_files: dict[str, int] = {run: 0 for run in schedule}
    per_run_device_hashes: dict[str, set[str]] = {run: set() for run in schedule}
    per_run_host_hashes: dict[str, set[str]] = {run: set() for run in schedule}
    for entry in entries:
        run_id = entry["run_id"]
        if run_id not in schedule:
            raise ContractError(f"raw manifest contains unregistered run: {run_id}")
        node = parse_int(entry["node_id"], f"{run_id}.node_id")
        pair = (run_id, node)
        if pair not in expected_pairs or pair in observed_pairs:
            raise ContractError(f"raw manifest missing/duplicate unexpected run-node pair: {pair}")
        observed_pairs.add(pair)
        planned = schedule[run_id]
        ledger = ledgers[run_id]
        for field in ("block_id", "arm"):
            if entry[field] != planned[field]:
                raise ContractError(f"{run_id}/node{node} raw manifest mismatch: {field}")
        if entry["protocol_id"] != PROTOCOL_ID or entry["status"] != ledger["actual_status"]:
            raise ContractError(f"{run_id}/node{node} raw manifest identity/status mismatch")
        require_sha256(entry["sha256"], f"{run_id}/node{node}.sha256")
        require_sha256(entry["firmware_sha256"], f"{run_id}/node{node}.firmware_sha256")
        require_sha256(entry["parameter_sha256"], f"{run_id}/node{node}.parameter_sha256")
        path = resolve_beneath(raw_dir, entry["relative_path"])
        if not path.is_file():
            raise ContractError(f"raw file does not exist: {path}")
        if path.stat().st_size != parse_int(entry["bytes"], f"{path}.bytes"):
            raise ContractError(f"raw file byte-size mismatch: {path}")
        if sha256_file(path) != entry["sha256"]:
            raise ContractError(f"raw file SHA-256 mismatch: {path}")
        row_count, has_stop, device_hashes, host_hashes = check_raw_file(
            path, entry, ledger, required, monotonic, sha_columns)
        if row_count != parse_int(entry["row_count"], f"{path}.row_count"):
            raise ContractError(f"raw file row-count mismatch: {path}")
        if row_count == 0 and not (ledger["actual_status"] in {"failed", "aborted"}
                                   and ledger["stop_stage"] == "pre_actuation"):
            raise ContractError(f"zero-row raw log is allowed only for explicit pre-actuation failures: {path}")
        total_rows += row_count
        per_run_rows[run_id] += row_count
        per_run_files[run_id] += 1
        per_run_device_hashes[run_id].update(device_hashes)
        per_run_host_hashes[run_id].update(host_hashes)
        failed_raw_stop_evidence[run_id] |= has_stop
    missing_pairs = sorted(expected_pairs - observed_pairs)
    if missing_pairs:
        raise ContractError(f"raw manifest does not retain all scheduled run-node files: {missing_pairs[:8]}")
    for run_id, ledger in ledgers.items():
        if per_run_rows[run_id] != parse_int(ledger["raw_rows"], f"{run_id}.raw_rows"):
            raise ContractError(f"{run_id} ledger/raw row-count mismatch")
        if per_run_files[run_id] != parse_int(ledger["raw_file_count"], f"{run_id}.raw_file_count"):
            raise ContractError(f"{run_id} ledger/raw file-count mismatch")
        if ledger["actual_status"] in {"failed", "aborted"} and per_run_rows[run_id] > 0:
            if not failed_raw_stop_evidence[run_id]:
                raise ContractError(f"{run_id} failed/aborted but raw logs contain no non-normal stop reason")
    failed = sum(row["actual_status"] == "failed" for row in ledgers.values())
    aborted = sum(row["actual_status"] == "aborted" for row in ledgers.values())
    return total_rows, failed, aborted, per_run_device_hashes, per_run_host_hashes


def check_reset_events(path: Path, ledgers: dict[str, dict[str, str]],
                       device_hashes: dict[str, set[str]],
                       host_hashes: dict[str, set[str]]) -> None:
    header, rows = read_csv(path)
    require_columns(header, RESET_COLUMNS, "hardware reset-event ledger")
    if len(rows) != len(ledgers):
        raise ContractError("reset-event ledger must contain exactly one event per run")
    by_run: dict[str, dict[str, str]] = {}
    chronological: list[tuple[int, datetime, datetime, datetime, str]] = []
    for row in rows:
        run_id = row["run_id"]
        if run_id not in ledgers or run_id in by_run:
            raise ContractError(f"reset-event ledger has unknown/duplicate run: {run_id}")
        ledger = ledgers[run_id]
        if row["protocol_id"] != PROTOCOL_ID:
            raise ContractError(f"{run_id} reset-event protocol mismatch")
        if row["reset_event_id"] != ledger["reset_event_id"]:
            raise ContractError(f"{run_id} reset_event_id is not bound to ledger")
        if row["acquisition_order"] != ledger["acquisition_order"]:
            raise ContractError(f"{run_id} reset-event acquisition order mismatch")
        if row["event_type"] != "HARDWARE_RESET_COMPLETE":
            raise ContractError(f"{run_id} lacks HARDWARE_RESET_COMPLETE evidence")
        if not row["operator_id"].strip() or row["operator_id"] != ledger["operator_id"]:
            raise ContractError(f"{run_id} reset operator is absent or differs from ledger")
        for flag in ("hardware_power_cycle_confirmed", "load_reset_confirmed",
                     "cooldown_complete"):
            if not parse_bool(row[flag], f"{run_id}.{flag}"):
                raise ContractError(f"{run_id} reset independence flag is false: {flag}")
        event_time = _utc(row["event_time_utc"], f"{run_id}.event_time_utc")
        started = _utc(ledger["started_at_utc"], f"{run_id}.started_at_utc")
        ended = _utc(ledger["ended_at_utc"], f"{run_id}.ended_at_utc")
        if not event_time <= started <= ended:
            raise ContractError(f"{run_id} reset/start/end timestamps are not causal")
        declared_devices = set(filter(None, row["device_log_sha256s"].split("|")))
        for value in declared_devices:
            require_sha256(value, f"{run_id}.device_log_sha256s")
        require_sha256(row["host_log_sha256"], f"{run_id}.host_log_sha256")
        if declared_devices != device_hashes[run_id]:
            raise ContractError(f"{run_id} reset event is not bound to exact device-log hashes")
        if host_hashes[run_id] != {row["host_log_sha256"]}:
            raise ContractError(f"{run_id} reset event is not bound to exact host-log hash")
        chronological.append((parse_int(row["acquisition_order"],
                                        f"{run_id}.acquisition_order"),
                              event_time, started, ended, run_id))
        by_run[run_id] = row
    chronological.sort()
    if [item[0] for item in chronological] != list(range(1, len(chronological) + 1)):
        raise ContractError("executed acquisition order is not exactly 1..run_count")
    for previous, current in zip(chronological, chronological[1:]):
        if not (previous[1] < previous[2] <= previous[3] < current[1] < current[2] <= current[3]):
            raise ContractError(
                "reset/start/end times do not follow non-overlapping randomised acquisition order")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--schedule-manifest", type=Path, required=True)
    parser.add_argument("--run-ledger", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--reset-events", type=Path, required=True,
                        help="Timestamped reset ledger bound to device/host log hashes.")
    parser.add_argument("--schema", type=Path,
                        default=Path(__file__).with_name("RAW_DATA_COLUMNS.json"))
    parser.add_argument("--report", type=Path,
                        help="Optional new JSON report path; overwrite is prohibited.")
    args = parser.parse_args()
    try:
        explicit_artifacts = {
            "raw directory": args.raw_dir, "schedule": args.schedule,
            "schedule manifest": args.schedule_manifest, "run ledger": args.run_ledger,
            "raw manifest": args.raw_manifest, "reset events": args.reset_events,
            "raw schema": args.schema,
        }
        for label, path in explicit_artifacts.items():
            resolved = path.resolve()
            expected = resolved.is_dir() if label == "raw directory" else resolved.is_file()
            if not expected:
                raise ContractError(f"explicit {label} artifact is missing or wrong type: {path}")
        schedule, block_count = check_schedule(args.schedule, args.schedule_manifest)
        ledger = check_ledger(args.run_ledger, schedule)
        dense_rows, failed, aborted, device_hashes, host_hashes = check_raw(
            args.raw_dir, args.raw_manifest, ledger, schedule, args.schema)
        check_reset_events(args.reset_events, ledger, device_hashes, host_hashes)
        report = {
            "schema": "PHYS-E2E-OCT-R34-raw-validation-v1",
            "passed": True,
            "protocol_id": PROTOCOL_ID,
            "n_independent_hardware_runs": len(ledger),
            "n_matched_hardware_blocks": block_count,
            "n_dense_controller_cycle_rows_not_independent": dense_rows,
            "failed_runs_retained": failed,
            "aborted_runs_retained": aborted,
            "independent_unit": "separately_reset_hardware_trial",
            "uncertainty_unit": "matched_hardware_block",
            "dense_samples_are_n": False,
            "schedule_sha256": sha256_file(args.schedule),
            "schedule_manifest_sha256": sha256_file(args.schedule_manifest),
            "run_ledger_sha256": sha256_file(args.run_ledger),
            "raw_manifest_sha256": sha256_file(args.raw_manifest),
            "reset_events_sha256": sha256_file(args.reset_events),
            "raw_schema_sha256": sha256_file(args.schema),
            "warning": "Passing validates evidence structure and identity, not scientific efficacy.",
        }
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.report:
            with args.report.open("x", encoding="utf-8") as handle:
                handle.write(payload)
        print(payload, end="")
        return 0
    except (ContractError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
