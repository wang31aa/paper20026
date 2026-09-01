#!/usr/bin/env python3
"""Validate structural consistency of a byte-bound, locally signed bundle.

This validator never acquires data and never updates EVIDENCE_STATUS.json.
Exit zero means only that the supplied immutable bundle is structurally
consistent. This implementation has no hardware-rooted or independently
reviewed trust anchor and therefore can never grant manuscript evidence
eligibility or score credit.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import hmac
import json
from pathlib import Path
import re
import sys
from typing import Any


PROTOCOL_ID = "PHYS-E2E-OCT-R34"
GATE_SCHEMA = "PHYS-E2E-OCT-R34-promotion-gate-v1"
APPARATUS_SCHEMA = "PHYS-E2E-OCT-R34-apparatus-v1"
ARTIFACT_SCHEMA = "PHYS-E2E-OCT-R34-artifact-index-v1"
REPORT_SCHEMA = "PHYS-E2E-OCT-R34-signed-pass-v1"
SIGNER_SCHEMA = "PHYS-E2E-OCT-R34-signer-registry-v1"
ARMS = ("observer_loop", "oracle_target", "no_target")
REQUIRED_REPORTS = {
    "physical_raw_validation": "physical_validator",
    "causal_equation_replay": "equation_reviewer",
    "certificate_validation": "certificate_reviewer",
    "independent_confirmation": "independent_confirmer",
}
REQUIRED_ARTIFACTS = {
    "frozen_parameters",
    "raw_file_manifest",
    "firmware_binary_manifest",
    "leader_firmware_binary",
    "follower1_firmware_binary",
    "follower2_firmware_binary",
    "host_executable_or_container",
    "certificate_inputs",
}
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
LEDGER_REQUIRED = {
    "protocol_id", "run_id", "block_id", "profile_id", "load_id", "repeat_id",
    "arm", "acquisition_order", "actual_status", "retain_in_analysis",
    "firmware_manifest_sha256", "parameter_sha256",
}
SCHEDULE_REQUIRED = {
    "protocol_id", "schedule_status", "block_id", "profile_id", "load_id",
    "repeat_id", "arm", "acquisition_order", "planned_run_id", "randomisation_seed",
}
IDENTITY_REQUIRED = {
    "protocol_id", "run_id", "block_id", "arm", "node_id", "role",
    "apparatus_id", "hardware_serial", "motor_serial", "encoder_serial",
    "calibration_sha256", "apparatus_manifest_sha256",
}
NODE_ROLES = {0: "physical_leader", 1: "physical_follower", 2: "physical_follower"}


class GateError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError(f"JSON root must be object: {path}")
    return value


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = list(reader.fieldnames or [])
            rows = list(reader)
    except OSError as exc:
        raise GateError(f"cannot read CSV {path}: {exc}") from exc
    if not header or len(header) != len(set(header)):
        raise GateError(f"missing or duplicate CSV header: {path}")
    return header, rows


def require_columns(header: list[str], required: set[str], label: str) -> None:
    missing = sorted(required - set(header))
    if missing:
        raise GateError(f"{label} missing columns: {missing}")


def require_sha(value: str, label: str) -> None:
    if not SHA_RE.fullmatch(value or ""):
        raise GateError(f"{label} is not lowercase SHA-256")


def parse_bool(value: str, label: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise GateError(f"{label} must be true or false")


def parse_time(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GateError(f"{label} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise GateError(f"{label} lacks timezone")
    return parsed


def check_apparatus(path: Path) -> tuple[dict[str, Any], str, dict[int, dict[str, str]]]:
    item = load_json(path)
    if item.get("schema") != APPARATUS_SCHEMA or item.get("protocol_id") != PROTOCOL_ID:
        raise GateError("apparatus manifest schema/protocol mismatch")
    if item.get("status") != "frozen_before_primary":
        raise GateError("apparatus manifest is not frozen_before_primary")
    if not item.get("apparatus_id") or not item.get("topology_id"):
        raise GateError("apparatus manifest lacks apparatus/topology identity")
    parse_time(str(item.get("frozen_at_utc", "")), "apparatus.frozen_at_utc")
    nodes = item.get("nodes")
    if not isinstance(nodes, list) or len(nodes) != 3:
        raise GateError("apparatus manifest must contain exactly nodes 0,1,2")
    by_node: dict[int, dict[str, str]] = {}
    for raw in nodes:
        if not isinstance(raw, dict):
            raise GateError("apparatus node must be object")
        try:
            node = int(raw.get("node_id"))
        except (TypeError, ValueError) as exc:
            raise GateError("apparatus node_id invalid") from exc
        if node in by_node or node not in NODE_ROLES or raw.get("role") != NODE_ROLES[node]:
            raise GateError(f"apparatus node/role invalid: {node}")
        for field in ("hardware_serial", "motor_serial", "encoder_serial"):
            if not str(raw.get(field, "")).strip():
                raise GateError(f"apparatus node {node} lacks {field}")
        require_sha(str(raw.get("calibration_sha256", "")),
                    f"apparatus node {node} calibration_sha256")
        by_node[node] = {key: str(value) for key, value in raw.items()}
    if set(by_node) != set(NODE_ROLES):
        raise GateError("apparatus nodes must be exactly 0,1,2")
    return item, sha256_file(path), by_node


def check_schedule(path: Path, manifest_path: Path) -> tuple[dict[str, dict[str, str]], str, str]:
    manifest = load_json(manifest_path)
    if manifest.get("schema") != "PHYS-E2E-OCT-R34-randomisation-v1":
        raise GateError("schedule manifest schema mismatch")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise GateError("schedule protocol mismatch")
    if (manifest.get("status") != "future_plan_only_not_executed"
            or manifest.get("contains_measurements") is not False
            or manifest.get("contains_outcomes") is not False):
        raise GateError("schedule manifest is not a frozen outcome-free plan")
    profiles = manifest.get("profiles")
    loads = manifest.get("loads")
    blocks_per_cell = manifest.get("matched_blocks_per_profile_load")
    if (not isinstance(profiles, list) or not profiles
            or len(profiles) != len(set(profiles))
            or not all(isinstance(item, str) and item for item in profiles)):
        raise GateError("schedule manifest profiles must be a nonempty unique list")
    if (not isinstance(loads, list) or not loads
            or len(loads) != len(set(loads))
            or not all(isinstance(item, str) and item for item in loads)):
        raise GateError("schedule manifest loads must be a nonempty unique list")
    if not isinstance(blocks_per_cell, int) or isinstance(blocks_per_cell, bool):
        raise GateError("matched_blocks_per_profile_load must be an integer")
    if blocks_per_cell < 10:
        raise GateError("matched_blocks_per_profile_load must be at least 10")
    if manifest.get("arms") != list(ARMS):
        raise GateError("schedule manifest arms mismatch")
    parse_time(str(manifest.get("frozen_at_utc", "")), "schedule.frozen_at_utc")
    seed = manifest.get("randomisation_seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise GateError("schedule manifest randomisation_seed invalid")
    require_sha(str(manifest.get("schedule_sha256", "")), "schedule_sha256")
    schedule_hash = sha256_file(path)
    if schedule_hash != manifest["schedule_sha256"]:
        raise GateError("schedule byte hash mismatch")
    header, rows = read_csv(path)
    require_columns(header, SCHEDULE_REQUIRED, "schedule")
    if not rows:
        raise GateError("schedule empty")
    by_run: dict[str, dict[str, str]] = {}
    blocks: dict[str, list[dict[str, str]]] = {}
    orders: list[int] = []
    for row in rows:
        run = row["planned_run_id"]
        if row["protocol_id"] != PROTOCOL_ID or row["schedule_status"] != "planned_not_executed":
            raise GateError(f"schedule row invalid: {run}")
        if run in by_run or row["arm"] not in ARMS:
            raise GateError(f"duplicate run or invalid arm: {run}")
        by_run[run] = row
        blocks.setdefault(row["block_id"], []).append(row)
        try:
            orders.append(int(row["acquisition_order"]))
            row_seed = int(row["randomisation_seed"])
        except ValueError as exc:
            raise GateError(f"schedule numeric identity invalid: {run}") from exc
        if row_seed != seed:
            raise GateError(f"schedule randomisation seed mismatch: {run}")
    if sorted(orders) != list(range(1, len(rows) + 1)):
        raise GateError("schedule acquisition_order must be exactly 1..planned_run_count")
    for block, group in blocks.items():
        if sorted(row["arm"] for row in group) != sorted(ARMS):
            raise GateError(f"block {block} lacks exactly three arms")
        identities = {(r["profile_id"], r["load_id"], r["repeat_id"]) for r in group}
        if len(identities) != 1:
            raise GateError(f"block {block} mixes experimental identity")
    cells: dict[tuple[str, str], set[str]] = {}
    cell_repeats: dict[tuple[str, str], set[str]] = {}
    for block, group in blocks.items():
        identity = (group[0]["profile_id"], group[0]["load_id"])
        cells.setdefault(identity, set()).add(block)
        repeat = group[0]["repeat_id"]
        if repeat in cell_repeats.setdefault(identity, set()):
            raise GateError(f"schedule cell {identity} reuses repeat_id {repeat}")
        cell_repeats[identity].add(repeat)
    expected_cells = {(profile, load) for profile in profiles for load in loads}
    if set(cells) != expected_cells:
        raise GateError("schedule does not cover the exact profile/load Cartesian design")
    for cell, cell_blocks in cells.items():
        if len(cell_blocks) != blocks_per_cell:
            raise GateError(
                f"schedule cell {cell} has {len(cell_blocks)} blocks, expected {blocks_per_cell}")
        if len(cell_repeats[cell]) != blocks_per_cell:
            raise GateError(f"schedule cell {cell} lacks unique repeat identities")
    expected_block_count = len(profiles) * len(loads) * blocks_per_cell
    if (len(blocks) != expected_block_count
            or manifest.get("planned_run_count") != len(rows)
            or manifest.get("planned_block_count") != len(blocks)):
        raise GateError("schedule manifest counts mismatch")
    return by_run, schedule_hash, sha256_file(manifest_path)


def check_ledger(path: Path, schedule: dict[str, dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    header, rows = read_csv(path)
    require_columns(header, LEDGER_REQUIRED, "ledger")
    by_run: dict[str, dict[str, str]] = {}
    completed = {arm: 0 for arm in ARMS}
    total = {arm: 0 for arm in ARMS}
    failed = 0
    aborted = 0
    for row in rows:
        run = row["run_id"]
        if run in by_run or run not in schedule:
            raise GateError(f"ledger duplicate/unregistered run: {run}")
        planned = schedule[run]
        for field, pfield in (("block_id", "block_id"), ("profile_id", "profile_id"),
                              ("load_id", "load_id"), ("repeat_id", "repeat_id"),
                              ("arm", "arm"), ("acquisition_order", "acquisition_order")):
            if row[field] != planned[pfield]:
                raise GateError(f"ledger/schedule mismatch {run}.{field}")
        if row["protocol_id"] != PROTOCOL_ID or row["actual_status"] not in {"completed", "failed", "aborted"}:
            raise GateError(f"ledger status/protocol invalid: {run}")
        if not parse_bool(row["retain_in_analysis"], f"{run}.retain_in_analysis"):
            raise GateError(f"ledger excludes run: {run}")
        require_sha(row["firmware_manifest_sha256"], f"{run}.firmware_manifest_sha256")
        require_sha(row["parameter_sha256"], f"{run}.parameter_sha256")
        arm = row["arm"]
        total[arm] += 1
        completed[arm] += row["actual_status"] == "completed"
        failed += row["actual_status"] == "failed"
        aborted += row["actual_status"] == "aborted"
        by_run[run] = row
    missing = sorted(set(schedule) - set(by_run))
    if missing:
        raise GateError(f"ledger misses scheduled runs: {missing[:8]}")
    if any(completed[arm] == 0 for arm in ARMS):
        raise GateError("at least one completed run is required in every arm")
    counts = {
        "registered_runs": len(by_run),
        "registered_blocks": len({row["block_id"] for row in rows}),
        "total_runs_by_arm": total,
        "completed_runs_by_arm": completed,
        "failed_runs": failed,
        "aborted_runs": aborted,
    }
    return by_run, counts


def check_run_identity(path: Path, schedule: dict[str, dict[str, str]],
                       ledger: dict[str, dict[str, str]], apparatus: dict[str, Any],
                       apparatus_hash: str, nodes: dict[int, dict[str, str]]) -> None:
    header, rows = read_csv(path)
    require_columns(header, IDENTITY_REQUIRED, "run identity")
    expected = {(run, node) for run in schedule for node in NODE_ROLES}
    seen: set[tuple[str, int]] = set()
    per_block_node: dict[tuple[str, int], set[tuple[str, ...]]] = {}
    for row in rows:
        run = row["run_id"]
        try:
            node = int(row["node_id"])
        except ValueError as exc:
            raise GateError("run identity node_id invalid") from exc
        pair = (run, node)
        if pair not in expected or pair in seen:
            raise GateError(f"run identity unexpected/duplicate pair: {pair}")
        seen.add(pair)
        planned = schedule[run]
        if row["protocol_id"] != PROTOCOL_ID or row["block_id"] != planned["block_id"] or row["arm"] != planned["arm"]:
            raise GateError(f"run identity schedule mismatch: {pair}")
        if row["apparatus_id"] != apparatus["apparatus_id"]:
            raise GateError(f"cross-apparatus run identity: {pair}")
        if row["role"] != NODE_ROLES[node]:
            raise GateError(f"run identity role mismatch: {pair}")
        require_sha(row["calibration_sha256"], f"{pair}.calibration_sha256")
        require_sha(row["apparatus_manifest_sha256"], f"{pair}.apparatus_manifest_sha256")
        if row["apparatus_manifest_sha256"] != apparatus_hash:
            raise GateError(f"run identity apparatus hash mismatch: {pair}")
        frozen = nodes[node]
        identity = (row["hardware_serial"], row["motor_serial"], row["encoder_serial"],
                    row["calibration_sha256"])
        expected_identity = (frozen["hardware_serial"], frozen["motor_serial"],
                             frozen["encoder_serial"], frozen["calibration_sha256"])
        if identity != expected_identity:
            raise GateError(f"run identity differs from frozen apparatus: {pair}")
        per_block_node.setdefault((row["block_id"], node), set()).add(identity)
    missing = sorted(expected - seen)
    if missing:
        raise GateError(f"run identity misses run/node pairs: {missing[:8]}")
    for key, identities in per_block_node.items():
        if len(identities) != 1:
            raise GateError(f"hardware identity changes across matched arms: {key}")
    parameter_hashes = {row["parameter_sha256"] for row in ledger.values()}
    firmware_hashes = {row["firmware_manifest_sha256"] for row in ledger.values()}
    if len(parameter_hashes) != 1 or len(firmware_hashes) != 1:
        raise GateError("parameter or firmware manifest changes within registered batch")


def check_artifacts(path: Path, expected_parameter: str, expected_firmware: str) -> tuple[dict[str, str], str]:
    index = load_json(path)
    if index.get("schema") != ARTIFACT_SCHEMA or index.get("protocol_id") != PROTOCOL_ID:
        raise GateError("artifact index schema/protocol mismatch")
    prohibited_decision_fields = {
        "score_credit",
        "score_credit_for_physical_e2e",
        "manuscript_evidence_eligible",
        "manuscript_physical_e2e_claim_allowed",
        "physical_origin_authenticated",
        "physical_evidence_eligible",
    }
    attempted = sorted(prohibited_decision_fields.intersection(index))
    if attempted:
        raise GateError(
            "artifact index may not request evidence decisions or score credit: "
            + ", ".join(attempted)
        )
    entries = index.get("artifacts")
    if not isinstance(entries, list):
        raise GateError("artifact index artifacts must be list")
    hashes: dict[str, str] = {}
    base = path.parent.resolve()
    for entry in entries:
        if not isinstance(entry, dict):
            raise GateError("artifact entry must be object")
        artifact_id = str(entry.get("artifact_id", ""))
        raw_path = str(entry.get("path", ""))
        if artifact_id in hashes or not artifact_id or not raw_path:
            raise GateError("artifact id/path missing or duplicate")
        artifact = Path(raw_path)
        if not artifact.is_absolute():
            artifact = (base / artifact).resolve()
        if not artifact.is_file():
            raise GateError(f"artifact path is not file: {artifact}")
        declared = str(entry.get("sha256", ""))
        require_sha(declared, f"artifact {artifact_id}.sha256")
        actual = sha256_file(artifact)
        if actual != declared:
            raise GateError(f"artifact byte hash mismatch: {artifact_id}")
        hashes[artifact_id] = actual
    missing = sorted(REQUIRED_ARTIFACTS - set(hashes))
    if missing:
        raise GateError(f"artifact index lacks required artifacts: {missing}")
    if hashes["frozen_parameters"] != expected_parameter:
        raise GateError("ledger parameter hash does not bind frozen parameter bytes")
    if hashes["firmware_binary_manifest"] != expected_firmware:
        raise GateError("ledger firmware hash does not bind firmware manifest bytes")
    return hashes, sha256_file(path)


def evidence_root(core_hashes: dict[str, str], artifact_hashes: dict[str, str]) -> str:
    payload = {
        "schema": GATE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "core": dict(sorted(core_hashes.items())),
        "artifacts": dict(sorted(artifact_hashes.items())),
    }
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def check_signers(path: Path) -> dict[str, dict[str, Any]]:
    item = load_json(path)
    if item.get("schema") != SIGNER_SCHEMA or item.get("protocol_id") != PROTOCOL_ID:
        raise GateError("signer registry schema/protocol mismatch")
    entries = item.get("signers")
    if not isinstance(entries, list):
        raise GateError("signers must be list")
    result: dict[str, dict[str, Any]] = {}
    base = path.parent.resolve()
    for entry in entries:
        signer = str(entry.get("signer_id", ""))
        if not signer or signer in result:
            raise GateError("missing/duplicate signer_id")
        if entry.get("algorithm") != "hmac-sha256":
            raise GateError(f"unsupported signer algorithm: {signer}")
        key_path = Path(str(entry.get("key_path", "")))
        if not key_path.is_absolute():
            key_path = (base / key_path).resolve()
        if not key_path.is_file():
            raise GateError(f"signer key missing: {signer}")
        key_hash = str(entry.get("key_sha256", ""))
        require_sha(key_hash, f"{signer}.key_sha256")
        if sha256_file(key_path) != key_hash:
            raise GateError(f"signer key hash mismatch: {signer}")
        result[signer] = {**entry, "key_path_resolved": key_path}
    return result


def check_reports(paths: list[Path], root: str, signers: dict[str, dict[str, Any]]) -> dict[str, str]:
    reports: dict[str, str] = {}
    signer_by_report: dict[str, str] = {}
    for path in paths:
        report = load_json(path)
        if report.get("schema") != REPORT_SCHEMA or report.get("protocol_id") != PROTOCOL_ID:
            raise GateError(f"report schema/protocol mismatch: {path}")
        report_type = str(report.get("report_type", ""))
        if report_type not in REQUIRED_REPORTS or report_type in reports:
            raise GateError(f"unexpected/duplicate report type: {report_type}")
        if report.get("passed") is not True or report.get("evidence_root_sha256") != root:
            raise GateError(f"report is not PASS on common evidence root: {report_type}")
        if report.get("signature_algorithm") != "hmac-sha256":
            raise GateError(f"report signature algorithm invalid: {report_type}")
        parse_time(str(report.get("signed_at_utc", "")), f"{report_type}.signed_at_utc")
        signer = str(report.get("signer_id", ""))
        record = signers.get(signer)
        if not record or record.get("role") != REQUIRED_REPORTS[report_type]:
            raise GateError(f"report signer/role invalid: {report_type}")
        if report_type == "independent_confirmation" and record.get("independent") is not True:
            raise GateError("independent confirmation signer is not marked independent")
        signature = str(report.get("signature", ""))
        require_sha(signature, f"{report_type}.signature")
        unsigned = dict(report)
        unsigned.pop("signature", None)
        key = Path(record["key_path_resolved"]).read_bytes()
        expected = hmac.new(key, canonical_bytes(unsigned), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise GateError(f"report signature mismatch: {report_type}")
        reports[report_type] = sha256_file(path)
        signer_by_report[report_type] = signer
    missing = sorted(set(REQUIRED_REPORTS) - set(reports))
    if missing:
        raise GateError(f"missing signed PASS reports: {missing}")
    independent_signer = signer_by_report["independent_confirmation"]
    if independent_signer in {signer_by_report["physical_raw_validation"],
                              signer_by_report["causal_equation_replay"],
                              signer_by_report["certificate_validation"]}:
        raise GateError("independent confirmer must differ from primary validators")
    return reports


def check_source_data_report(path: Path | None) -> tuple[bool, str | None]:
    if path is None:
        return False, None
    if not path.resolve().is_file():
        raise GateError(f"Source Data validator report is not a regular file: {path}")
    report = load_json(path)
    if (report.get("schema") != "PHYS-E2E-OCT-R34-source-data-validation-v1"
            or report.get("passed") is not True):
        raise GateError("Source Data validator report is not a recognised PASS report")
    if report.get("figure_timeseries_verified") is not True:
        raise GateError("Source Data report does not verify the figure time series")
    for field in ("raw_manifest_sha256", "physical_validator_report_sha256",
                  "timeseries_sha256"):
        require_sha(str(report.get(field, "")), f"Source Data report.{field}")
    verified_raw = report.get("verified_raw_file_sha256s")
    if (not isinstance(verified_raw, list) or not verified_raw
            or len(verified_raw) != len(set(verified_raw))):
        raise GateError("Source Data report lacks unique verified raw-file hashes")
    for value in verified_raw:
        require_sha(str(value), "Source Data report.verified_raw_file_sha256s")
    return True, sha256_file(path)


def validate(args: argparse.Namespace) -> dict[str, Any]:
    apparatus, apparatus_hash, nodes = check_apparatus(args.apparatus_manifest)
    schedule, schedule_hash, schedule_manifest_hash = check_schedule(
        args.schedule, args.schedule_manifest)
    ledger, counts = check_ledger(args.run_ledger, schedule)
    check_run_identity(args.run_identity, schedule, ledger, apparatus, apparatus_hash, nodes)
    parameter_hashes = {row["parameter_sha256"] for row in ledger.values()}
    firmware_hashes = {row["firmware_manifest_sha256"] for row in ledger.values()}
    artifact_hashes, artifact_index_hash = check_artifacts(
        args.artifact_index, next(iter(parameter_hashes)), next(iter(firmware_hashes)))
    core_hashes = {
        "apparatus_manifest": apparatus_hash,
        "schedule": schedule_hash,
        "schedule_manifest": schedule_manifest_hash,
        "run_ledger": sha256_file(args.run_ledger),
        "run_identity": sha256_file(args.run_identity),
        "artifact_index": artifact_index_hash,
    }
    figure_timeseries_verified, source_report_hash = check_source_data_report(
        getattr(args, "source_data_validator_report", None))
    if source_report_hash is not None:
        core_hashes["source_data_validator_report"] = source_report_hash
    root = evidence_root(core_hashes, artifact_hashes)
    signers = check_signers(args.signer_registry)
    report_hashes = check_reports(args.report, root, signers)
    return {
        "schema": GATE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "passed": True,
        "structural_bundle_pass": True,
        "physical_origin_authenticated": False,
        "manuscript_evidence_eligible": False,
        "score_credit": 0,
        "figure_timeseries_verified": figure_timeseries_verified,
        "figure_chain_complete": False,
        "figure_chain_decision": (
            "incomplete: this gate does not validate a figure builder, editable outputs "
            "or panel-provenance manifest"
        ),
        "decision_basis": (
            "local byte hashes and local HMAC consistency only; no out-of-band "
            "hardware-rooted attestation verifier is implemented or independently reviewed"
        ),
        "evidence_root_sha256": root,
        "apparatus_id": apparatus["apparatus_id"],
        "topology_id": apparatus["topology_id"],
        "derived_counts": counts,
        "core_hashes": core_hashes,
        "artifact_hashes": artifact_hashes,
        "signed_report_hashes": report_hashes,
        "warning": (
            "STRUCTURAL PASS ONLY. This output is non-evidentiary, grants no manuscript "
            "claim and no score credit, and does not alter EVIDENCE_STATUS.json."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apparatus-manifest", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--schedule-manifest", type=Path, required=True)
    parser.add_argument("--run-ledger", type=Path, required=True)
    parser.add_argument("--run-identity", type=Path, required=True)
    parser.add_argument("--artifact-index", type=Path, required=True)
    parser.add_argument("--signer-registry", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--source-data-validator-report", type=Path,
                        help=("Optional byte-bound Source Data PASS report. If absent, "
                              "figure_timeseries_verified and figure_chain_complete are false."))
    parser.add_argument("--output", type=Path,
                        help="Optional new derived report; existing files are never overwritten")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = validate(args)
        payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(payload)
        print(payload, end="")
        return 0
    except (GateError, OSError, ValueError) as exc:
        print(f"FAIL promotion gate: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
