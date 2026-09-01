#!/usr/bin/env python3
"""Validate future run-level Source Data without treating dense samples as n."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from decimal import Decimal, InvalidOperation

from contract_common import (
    ARMS, ContractError, load_json, parse_bool, parse_float, parse_int, read_csv,
    require_columns, require_sha256, resolve_beneath, sha256_file,
)

RUN_COLUMNS = [
    "source_data_schema", "panel_id", "metric", "unit_id_type", "unit_id",
    "run_id", "block_id", "profile_id", "load_id", "repeat_id", "arm",
    "run_status", "retain_in_analysis", "value", "value_unit",
    "n_independent_units", "n_dense_samples", "analysis_version",
    "raw_manifest_sha256", "validator_report_sha256",
]
LEDGER_COLUMNS = [
    "run_id", "block_id", "profile_id", "load_id", "repeat_id", "arm",
    "actual_status", "retain_in_analysis",
]
TIME_COLUMNS = [
    "source_data_schema", "panel_id", "run_id", "block_id", "arm", "node_id",
    "cycle_index", "time_s", "series_name", "value", "value_unit", "run_status",
    "retain_in_analysis", "display_transform", "raw_manifest_sha256",
    "validator_report_sha256",
]
SCHEMA = "PHYS-E2E-OCT-R34-source-data-v1"
RAW_MANIFEST_COLUMNS = [
    "relative_path", "sha256", "bytes", "run_id", "block_id", "arm", "node_id",
]
THIN_RE = re.compile(r"^display_only_every_([1-9][0-9]*)th_cycle$")


def decimal_value(value: str, label: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, TypeError) as exc:
        raise ContractError(f"{label} must be a finite decimal") from exc
    if not parsed.is_finite():
        raise ContractError(f"{label} must be a finite decimal")
    return parsed


def transform_raw(value: Decimal, transform: str, cycle: int) -> Decimal:
    if transform == "none":
        return value
    if transform == "absolute":
        return abs(value)
    if transform == "square":
        return value * value
    match = THIN_RE.fullmatch(transform)
    if match:
        stride = int(match.group(1))
        if cycle % stride != 0:
            raise ContractError(
                f"cycle {cycle} violates declared display thinning stride {stride}")
        return value
    raise ContractError(f"unsupported deterministic display_transform: {transform!r}")


def load_raw_series(raw_dir: Path, manifest_rows: list[dict[str, str]],
                    required_pairs: set[tuple[str, str]]) -> tuple[
                        dict[tuple[str, str], dict[int, dict[str, str]]], set[str]]:
    entries: dict[tuple[str, str], dict[str, str]] = {}
    for entry in manifest_rows:
        pair = (entry["run_id"], entry["node_id"])
        if pair in entries:
            raise ContractError(f"raw manifest duplicates run/node: {pair}")
        entries[pair] = entry
    missing = sorted(required_pairs - set(entries))
    if missing:
        raise ContractError(f"time series lacks raw-file manifest entries: {missing[:8]}")
    result: dict[tuple[str, str], dict[int, dict[str, str]]] = {}
    verified_hashes: set[str] = set()
    for pair in sorted(required_pairs):
        entry = entries[pair]
        require_sha256(entry["sha256"], f"raw manifest {pair}.sha256")
        raw_path = resolve_beneath(raw_dir, entry["relative_path"])
        if not raw_path.is_file():
            raise ContractError(f"time-series raw file is absent: {raw_path}")
        if raw_path.stat().st_size != parse_int(entry["bytes"], f"raw manifest {pair}.bytes"):
            raise ContractError(f"time-series raw file size mismatch: {raw_path}")
        actual_hash = sha256_file(raw_path)
        if actual_hash != entry["sha256"]:
            raise ContractError(f"time-series raw file byte hash mismatch: {raw_path}")
        header, rows = read_csv(raw_path)
        require_columns(header, ["run_id", "block_id", "arm", "node_id",
                                        "cycle_index", "local_time_us"],
                        f"time-series raw log {raw_path.name}")
        by_cycle: dict[int, dict[str, str]] = {}
        for row in rows:
            if ((row["run_id"], row["node_id"]) != pair
                    or row["block_id"] != entry["block_id"]
                    or row["arm"] != entry["arm"]):
                raise ContractError(f"raw time-series identity mismatch: {raw_path}")
            cycle = parse_int(row["cycle_index"], f"{raw_path}.cycle_index")
            if cycle in by_cycle:
                raise ContractError(f"duplicate raw cycle {pair}/{cycle}")
            by_cycle[cycle] = row
        result[pair] = by_cycle
        verified_hashes.add(actual_hash)
    return result, verified_hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-level", type=Path, required=True)
    parser.add_argument("--run-ledger", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True,
                        help="Exact raw-file manifest validated by the physical report.")
    parser.add_argument("--physical-validator-report", type=Path, required=True,
                        help="Exact PASS report emitted by validate_physical_e2e.py.")
    parser.add_argument("--timeseries", type=Path)
    parser.add_argument("--raw-dir", type=Path,
                        help="Required with --timeseries; immutable raw logs named by the manifest.")
    parser.add_argument("--report", type=Path,
                        help="Optional new report path; overwrite is prohibited.")
    args = parser.parse_args()
    try:
        for label, path in (("run ledger", args.run_ledger),
                            ("raw manifest", args.raw_manifest),
                            ("physical validator report", args.physical_validator_report),
                            ("run-level Source Data", args.run_level)):
            if not path.resolve().is_file():
                raise ContractError(f"explicit {label} artifact is not a regular file: {path}")
        raw_manifest_hash = sha256_file(args.raw_manifest)
        physical_report_hash = sha256_file(args.physical_validator_report)
        physical_report = load_json(args.physical_validator_report)
        if (physical_report.get("schema") != "PHYS-E2E-OCT-R34-raw-validation-v1"
                or physical_report.get("passed") is not True):
            raise ContractError("physical validator report is not a recognised PASS report")
        if physical_report.get("raw_manifest_sha256") != raw_manifest_hash:
            raise ContractError("physical validator report is not bound to the supplied raw manifest bytes")
        if physical_report.get("run_ledger_sha256") != sha256_file(args.run_ledger):
            raise ContractError("physical validator report is not bound to the supplied run ledger bytes")

        ledger_header, ledger_rows = read_csv(args.run_ledger)
        require_columns(ledger_header, LEDGER_COLUMNS, "run ledger")
        ledger = {row["run_id"]: row for row in ledger_rows}
        if len(ledger) != len(ledger_rows):
            raise ContractError("run ledger contains duplicate run_id")

        header, rows = read_csv(args.run_level)
        require_columns(header, RUN_COLUMNS, "run-level Source Data")
        if not rows:
            raise ContractError("run-level Source Data is empty; template headers are not evidence")
        groups: dict[tuple[str, str], list[dict[str, str]]] = {}
        represented_runs: set[str] = set()
        for row in rows:
            if row["source_data_schema"] != SCHEMA:
                raise ContractError("unexpected Source Data schema")
            run_id = row["run_id"]
            if run_id not in ledger:
                raise ContractError(f"Source Data contains unregistered run: {run_id}")
            registered = ledger[run_id]
            for field, ledger_field in (
                ("run_status", "actual_status"), ("block_id", "block_id"),
                ("profile_id", "profile_id"), ("load_id", "load_id"),
                ("repeat_id", "repeat_id"), ("arm", "arm"),
            ):
                if row[field] != registered[ledger_field]:
                    raise ContractError(f"{run_id} Source Data/ledger {field} mismatch")
            if not parse_bool(row["retain_in_analysis"], f"{run_id}.retain_in_analysis"):
                raise ContractError(f"{run_id} is excluded from Source Data")
            if not parse_bool(registered["retain_in_analysis"], f"{run_id}.ledger retain"):
                raise ContractError(f"{run_id} ledger deletes a registered run")
            unit_type = row["unit_id_type"]
            if unit_type == "hardware_trial":
                if row["unit_id"] != run_id:
                    raise ContractError(f"{run_id} hardware_trial unit_id must equal run_id")
            elif unit_type == "matched_hardware_block":
                if row["unit_id"] != row["block_id"]:
                    raise ContractError(f"{run_id} matched block unit_id must equal block_id")
            else:
                raise ContractError(f"invalid independent unit type: {unit_type!r}")
            parse_float(row["value"], f"{run_id}.{row['metric']}.value")
            if not row["value_unit"].strip():
                raise ContractError(f"{run_id}.{row['metric']} lacks value_unit")
            require_sha256(row["raw_manifest_sha256"], f"{run_id}.raw_manifest_sha256")
            require_sha256(row["validator_report_sha256"], f"{run_id}.validator_report_sha256")
            if row["raw_manifest_sha256"] != raw_manifest_hash:
                raise ContractError(f"{run_id} cites different raw-manifest bytes")
            if row["validator_report_sha256"] != physical_report_hash:
                raise ContractError(f"{run_id} cites different physical-validator report bytes")
            represented_runs.add(run_id)
            groups.setdefault((row["panel_id"], row["metric"]), []).append(row)

        for (panel, metric), group in groups.items():
            run_ids = [row["run_id"] for row in group]
            if len(run_ids) != len(set(run_ids)):
                raise ContractError(f"{panel}/{metric} contains duplicate run rows")
            unit_types = {row["unit_id_type"] for row in group}
            if len(unit_types) != 1:
                raise ContractError(f"{panel}/{metric} mixes independent-unit definitions")
            expected_n = len({row["unit_id"] for row in group})
            declared_n = {parse_int(row["n_independent_units"],
                                    f"{panel}/{metric}.n_independent_units") for row in group}
            if declared_n != {expected_n}:
                raise ContractError(
                    f"{panel}/{metric} n must equal {expected_n} distinct independent units, not sample rows")
            dense_counts = {parse_int(row["n_dense_samples"],
                                      f"{panel}/{metric}.n_dense_samples") for row in group}
            if len(dense_counts) != 1 or min(dense_counts) < len(group):
                raise ContractError(f"{panel}/{metric} has inconsistent dense-sample accounting")
            completed = {run_id for run_id, item in ledger.items()
                         if item["actual_status"] == "completed"}
            present = {row["run_id"] for row in group}
            omitted = sorted(completed - present)
            if omitted:
                raise ContractError(
                    f"{panel}/{metric} omits completed ledger runs: {omitted[:8]}")
            by_block: dict[str, list[dict[str, str]]] = {}
            for row in group:
                if row["run_status"] == "completed":
                    by_block.setdefault(row["block_id"], []).append(row)
            completed_blocks = {
                row["block_id"] for row in ledger.values()
                if row["actual_status"] == "completed"
            }
            for block in completed_blocks:
                ledger_block = [item for item in ledger.values()
                                if item["block_id"] == block]
                if all(item["actual_status"] == "completed" for item in ledger_block):
                    arms = [row["arm"] for row in by_block.get(block, [])]
                    if sorted(arms) != sorted(ARMS):
                        raise ContractError(
                            f"{panel}/{metric} block {block} lacks exactly one row per arm")

        failed_or_aborted = {
            run_id for run_id, row in ledger.items()
            if row["actual_status"] in {"failed", "aborted"}
        }
        missing_runs = sorted(set(ledger) - represented_runs)
        if missing_runs:
            raise ContractError(f"ledger runs are absent from Source Data: {missing_runs[:8]}")

        timeseries_rows = 0
        figure_timeseries_verified = False
        timeseries_hash = None
        verified_raw_file_hashes: set[str] = set()
        if args.timeseries:
            if args.raw_dir is None or not args.raw_dir.resolve().is_dir():
                raise ContractError("--timeseries requires an explicit regular --raw-dir")
            if not args.timeseries.resolve().is_file():
                raise ContractError(f"explicit time-series artifact is not a regular file: {args.timeseries}")
            time_header, time_rows = read_csv(args.timeseries)
            require_columns(time_header, TIME_COLUMNS, "time-series Source Data")
            manifest_header, manifest_rows = read_csv(args.raw_manifest)
            require_columns(manifest_header, ["run_id", "block_id", "arm", "node_id"],
                            "raw-file manifest")
            raw_identities = {(r["run_id"], r["block_id"], r["arm"], r["node_id"])
                              for r in manifest_rows}
            seen_keys: set[tuple[str, str, str, int]] = set()
            last: dict[tuple[str, str, str], tuple[int, float]] = {}
            required_pairs = {(row["run_id"], row["node_id"]) for row in time_rows}
            raw_series, verified_raw_file_hashes = load_raw_series(
                args.raw_dir, manifest_rows, required_pairs)
            for row in time_rows:
                if row["source_data_schema"] != SCHEMA or row["run_id"] not in ledger:
                    raise ContractError("time-series Source Data identity mismatch")
                registered = ledger[row["run_id"]]
                for field, ledger_field in (("block_id", "block_id"), ("arm", "arm"),
                                            ("run_status", "actual_status"),
                                            ("retain_in_analysis", "retain_in_analysis")):
                    if row[field] != registered[ledger_field]:
                        raise ContractError(f"time-series {row['run_id']} {field} mismatch")
                identity = (row["run_id"], row["block_id"], row["arm"], row["node_id"])
                if identity not in raw_identities:
                    raise ContractError(f"time-series row has no matching raw-manifest node: {identity}")
                cycle = parse_int(row["cycle_index"], "time-series cycle_index")
                time_s = parse_float(row["time_s"], "time-series time_s")
                parse_float(row["value"], "time-series value")
                if not row["panel_id"].strip() or not row["series_name"].strip():
                    raise ContractError("time-series panel_id and series_name must be explicit")
                if not row["value_unit"].strip():
                    raise ContractError("time-series value_unit must be explicit")
                key = (row["run_id"], row["node_id"], row["series_name"], cycle)
                if key in seen_keys:
                    raise ContractError(f"duplicate time-series identity/cycle key: {key}")
                seen_keys.add(key)
                series = key[:3]
                if series in last and (cycle <= last[series][0] or time_s <= last[series][1]):
                    raise ContractError(f"non-monotone time-series cycle/time for {series}")
                last[series] = (cycle, time_s)
                if not row["display_transform"].strip():
                    raise ContractError("time-series display transform must be explicit, including 'none'")
                raw_row = raw_series[(row["run_id"], row["node_id"])].get(cycle)
                if raw_row is None:
                    raise ContractError(
                        f"time-series cycle has no validated raw row: {row['run_id']}/{row['node_id']}/{cycle}")
                series_name = row["series_name"]
                if series_name not in raw_row:
                    raise ContractError(f"series_name is not a raw-log column: {series_name}")
                raw_numeric = decimal_value(raw_row[series_name], f"raw {series_name}")
                expected_value = transform_raw(raw_numeric, row["display_transform"], cycle)
                plotted_value = decimal_value(row["value"], "time-series plotted value")
                if plotted_value != expected_value:
                    raise ContractError(
                        f"plotted value is not deterministic raw transform for {key}")
                raw_time_s = decimal_value(raw_row["local_time_us"], "raw local_time_us") / Decimal(1000000)
                if decimal_value(row["time_s"], "time-series time_s") != raw_time_s:
                    raise ContractError(f"time_s is not raw local_time_us/1e6 for {key}")
                require_sha256(row["raw_manifest_sha256"], "time-series raw_manifest_sha256")
                require_sha256(row["validator_report_sha256"], "time-series validator_report_sha256")
                if (row["raw_manifest_sha256"] != raw_manifest_hash
                        or row["validator_report_sha256"] != physical_report_hash):
                    raise ContractError("time-series artifact hashes do not resolve to supplied bytes")
            timeseries_rows = len(time_rows)
            if timeseries_rows == 0:
                raise ContractError("supplied time-series Source Data is empty")
            figure_timeseries_verified = True
            timeseries_hash = sha256_file(args.timeseries)

        report = {
            "schema": "PHYS-E2E-OCT-R34-source-data-validation-v1",
            "passed": True,
            "run_level_rows": len(rows),
            "time_series_rows": timeseries_rows,
            "panels": sorted({row["panel_id"] for row in rows}),
            "represented_independent_runs": len(represented_runs),
            "failed_or_aborted_runs_retained": len(failed_or_aborted),
            "dense_samples_are_n": False,
            "figure_timeseries_verified": figure_timeseries_verified,
            "raw_manifest_sha256": raw_manifest_hash,
            "physical_validator_report_sha256": physical_report_hash,
            "timeseries_sha256": timeseries_hash,
            "verified_raw_file_sha256s": sorted(verified_raw_file_hashes),
            "n_rule": "n_independent_units equals distinct hardware trials or matched blocks",
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
