"""TEST_ONLY contract tests. No row is a measurement or physical result."""

import csv
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import validate_physical_e2e as physical
import validate_source_data as source


H = "1" * 64
ARMS = ("observer_loop", "oracle_target", "no_target")


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SourceDataP1Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="TEST_ONLY_round36_")
        self.root = Path(self.temp.name)
        self.ledger = self.root / "TEST_ONLY_ledger.csv"
        ledger_rows = []
        for index, arm in enumerate(ARMS, 1):
            ledger_rows.append({
                "run_id": f"TEST_ONLY_RUN_{index}", "block_id": "TEST_ONLY_BLOCK",
                "profile_id": "TEST_ONLY_PROFILE", "load_id": "TEST_ONLY_LOAD",
                "repeat_id": "1", "arm": arm, "actual_status": "completed",
                "retain_in_analysis": "true",
            })
        write_csv(self.ledger, source.LEDGER_COLUMNS, ledger_rows)
        self.raw_manifest = self.root / "TEST_ONLY_raw_manifest.csv"
        self.raw_dir = self.root / "TEST_ONLY_raw"
        self.raw_dir.mkdir()
        raw_path = self.raw_dir / "TEST_ONLY_RUN_1_node0.csv"
        write_csv(raw_path,
                  ["run_id", "block_id", "arm", "node_id", "cycle_index",
                   "local_time_us", "tracking_error"],
                  [{"run_id": "TEST_ONLY_RUN_1", "block_id": "TEST_ONLY_BLOCK",
                    "arm": "observer_loop", "node_id": "0", "cycle_index": "1",
                    "local_time_us": "0", "tracking_error": "0"}])
        manifest_columns = ["run_id", "block_id", "arm", "node_id",
                            "relative_path", "sha256", "bytes"]
        manifest_rows = [
            {"run_id": row["run_id"], "block_id": row["block_id"],
             "arm": row["arm"], "node_id": str(node),
             "relative_path": (raw_path.name if index == 1 and node == 0
                               else f"TEST_ONLY_UNUSED_{index}_{node}.csv"),
             "sha256": (digest(raw_path) if index == 1 and node == 0 else H),
             "bytes": (str(raw_path.stat().st_size) if index == 1 and node == 0 else "0")}
            for index, row in enumerate(ledger_rows, 1) for node in range(3)
        ]
        write_csv(self.raw_manifest, manifest_columns, manifest_rows)
        self.report = self.root / "TEST_ONLY_physical_report.json"
        self.report.write_text(json.dumps({
            "schema": "PHYS-E2E-OCT-R34-raw-validation-v1", "passed": True,
            "raw_manifest_sha256": digest(self.raw_manifest),
            "run_ledger_sha256": digest(self.ledger),
            "test_only": True,
        }), encoding="utf-8")
        self.run_level = self.root / "TEST_ONLY_source.csv"
        rows = []
        for index, arm in enumerate(ARMS, 1):
            rows.append(dict(zip(source.RUN_COLUMNS, [
                source.SCHEMA, "TEST_ONLY_PANEL", "TEST_ONLY_METRIC",
                "hardware_trial", f"TEST_ONLY_RUN_{index}", f"TEST_ONLY_RUN_{index}",
                "TEST_ONLY_BLOCK", "TEST_ONLY_PROFILE", "TEST_ONLY_LOAD", "1", arm,
                "completed", "true", "0", "TEST_ONLY_UNIT", "3", "3",
                "TEST_ONLY_ANALYSIS", digest(self.raw_manifest), digest(self.report),
            ])))
        write_csv(self.run_level, source.RUN_COLUMNS, rows)

    def tearDown(self):
        self.temp.cleanup()

    def run_validator(self, extra=()):
        argv = ["validate_source_data.py", "--run-level", str(self.run_level),
                "--run-ledger", str(self.ledger), "--raw-manifest",
                str(self.raw_manifest), "--physical-validator-report", str(self.report), *extra]
        with patch("sys.argv", argv), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            return source.main()

    def run_validator_with_output(self, extra=()):
        argv = ["validate_source_data.py", "--run-level", str(self.run_level),
                "--run-ledger", str(self.ledger), "--raw-manifest",
                str(self.raw_manifest), "--physical-validator-report", str(self.report), *extra]
        output = StringIO()
        with patch("sys.argv", argv), redirect_stdout(output), redirect_stderr(StringIO()):
            code = source.main()
        return code, json.loads(output.getvalue()) if code == 0 else None

    def test_exact_three_arm_completed_coverage_passes(self):
        code, report = self.run_validator_with_output()
        self.assertEqual(code, 0)
        self.assertFalse(report["figure_timeseries_verified"])
        self.assertIsNone(report["timeseries_sha256"])

    def test_omitted_completed_run_fails(self):
        _, rows = source.read_csv(self.run_level)
        write_csv(self.run_level, source.RUN_COLUMNS, rows[:-1])
        self.assertEqual(self.run_validator(), 1)

    def test_hash_must_resolve_to_supplied_bytes(self):
        _, rows = source.read_csv(self.run_level)
        rows[0]["raw_manifest_sha256"] = H
        write_csv(self.run_level, source.RUN_COLUMNS, rows)
        self.assertEqual(self.run_validator(), 1)

    def test_duplicate_timeseries_key_fails(self):
        timeseries = self.root / "TEST_ONLY_timeseries.csv"
        row = dict(zip(source.TIME_COLUMNS, [
            source.SCHEMA, "TEST_ONLY_PANEL", "TEST_ONLY_RUN_1", "TEST_ONLY_BLOCK",
            "observer_loop", "0", "1", "0", "tracking_error", "0",
            "TEST_ONLY_UNIT", "completed", "true", "none",
            digest(self.raw_manifest), digest(self.report),
        ]))
        write_csv(timeseries, source.TIME_COLUMNS, [row, row])
        self.assertEqual(self.run_validator(("--timeseries", str(timeseries),
                                             "--raw-dir", str(self.raw_dir))), 1)

    def test_plotted_value_not_recomputed_from_raw_fails(self):
        timeseries = self.root / "TEST_ONLY_timeseries_value.csv"
        row = dict(zip(source.TIME_COLUMNS, [
            source.SCHEMA, "TEST_ONLY_PANEL", "TEST_ONLY_RUN_1", "TEST_ONLY_BLOCK",
            "observer_loop", "0", "1", "0", "tracking_error", "1",
            "TEST_ONLY_UNIT", "completed", "true", "none",
            digest(self.raw_manifest), digest(self.report),
        ]))
        write_csv(timeseries, source.TIME_COLUMNS, [row])
        self.assertEqual(self.run_validator(("--timeseries", str(timeseries),
                                             "--raw-dir", str(self.raw_dir))), 1)

    def test_exact_raw_value_and_time_recompute_passes(self):
        timeseries = self.root / "TEST_ONLY_timeseries_valid.csv"
        row = dict(zip(source.TIME_COLUMNS, [
            source.SCHEMA, "TEST_ONLY_PANEL", "TEST_ONLY_RUN_1", "TEST_ONLY_BLOCK",
            "observer_loop", "0", "1", "0", "tracking_error", "0",
            "TEST_ONLY_UNIT", "completed", "true", "none",
            digest(self.raw_manifest), digest(self.report),
        ]))
        write_csv(timeseries, source.TIME_COLUMNS, [row])
        code, report = self.run_validator_with_output((
            "--timeseries", str(timeseries), "--raw-dir", str(self.raw_dir)))
        self.assertEqual(code, 0)
        self.assertTrue(report["figure_timeseries_verified"])
        self.assertEqual(report["timeseries_sha256"], digest(timeseries))
        self.assertEqual(report["verified_raw_file_sha256s"],
                         [digest(self.raw_dir / "TEST_ONLY_RUN_1_node0.csv")])


class ResetBindingP1Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="TEST_ONLY_reset_")
        self.path = Path(self.temp.name) / "TEST_ONLY_resets.csv"
        self.ledgers = {
            "TEST_ONLY_RUN_1": {
                "reset_event_id": "TEST_ONLY_RESET_1", "acquisition_order": "1",
                "started_at_utc": "2030-01-01T00:01:00Z",
                "ended_at_utc": "2030-01-01T00:02:00Z",
                "operator_id": "TEST_ONLY_OPERATOR",
            }
        }
        self.device = {"TEST_ONLY_RUN_1": {"2" * 64, "3" * 64, "4" * 64}}
        self.host = {"TEST_ONLY_RUN_1": {"5" * 64}}

    def tearDown(self):
        self.temp.cleanup()

    def row(self):
        return {
            "protocol_id": physical.PROTOCOL_ID, "reset_event_id": "TEST_ONLY_RESET_1",
            "run_id": "TEST_ONLY_RUN_1", "acquisition_order": "1",
            "event_time_utc": "2030-01-01T00:00:00Z",
            "event_type": "HARDWARE_RESET_COMPLETE",
            "device_log_sha256s": "|".join(sorted(self.device["TEST_ONLY_RUN_1"])),
            "host_log_sha256": "5" * 64,
            "operator_id": "TEST_ONLY_OPERATOR",
            "hardware_power_cycle_confirmed": "true",
            "load_reset_confirmed": "true", "cooldown_complete": "true",
        }

    def test_reset_is_bound_to_exact_device_and_host_hashes(self):
        write_csv(self.path, physical.RESET_COLUMNS, [self.row()])
        physical.check_reset_events(self.path, self.ledgers, self.device, self.host)

    def test_unbound_reset_hash_fails(self):
        row = self.row()
        row["host_log_sha256"] = "6" * 64
        write_csv(self.path, physical.RESET_COLUMNS, [row])
        with self.assertRaises(physical.ContractError):
            physical.check_reset_events(self.path, self.ledgers, self.device, self.host)


if __name__ == "__main__":
    unittest.main()
