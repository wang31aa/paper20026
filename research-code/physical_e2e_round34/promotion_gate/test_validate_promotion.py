from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
from pathlib import Path
import tempfile
import unittest

try:
    from . import validate_promotion as gate
except ImportError:  # Direct execution from promotion_gate/.
    import validate_promotion as gate


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class PromotionFixture:
    def __init__(self, root: Path, blocks: int = 10) -> None:
        self.root = root
        self.blocks = blocks
        self.apparatus = root / "apparatus.json"
        self.schedule = root / "schedule.csv"
        self.schedule_manifest = root / "schedule_manifest.json"
        self.ledger = root / "ledger.csv"
        self.identity = root / "identity.csv"
        self.artifact_index = root / "artifacts.json"
        self.signer_registry = root / "signers.json"
        self.reports: list[Path] = []
        self._build()

    def _build(self) -> None:
        calibration = {node: hashlib.sha256(f"cal-{node}".encode()).hexdigest() for node in range(3)}
        apparatus_value = {
            "schema": gate.APPARATUS_SCHEMA,
            "protocol_id": gate.PROTOCOL_ID,
            "status": "frozen_before_primary",
            "frozen_at_utc": "2026-07-30T10:00:00Z",
            "apparatus_id": "rig-A",
            "topology_id": "physical_chain_target_to_1_to_2",
            "nodes": [
                {
                    "node_id": node,
                    "role": gate.NODE_ROLES[node],
                    "hardware_serial": f"board-{node}",
                    "motor_serial": f"motor-{node}",
                    "encoder_serial": f"encoder-{node}",
                    "calibration_sha256": calibration[node],
                }
                for node in range(3)
            ],
        }
        write_json(self.apparatus, apparatus_value)
        apparatus_hash = gate.sha256_file(self.apparatus)

        schedule_fields = sorted(gate.SCHEDULE_REQUIRED)
        schedule_rows = []
        order = 0
        for repeat in range(1, self.blocks + 1):
            for arm in gate.ARMS:
                order += 1
                schedule_rows.append({
                    "protocol_id": gate.PROTOCOL_ID,
                    "schedule_status": "planned_not_executed",
                    "block_id": f"block-{repeat}",
                    "profile_id": "profile-1",
                    "load_id": "load-1",
                    "repeat_id": f"repeat-{repeat}",
                    "arm": arm,
                    "acquisition_order": str(order),
                    "planned_run_id": f"run-{repeat}-{arm}",
                    "randomisation_seed": "7",
                })
        write_csv(self.schedule, schedule_fields, schedule_rows)
        write_json(self.schedule_manifest, {
            "schema": "PHYS-E2E-OCT-R34-randomisation-v1",
            "protocol_id": gate.PROTOCOL_ID,
            "status": "future_plan_only_not_executed",
            "frozen_at_utc": "2026-07-30T09:00:00Z",
            "profiles": ["profile-1"],
            "loads": ["load-1"],
            "matched_blocks_per_profile_load": self.blocks,
            "arms": list(gate.ARMS),
            "randomisation_seed": 7,
            "schedule_sha256": gate.sha256_file(self.schedule),
            "planned_run_count": 3 * self.blocks,
            "planned_block_count": self.blocks,
            "contains_measurements": False,
            "contains_outcomes": False,
        })

        artifact_paths: dict[str, Path] = {}
        for artifact_id in sorted(gate.REQUIRED_ARTIFACTS):
            item = self.root / f"{artifact_id}.bin"
            item.write_bytes(f"fixture:{artifact_id}".encode())
            artifact_paths[artifact_id] = item
        parameter_hash = gate.sha256_file(artifact_paths["frozen_parameters"])
        firmware_manifest_hash = gate.sha256_file(artifact_paths["firmware_binary_manifest"])

        ledger_fields = sorted(gate.LEDGER_REQUIRED)
        ledger_rows = []
        for row in schedule_rows:
            ledger_rows.append({
                "protocol_id": gate.PROTOCOL_ID,
                "run_id": row["planned_run_id"],
                "block_id": row["block_id"],
                "profile_id": row["profile_id"],
                "load_id": row["load_id"],
                "repeat_id": row["repeat_id"],
                "arm": row["arm"],
                "acquisition_order": row["acquisition_order"],
                "actual_status": "completed",
                "retain_in_analysis": "true",
                "firmware_manifest_sha256": firmware_manifest_hash,
                "parameter_sha256": parameter_hash,
            })
        write_csv(self.ledger, ledger_fields, ledger_rows)

        identity_fields = sorted(gate.IDENTITY_REQUIRED)
        identity_rows = []
        for scheduled in schedule_rows:
            for node in range(3):
                identity_rows.append({
                    "protocol_id": gate.PROTOCOL_ID,
                    "run_id": scheduled["planned_run_id"],
                    "block_id": scheduled["block_id"],
                    "arm": scheduled["arm"],
                    "node_id": str(node),
                    "role": gate.NODE_ROLES[node],
                    "apparatus_id": "rig-A",
                    "hardware_serial": f"board-{node}",
                    "motor_serial": f"motor-{node}",
                    "encoder_serial": f"encoder-{node}",
                    "calibration_sha256": calibration[node],
                    "apparatus_manifest_sha256": apparatus_hash,
                })
        write_csv(self.identity, identity_fields, identity_rows)

        write_json(self.artifact_index, {
            "schema": gate.ARTIFACT_SCHEMA,
            "protocol_id": gate.PROTOCOL_ID,
            "artifacts": [
                {"artifact_id": artifact_id, "path": str(path), "sha256": gate.sha256_file(path)}
                for artifact_id, path in sorted(artifact_paths.items())
            ],
        })

        signer_entries = []
        for report_type, role in gate.REQUIRED_REPORTS.items():
            signer = f"signer-{report_type}"
            key_path = self.root / f"{signer}.key"
            key_path.write_bytes(f"test-secret-{report_type}".encode())
            signer_entries.append({
                "signer_id": signer,
                "role": role,
                "independent": report_type == "independent_confirmation",
                "algorithm": "hmac-sha256",
                "key_path": str(key_path),
                "key_sha256": gate.sha256_file(key_path),
            })
        write_json(self.signer_registry, {
            "schema": gate.SIGNER_SCHEMA,
            "protocol_id": gate.PROTOCOL_ID,
            "signers": signer_entries,
        })
        self._write_reports(signer_entries)

    def compute_root(self) -> str:
        artifact_index = json.loads(self.artifact_index.read_text())
        artifact_hashes = {entry["artifact_id"]: entry["sha256"] for entry in artifact_index["artifacts"]}
        core = {
            "apparatus_manifest": gate.sha256_file(self.apparatus),
            "schedule": gate.sha256_file(self.schedule),
            "schedule_manifest": gate.sha256_file(self.schedule_manifest),
            "run_ledger": gate.sha256_file(self.ledger),
            "run_identity": gate.sha256_file(self.identity),
            "artifact_index": gate.sha256_file(self.artifact_index),
        }
        return gate.evidence_root(core, artifact_hashes)

    def _write_reports(self, signer_entries: list[dict]) -> None:
        root = self.compute_root()
        signer_map = {entry["role"]: entry for entry in signer_entries}
        self.reports = []
        for report_type, role in gate.REQUIRED_REPORTS.items():
            signer = signer_map[role]
            report = {
                "schema": gate.REPORT_SCHEMA,
                "protocol_id": gate.PROTOCOL_ID,
                "report_type": report_type,
                "passed": True,
                "evidence_root_sha256": root,
                "signer_id": signer["signer_id"],
                "signature_algorithm": "hmac-sha256",
                "signed_at_utc": "2026-07-30T11:00:00Z",
            }
            key = Path(signer["key_path"]).read_bytes()
            report["signature"] = hmac.new(key, gate.canonical_bytes(report), hashlib.sha256).hexdigest()
            path = self.root / f"report-{report_type}.json"
            write_json(path, report)
            self.reports.append(path)

    def args(self) -> argparse.Namespace:
        return argparse.Namespace(
            apparatus_manifest=self.apparatus,
            schedule=self.schedule,
            schedule_manifest=self.schedule_manifest,
            run_ledger=self.ledger,
            run_identity=self.identity,
            artifact_index=self.artifact_index,
            signer_registry=self.signer_registry,
            report=self.reports,
            output=None,
            source_data_validator_report=None,
        )


class PromotionGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = PromotionFixture(Path(self.temp.name))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_fully_software_consistent_bundle_remains_non_evidentiary(self) -> None:
        result = gate.validate(self.fixture.args())
        self.assertTrue(result["passed"])
        self.assertTrue(result["structural_bundle_pass"])
        self.assertFalse(result["physical_origin_authenticated"])
        self.assertFalse(result["manuscript_evidence_eligible"])
        self.assertEqual(result["score_credit"], 0)
        self.assertNotIn("physical_evidence_eligible", result)
        self.assertNotIn("manuscript_physical_e2e_claim_allowed", result)
        self.assertNotIn("score_credit_for_physical_e2e", result)
        self.assertEqual(result["derived_counts"]["completed_runs_by_arm"],
                         {arm: 10 for arm in gate.ARMS})
        self.assertFalse(result["figure_timeseries_verified"])
        self.assertFalse(result["figure_chain_complete"])

    def test_one_block_per_profile_load_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            fixture = PromotionFixture(Path(root), blocks=1)
            with self.assertRaisesRegex(gate.GateError, "at least 10"):
                gate.validate(fixture.args())

    def test_nine_blocks_per_profile_load_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            fixture = PromotionFixture(Path(root), blocks=9)
            with self.assertRaisesRegex(gate.GateError, "at least 10"):
                gate.validate(fixture.args())

    def test_unverified_source_report_cannot_complete_figure_chain(self) -> None:
        report = Path(self.temp.name) / "TEST_ONLY_source_report.json"
        write_json(report, {
            "schema": "PHYS-E2E-OCT-R34-source-data-validation-v1",
            "passed": True,
            "figure_timeseries_verified": False,
        })
        args = self.fixture.args()
        args.source_data_validator_report = report
        with self.assertRaisesRegex(gate.GateError, "does not verify"):
            gate.validate(args)

    def test_artifact_byte_tamper_fails(self) -> None:
        index = json.loads(self.fixture.artifact_index.read_text())
        Path(index["artifacts"][0]["path"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(gate.GateError, "artifact byte hash mismatch"):
            gate.validate(self.fixture.args())

    def test_cross_apparatus_identity_fails(self) -> None:
        header, rows = gate.read_csv(self.fixture.identity)
        rows[0]["apparatus_id"] = "rig-B"
        write_csv(self.fixture.identity, header, rows)
        with self.assertRaisesRegex(gate.GateError, "cross-apparatus"):
            gate.validate(self.fixture.args())

    def test_changed_hardware_across_arm_fails(self) -> None:
        header, rows = gate.read_csv(self.fixture.identity)
        row = next(item for item in rows if item["arm"] == "oracle_target" and item["node_id"] == "2")
        row["motor_serial"] = "other-motor"
        write_csv(self.fixture.identity, header, rows)
        with self.assertRaisesRegex(gate.GateError, "frozen apparatus"):
            gate.validate(self.fixture.args())

    def test_missing_arm_in_ledger_fails(self) -> None:
        header, rows = gate.read_csv(self.fixture.ledger)
        rows = [row for row in rows if row["arm"] != "no_target"]
        write_csv(self.fixture.ledger, header, rows)
        with self.assertRaisesRegex(gate.GateError, "misses scheduled runs"):
            gate.validate(self.fixture.args())

    def test_no_completed_run_in_arm_fails(self) -> None:
        header, rows = gate.read_csv(self.fixture.ledger)
        for row in rows:
            if row["arm"] == "oracle_target":
                row["actual_status"] = "failed"
        write_csv(self.fixture.ledger, header, rows)
        with self.assertRaisesRegex(gate.GateError, "completed run"):
            gate.validate(self.fixture.args())

    def test_report_wrong_evidence_root_fails(self) -> None:
        path = self.fixture.reports[0]
        report = json.loads(path.read_text())
        report["evidence_root_sha256"] = "0" * 64
        write_json(path, report)
        with self.assertRaisesRegex(gate.GateError, "common evidence root"):
            gate.validate(self.fixture.args())

    def test_report_signature_tamper_fails(self) -> None:
        path = self.fixture.reports[0]
        report = json.loads(path.read_text())
        report["signature"] = "f" * 64
        write_json(path, report)
        with self.assertRaisesRegex(gate.GateError, "signature mismatch"):
            gate.validate(self.fixture.args())

    def test_missing_independent_report_fails(self) -> None:
        args = self.fixture.args()
        args.report = args.report[:-1]
        with self.assertRaisesRegex(gate.GateError, "missing signed PASS reports"):
            gate.validate(args)

    def test_score_credit_cannot_be_requested_in_artifact_index(self) -> None:
        index = json.loads(self.fixture.artifact_index.read_text())
        index["score_credit"] = 25
        write_json(self.fixture.artifact_index, index)
        with self.assertRaisesRegex(gate.GateError, "may not request evidence decisions or score credit"):
            gate.validate(self.fixture.args())

    def test_claim_eligibility_cannot_be_requested_in_artifact_index(self) -> None:
        index = json.loads(self.fixture.artifact_index.read_text())
        index["manuscript_evidence_eligible"] = True
        write_json(self.fixture.artifact_index, index)
        with self.assertRaisesRegex(gate.GateError, "may not request evidence decisions or score credit"):
            gate.validate(self.fixture.args())


if __name__ == "__main__":
    unittest.main()
