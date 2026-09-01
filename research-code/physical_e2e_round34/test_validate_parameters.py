#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import validate_parameters as validator


ROOT = Path(__file__).resolve().parent


def hardware_document() -> dict:
    return {
        "apparatus": {
            "physical_motor_count": 3,
            "nodes": [
                {
                    "node_id": 0,
                    "role": "physical_leader",
                    "hardware_serial": "HW0",
                    "motor_serial": "M0",
                    "encoder_serial": "E0",
                },
                {
                    "node_id": 1,
                    "role": "physical_follower",
                    "hardware_serial": "HW1",
                    "motor_serial": "M1",
                    "encoder_serial": "E1",
                },
                {
                    "node_id": 2,
                    "role": "physical_follower",
                    "hardware_serial": "HW2",
                    "motor_serial": "M2",
                    "encoder_serial": "E2",
                },
            ],
        }
    }


def topology_document() -> dict:
    return {
        "topology": {
            "follower_adjacency": [[0, 0], [1, 0]],
            "pinning": [1, 0],
            "pinned_laplacian": [[1, 0], [-1, 1]],
        }
    }


def certificate_document(f_gain: float = 0.2) -> dict:
    return {
        "certificate": {
            "augmented_state_order": 4,
            "lambda": 0.5,
            "disturbance_P_norm_bound": 0.01,
            "invariant_radius_P_norm": 0.1,
            "P": [
                [0.25, 0, 0, 0],
                [0, 0.25, 0, 0],
                [0, 0, 0.25, 0],
                [0, 0, 0, 0.25],
            ],
            "P_min_eigenvalue_lower_bound": 0.24,
            "contraction_margin_lower_bound": 0.05,
            "vertex_count": 1,
            "mode_count": 1,
            "mode_ids": ["delay_0"],
            "verification_tolerance": 0,
            "system_matrices": [
                {
                    "mode_id": "delay_0",
                    "vertex_id": "v0",
                    "F": [
                        [f_gain, 0, 0, 0],
                        [0, f_gain, 0, 0],
                        [0, 0, f_gain, 0],
                        [0, 0, 0, f_gain],
                    ],
                }
            ],
        }
    }


class DraftTests(unittest.TestCase):
    def test_template_lints_only_as_draft(self) -> None:
        template = validator.load_json(ROOT / "PARAMETER_TEMPLATE.json")
        report = validator.lint_draft(template)
        self.assertIn("draft identity", report.checks)
        self.assertEqual(template["status"], "draft")

    def test_default_cli_blocks_draft(self) -> None:
        exit_code = validator.main([str(ROOT / "PARAMETER_TEMPLATE.json")])
        self.assertEqual(exit_code, 2)

    def test_draft_cli_is_explicit_and_non_evidentiary(self) -> None:
        exit_code = validator.main(
            [str(ROOT / "PARAMETER_TEMPLATE.json"), "--draft-lint"]
        )
        self.assertEqual(exit_code, 0)

    def test_outcome_field_is_rejected(self) -> None:
        with self.assertRaises(validator.ValidationFailure):
            validator.reject_outcome_fields({"status": "draft", "results": {}})


class HardwareTopologyTests(unittest.TestCase):
    def test_minimum_hardware_passes(self) -> None:
        validator.validate_hardware(hardware_document())

    def test_duplicate_motor_identity_fails(self) -> None:
        document = hardware_document()
        document["apparatus"]["nodes"][2]["motor_serial"] = "M1"
        with self.assertRaises(validator.ValidationFailure):
            validator.validate_hardware(document)

    def test_pinned_chain_passes(self) -> None:
        validator.validate_topology(topology_document())

    def test_inconsistent_laplacian_fails(self) -> None:
        document = topology_document()
        document["topology"]["pinned_laplacian"][1][0] = 0
        with self.assertRaises(validator.ValidationFailure):
            validator.validate_topology(document)

    def test_unrooted_graph_fails(self) -> None:
        document = topology_document()
        document["topology"]["follower_adjacency"] = [[0, 0], [0, 0]]
        document["topology"]["pinned_laplacian"] = [[1, 0], [0, 0]]
        with self.assertRaises(validator.ValidationFailure):
            validator.validate_topology(document)


class PartitionTests(unittest.TestCase):
    def test_disjoint_partitions_pass(self) -> None:
        document = {
            "data_partitions": {
                "calibration_run_ids": ["cal-1"],
                "identification_run_ids": ["id-1"],
                "confirmation_run_ids": ["conf-1"],
                "primary_run_ids": ["primary-1"],
            }
        }
        validator.validate_partitions(document)

    def test_overlap_fails(self) -> None:
        document = {
            "data_partitions": {
                "calibration_run_ids": ["cal-1"],
                "identification_run_ids": ["shared"],
                "confirmation_run_ids": ["shared"],
                "primary_run_ids": ["primary-1"],
            }
        }
        with self.assertRaises(validator.ValidationFailure):
            validator.validate_partitions(document)


class CertificateTests(unittest.TestCase):
    def test_contracting_certificate_passes(self) -> None:
        validator.validate_certificate(certificate_document())

    def test_noncontracting_matrix_fails(self) -> None:
        with self.assertRaises(validator.ValidationFailure):
            validator.validate_certificate(certificate_document(f_gain=0.9))

    def test_nonpositive_P_fails(self) -> None:
        document = certificate_document()
        document["certificate"]["P"][3][3] = -0.25
        with self.assertRaises(validator.ValidationFailure):
            validator.validate_certificate(document)

    def test_wrong_matrix_count_fails(self) -> None:
        document = certificate_document()
        document["certificate"]["vertex_count"] = 2
        with self.assertRaises(validator.ValidationFailure):
            validator.validate_certificate(document)

    def test_invariant_radius_failure_is_detected(self) -> None:
        document = certificate_document()
        document["certificate"]["disturbance_P_norm_bound"] = 0.08
        with self.assertRaises(validator.ValidationFailure):
            validator.validate_certificate(document)


class HashTests(unittest.TestCase):
    def test_canonical_payload_hash_excludes_only_its_field(self) -> None:
        document = {
            "freeze": {"canonical_parameter_payload_sha256": "0" * 64},
            "protocol_id": "PHYS-E2E-OCT-R34",
        }
        first = validator.canonical_parameter_payload_sha256(document)
        document["freeze"]["canonical_parameter_payload_sha256"] = first
        second = validator.canonical_parameter_payload_sha256(document)
        self.assertEqual(first, second)

    def test_all_zero_hash_fails(self) -> None:
        with self.assertRaises(validator.ValidationFailure):
            validator.validate_hash_fields({"manifest_sha256": "0" * 64})

    def test_schema_is_meta_valid(self) -> None:
        schema = validator.load_json(ROOT / "PARAMETER_SCHEMA.json")
        validator.validate_schema_document(schema)

    def test_draft_cannot_satisfy_completed_schema(self) -> None:
        schema = validator.load_json(ROOT / "PARAMETER_SCHEMA.json")
        template = validator.load_json(ROOT / "PARAMETER_TEMPLATE.json")
        with self.assertRaises(validator.ValidationFailure):
            validator.validate_schema(template, schema)


if __name__ == "__main__":
    unittest.main()
