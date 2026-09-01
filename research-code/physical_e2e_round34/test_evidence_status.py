#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from validate_evidence_status import ELIGIBLE_STATUS, EXECUTED_STATUS, validate


ROOT = Path(__file__).resolve().parent


class EvidenceStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = json.loads((ROOT / "EVIDENCE_STATUS.json").read_text())

    def test_frozen_design_state_passes(self) -> None:
        self.assertEqual(validate(self.base), [])

    def test_design_cannot_claim_physical_execution(self) -> None:
        item = copy.deepcopy(self.base)
        item["physical_evidence_eligible"] = True
        item["manuscript_physical_e2e_claim_allowed"] = True
        item["score_credit_for_physical_e2e"] = 25
        self.assertTrue(validate(item))

    def _executed_pending_payload(self) -> dict:
        item = copy.deepcopy(self.base)
        item["status"] = EXECUTED_STATUS
        item["apparatus"]["registered_motor_nodes"] = 3
        item["apparatus"]["physical_leaders"] = 1
        item["apparatus"]["physical_followers"] = 2
        for key in (
            "primary_observer_loop_runs",
            "primary_oracle_target_runs",
            "primary_no_target_runs",
        ):
            item["executions"][key] = 1
        item["frozen_parameter_file"] = "future-parameter.json"
        item["raw_manifest"] = "future-raw-manifest.csv"
        item["firmware_binary_manifest"] = "future-firmware-manifest.json"
        item["certificate_verification"] = "future-certificate-report.json"
        return item

    def test_incomplete_executed_state_fails(self) -> None:
        item = copy.deepcopy(self.base)
        item["status"] = EXECUTED_STATUS
        self.assertTrue(validate(item))

    def test_complete_executed_pending_state_stays_noneligible_zero(self) -> None:
        item = self._executed_pending_payload()
        self.assertEqual(validate(item), [])
        self.assertFalse(item["physical_evidence_eligible"])
        self.assertFalse(item["manuscript_physical_e2e_claim_allowed"])
        self.assertEqual(item["score_credit_for_physical_e2e"], 0)

    def test_manual_eligible_status_is_always_rejected(self) -> None:
        item = self._executed_pending_payload()
        item["status"] = ELIGIBLE_STATUS
        item["physical_evidence_eligible"] = True
        item["manuscript_physical_e2e_claim_allowed"] = True
        item["score_credit_for_physical_e2e"] = 25
        item["independent_confirmation"] = {
            "completed": True,
            "record": "self-asserted-confirmation.json",
        }
        errors = validate(item)
        self.assertTrue(errors)
        self.assertTrue(any("manual eligible_independently_confirmed" in error
                            for error in errors))
        self.assertTrue(any("score credit is prohibited" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
