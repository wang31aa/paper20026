#!/usr/bin/env python3
"""TEST_ONLY fabricated fixtures for causal-replay contract tests.

The fixtures are software-test inputs, not simulations, measurements, results,
or evidence of hardware execution.
"""
from __future__ import annotations

import copy
import hashlib
import json
import unittest

import validate_causal_replay as replay
from contract_common import ContractError
from validate_parameters import canonical_parameter_payload_sha256


def fixture():
    certificate = {
        "status": "PASS",
        "classification": "PRE_OUTCOME_CERTIFICATE_ONLY_NOT_PHYSICAL_RESULT",
        "protocol_id": replay.PROTOCOL_ID,
        "contains_outcomes": False,
        "physical_result": False,
    }
    certificate_bytes = json.dumps(certificate, sort_keys=True).encode()
    certificate_hash = hashlib.sha256(certificate_bytes).hexdigest()
    parameters = {
        "protocol_id": replay.PROTOCOL_ID,
        "status": "preregistration_parameters_no_results",
        "freeze": {"canonical_parameter_payload_sha256": "0" * 64},
        "observer_controller": {
            "observer_gain_gamma": 0.2,
            "controller_gain_kappa": 0.3,
            "nominal_parameters": [
                {"node_id": 0, "a_nominal": 1.0, "b_nominal": 0.5, "c_nominal": 0.1},
                {"node_id": 1, "a_nominal": 0.8, "b_nominal": 2.0, "c_nominal": 0.1},
                {"node_id": 2, "a_nominal": 0.7, "b_nominal": 1.5, "c_nominal": -0.1},
            ],
        },
        "quantization_actuation": {
            "state_quantisation_step": 0.1,
            "observer_quantisation_step": 0.1,
            "pwm_quantisation_step": 1.0,
            "signed_pwm_min": -10,
            "signed_pwm_max": 10,
        },
        "timing_network": {
            "admissible_packet_ages_samples": [0, 1],
            "computation_deadline_us": 700,
            "max_consecutive_packet_holds": 1,
        },
        "certificate": {
            "verification_report_sha256": certificate_hash,
            "all_vertices_verified": True,
        },
    }
    parameters["freeze"]["canonical_parameter_payload_sha256"] = (
        canonical_parameter_payload_sha256(parameters)
    )
    parameter_file_hash = replay._canonical_file_hash(parameters)

    def raw(node, x, shat, obs_z, shat_next, ctrl_z, command, target="NA", nx="NA", ns="NA"):
        return {
            "protocol_id": replay.PROTOCOL_ID, "run_id": "TEST_ONLY_RUN",
            "arm": "observer_loop", "node_id": str(node), "local_tick": "5",
            "scheduled_time_us": "5000", "local_time_us": "5010",
            "computation_time_us": "200", "actuation_time_us": "5500",
            "deadline_missed": "false", "speed_normalised": str(x),
            "state_quantised_local": str(replay._quantize(x, 0.1)),
            "observer_state_pre": str(shat), "observer_state_post": str(shat_next),
            "observer_innovation": str(obs_z),
            "controller_network_innovation": str(ctrl_z),
            "leader_u_nominal": "0.4", "follower_u_nominal": str(command),
            "u_command": str(command), "u_pre_saturation": str(command),
            "follower_u_clipped_float": str(command), "u_post_saturation": "1.0",
            "follower_u_quantised_signed": "1.0", "direction_bit": "1",
            "pwm_count": "1", "pwm_register": "1", "pwm_magnitude": "1.0",
            "saturation_flag": "false", "pin_state": "true" if node == 1 else "false",
            "received_leader_state": str(target), "received_neighbor_state": str(nx),
            "received_neighbor_observer": str(ns),
            "frozen_parameter_sha256": parameter_file_hash,
        }

    rows = [
        raw(1, .26, 1.04, .2, 1.30, -.7, .621, target=.8),
        raw(2, .44, .96, .1, 1.24, .1, 1.022 / 1.5, nx=.3, ns=.9),
    ]

    def packet(receiver, sender, kind, state="NA", observer="NA", forcing="NA", seq=10):
        return {
            "protocol_id": replay.PROTOCOL_ID, "run_id": "TEST_ONLY_RUN",
            "arm": "observer_loop", "receiver_node": str(receiver),
            "sender_node": str(sender), "local_tick": "5", "sender_tick": "5",
            "packet_sequence": str(seq), "packet_kind": kind,
            "receive_time_us": "5200", "sender_time_us": "5100",
            "packet_age_samples": "0", "held_packet_flag": "false",
            "state_value": str(state), "observer_value": str(observer),
            "forcing_value": str(forcing),
            "frozen_parameter_sha256": parameter_file_hash,
        }
    packets = [
        packet(1, 0, "leader_state", state=.8, seq=1),
        packet(1, 0, "leader_forcing", forcing=.4, seq=2),
        packet(2, 1, "neighbor_state_observer", state=.3, observer=.9, seq=3),
        packet(2, 0, "leader_forcing", forcing=.4, seq=4),
    ]
    return parameters, certificate, certificate_hash, parameter_file_hash, rows, packets


class CausalReplayTests(unittest.TestCase):
    def run_fixture(self, mutate=None):
        p, c, ch, ph, rows, packets = fixture()
        if mutate:
            mutate(p, c, rows, packets)
        return replay.validate_replay(p, c, ch, rows, packets, parameter_file_sha256=ph)

    def test_test_only_fixture_passes_without_hardware_claim(self):
        report = self.run_fixture()
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["hardware_origin_authenticated"])
        self.assertFalse(report["physical_result"])
        self.assertEqual(report["checked_raw_record_count"], 2)

    def test_node2_target_packet_fails_acl(self):
        def mutate(_p, _c, _rows, packets):
            packets[2]["sender_node"] = "0"
            packets[2]["packet_kind"] = "leader_state"
        with self.assertRaisesRegex(ContractError, "ACL"):
            self.run_fixture(mutate)

    def test_node2_target_log_fails(self):
        def mutate(_p, _c, rows, _packets):
            rows[1]["received_leader_state"] = "0.8"
        with self.assertRaisesRegex(ContractError, "target state"):
            self.run_fixture(mutate)

    def test_shat_kplus1_used_by_controller_is_detected(self):
        def mutate(_p, _c, rows, _packets):
            rows[0]["follower_u_nominal"] = "0.66"
            rows[0]["u_command"] = "0.66"
            rows[0]["u_pre_saturation"] = "0.66"
            rows[0]["follower_u_clipped_float"] = "0.66"
        with self.assertRaisesRegex(ContractError, "replay"):
            self.run_fixture(mutate)

    def test_post_quantisation_identity_fails_closed(self):
        def mutate(_p, _c, rows, _packets):
            rows[0]["pwm_register"] = "2"
        with self.assertRaisesRegex(ContractError, "pwm_register"):
            self.run_fixture(mutate)

    def test_packet_received_after_actuation_fails(self):
        def mutate(_p, _c, _rows, packets):
            packets[0]["receive_time_us"] = "5600"
        with self.assertRaisesRegex(ContractError, "after actuation"):
            self.run_fixture(mutate)

    def test_certificate_hash_mismatch_fails(self):
        p, c, _ch, ph, rows, packets = fixture()
        with self.assertRaisesRegex(ContractError, "report bytes"):
            replay.validate_replay(p, c, "f" * 64, rows, packets, parameter_file_sha256=ph)

    def test_certificate_with_outcomes_fails(self):
        def mutate(_p, c, _rows, _packets):
            c["contains_outcomes"] = True
        with self.assertRaisesRegex(ContractError, "contains_outcomes"):
            self.run_fixture(mutate)

    def test_unconsumed_packet_fails(self):
        def mutate(_p, _c, _rows, packets):
            extra = copy.deepcopy(packets[-1])
            extra["local_tick"] = "6"
            extra["sender_tick"] = "6"
            extra["packet_sequence"] = "5"
            packets.append(extra)
        with self.assertRaisesRegex(ContractError, "unconsumed"):
            self.run_fixture(mutate)


if __name__ == "__main__":
    unittest.main()
