#!/usr/bin/env python3
from __future__ import annotations

import inspect
import unittest

try:  # Support both package discovery and direct execution from this directory.
    from .reference_model import (
        ContractViolation,
        FaultCode,
        FirmwareConfig,
        FirmwareState,
        FollowerReferenceModel,
        LeaderForcingSample,
        LeaderStateSample,
        NeighborSample,
        NON_EVIDENTIARY_CLASSIFICATION,
        PwmContract,
        ScalarNominalModel,
        TickInput,
        TimingContract,
        apply_pwm_contract,
        quantize_half_away_from_zero,
    )
except ImportError:
    from reference_model import (
    ContractViolation,
    FaultCode,
    FirmwareConfig,
    FirmwareState,
    FollowerReferenceModel,
    LeaderForcingSample,
    LeaderStateSample,
    NeighborSample,
    NON_EVIDENTIARY_CLASSIFICATION,
    PwmContract,
    ScalarNominalModel,
    TickInput,
    TimingContract,
    apply_pwm_contract,
    quantize_half_away_from_zero,
    )


def config_for(node_id: int) -> FirmwareConfig:
    return FirmwareConfig(
        protocol_id="PHYS-E2E-OCT-R34",
        experiment_id="unit-contract-only",
        node_id=node_id,
        role="physical_follower",
        pin_weight=1.0 if node_id == 1 else 0.0,
        incoming_state_weights=() if node_id == 1 else ((1, 1.0),),
        gamma=0.5,
        kappa=1.0,
        initial_observer_state=1.0,
        state_abs_max=20.0,
        timing=TimingContract(
            sample_period_us=1000,
            acquire_offset_us=100,
            receive_close_offset_us=200,
            pwm_deadline_offset_us=700,
            log_deadline_offset_us=900,
            jitter_abs_max_us=5,
            admissible_packet_ages=(0, 1),
            max_consecutive_holds=1,
        ),
        model=ScalarNominalModel(
            a0=1.0,
            b0=0.0,
            c0=0.0,
            ai=0.0,
            bi=1.0,
            ci=0.0,
        ),
        pwm=PwmContract(
            signed_min=-255.0,
            signed_max=255.0,
            quantization_step=1.0,
        ),
    )


def running_model(node_id: int = 1) -> FollowerReferenceModel:
    model = FollowerReferenceModel(config_for(node_id))
    model.begin_self_test()
    model.complete_self_test(True)
    model.verify_configuration(model.config.fingerprint())
    model.arm_zero()
    model.start("run-contract-unit", 0, 10_000)
    return model


def frame_for(
    node_id: int,
    tick: int = 0,
    leader_state: float = 0.0,
    deadline_miss: bool = False,
    include_forbidden_target: bool = False,
    packet_age: int = 0,
    held: bool = False,
) -> TickInput:
    scheduled = 10_000 + tick * 1000
    neighbors = ()
    if node_id == 2:
        neighbors = (
            NeighborSample(
                sender_node=1,
                sender_tick=tick - packet_age,
                sequence=1,
                x_q=0.25,
                shat_q=0.5,
                held=held,
            ),
        )
    target = None
    if node_id == 1 or include_forbidden_target:
        target = LeaderStateSample(
            sender_tick=tick,
            sequence=tick + 1,
            state_q=leader_state,
        )
    return TickInput(
        tick=tick,
        acquire_time_us=scheduled + 100,
        receive_close_time_us=scheduled + 200,
        compute_finish_time_us=scheduled + 500,
        pwm_write_time_us=scheduled + (701 if deadline_miss else 600),
        log_commit_time_us=scheduled + 800,
        x_local_q=2.0 if node_id == 1 else 1.5,
        forcing=LeaderForcingSample(
            sender_tick=tick,
            sequence=tick + 1,
            u0_applied=0.0,
        ),
        neighbors=neighbors,
        leader_state=target,
    )


class StateMachineTests(unittest.TestCase):
    def test_boot_and_pre_run_states_are_safe_zero(self) -> None:
        model = FollowerReferenceModel(config_for(1))
        self.assertEqual(model.state, FirmwareState.BOOT_SAFE)
        self.assertTrue(model.safe_zero_latched)
        self.assertEqual(model.last_pwm.pwm_register, 0)
        self.assertFalse(model.last_pwm.driver_enable)

    def test_wrong_configuration_hash_latches_zero(self) -> None:
        model = FollowerReferenceModel(config_for(1))
        model.begin_self_test()
        model.complete_self_test(True)
        with self.assertRaises(ContractViolation):
            model.verify_configuration("0" * 64)
        self.assertEqual(model.state, FirmwareState.SAFE_ZERO_LATCHED)
        self.assertEqual(model.first_fault, FaultCode.CONFIG_HASH_MISMATCH)
        self.assertEqual(model.last_pwm.effective_signed_pwm, 0)

    def test_start_without_arm_fails_closed(self) -> None:
        model = FollowerReferenceModel(config_for(1))
        with self.assertRaises(ContractViolation):
            model.start("run", 0, 0)
        self.assertEqual(model.state, FirmwareState.SAFE_ZERO_LATCHED)
        self.assertEqual(model.last_pwm.pwm_register, 0)

    def test_stop_cannot_resume_same_instance(self) -> None:
        model = running_model(1)
        model.stop()
        self.assertEqual(model.state, FirmwareState.SAFE_ZERO_LATCHED)
        with self.assertRaises(ContractViolation):
            model.start("second-run", 0, 0)
        self.assertEqual(model.last_pwm.pwm_register, 0)


class CausalityAndLeakageTests(unittest.TestCase):
    def test_controller_uses_shat_k_not_shat_kplus1(self) -> None:
        model = running_model(1)
        record = model.step(frame_for(1, leader_state=0.0))
        self.assertTrue(record.committed)
        self.assertEqual(record.shat_k_used_by_controller, 1.0)
        self.assertEqual(record.shat_kplus1_candidate, 0.5)
        self.assertEqual(record.pwm.v_unclipped, 0.0)
        self.assertEqual(model.observer_state, 0.5)

    def test_target_changes_next_observer_but_not_current_pwm(self) -> None:
        low_target = running_model(1)
        high_target = running_model(1)
        low = low_target.step(frame_for(1, leader_state=0.0))
        high = high_target.step(frame_for(1, leader_state=10.0))
        self.assertNotEqual(
            low.shat_kplus1_candidate, high.shat_kplus1_candidate
        )
        self.assertEqual(low.pwm.v_unclipped, high.pwm.v_unclipped)
        self.assertEqual(low.pwm.effective_signed_pwm, high.pwm.effective_signed_pwm)

    def test_unpinned_node_rejects_direct_target(self) -> None:
        model = running_model(2)
        record = model.step(frame_for(2, include_forbidden_target=True))
        self.assertFalse(record.committed)
        self.assertEqual(record.fault_code, FaultCode.TARGET_LEAKAGE)
        self.assertEqual(record.pwm.pwm_register, 0)
        self.assertEqual(model.state, FirmwareState.SAFE_ZERO_LATCHED)

    def test_controller_path_has_no_target_argument(self) -> None:
        source = inspect.getsource(FollowerReferenceModel.step)
        controller_section = source.split("controller_innovation =", 1)[1]
        controller_section = controller_section.split("pwm =", 1)[0]
        self.assertNotIn("leader_state", controller_section)


class PwmIdentityTests(unittest.TestCase):
    def test_half_ties_round_away_from_zero(self) -> None:
        self.assertEqual(quantize_half_away_from_zero(2.5, 1.0), (3, 3.0))
        self.assertEqual(quantize_half_away_from_zero(-2.5, 1.0), (-3, -3.0))

    def test_positive_saturation_identity(self) -> None:
        pwm = apply_pwm_contract(300.0, config_for(1).pwm)
        self.assertTrue(pwm.saturation_flag)
        self.assertEqual(pwm.v_clipped, 255.0)
        self.assertEqual(pwm.quantized_signed, 255.0)
        self.assertEqual(pwm.pwm_register, 255)
        self.assertEqual(pwm.effective_signed_pwm, 255.0)
        self.assertEqual(pwm.direction_bit, 0)

    def test_negative_direction_and_register_identity(self) -> None:
        pwm = apply_pwm_contract(-12.6, config_for(1).pwm)
        self.assertEqual(pwm.quantized_signed, -13.0)
        self.assertEqual(pwm.pwm_register, 13)
        self.assertEqual(pwm.effective_signed_pwm, -13.0)
        self.assertEqual(pwm.direction_bit, 1)


class TimingAndLatchTests(unittest.TestCase):
    def test_deadline_miss_latches_safe_zero(self) -> None:
        model = running_model(1)
        failed = model.step(frame_for(1, deadline_miss=True))
        self.assertEqual(failed.fault_code, FaultCode.DEADLINE_MISS)
        self.assertFalse(failed.committed)
        self.assertEqual(failed.pwm.pwm_register, 0)
        follow_up = model.step(frame_for(1))
        self.assertFalse(follow_up.committed)
        self.assertEqual(follow_up.pwm.pwm_register, 0)

    def test_skipped_tick_fails(self) -> None:
        model = running_model(1)
        record = model.step(frame_for(1, tick=1))
        self.assertEqual(record.fault_code, FaultCode.TICK_ORDER)
        self.assertEqual(record.pwm.effective_signed_pwm, 0)

    def test_uncertified_packet_age_fails(self) -> None:
        model = running_model(2)
        record = model.step(frame_for(2, packet_age=2))
        self.assertEqual(record.fault_code, FaultCode.PACKET_AGE)
        self.assertEqual(record.pwm.pwm_register, 0)

    def test_reused_unmarked_packet_sequence_fails(self) -> None:
        model = running_model(2)
        first = model.step(frame_for(2, tick=0))
        self.assertTrue(first.committed)
        second = model.step(frame_for(2, tick=1, packet_age=1))
        self.assertEqual(second.fault_code, FaultCode.PACKET_SEQUENCE)
        self.assertEqual(second.pwm.pwm_register, 0)


class ClassificationTests(unittest.TestCase):
    def test_every_record_is_non_evidentiary(self) -> None:
        model = running_model(1)
        record = model.step(frame_for(1))
        self.assertEqual(record.classification, NON_EVIDENTIARY_CLASSIFICATION)
        self.assertIn("NOT_DEPLOYABLE", record.classification)
        self.assertIn("NOT_HARDWARE_EVIDENCE", record.classification)


if __name__ == "__main__":
    unittest.main()
