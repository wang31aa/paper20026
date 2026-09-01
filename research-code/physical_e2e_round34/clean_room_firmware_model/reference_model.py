#!/usr/bin/env python3
"""Portable, non-deployable reference model for the clean-room firmware contract.

This module is a deterministic state-machine and equation oracle. It does not
access hardware, write PWM registers, provide real-time guarantees, or produce
experimental evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, Optional, Sequence, Tuple


NON_EVIDENTIARY_CLASSIFICATION = (
    "PORTABLE_REFERENCE_MODEL_NOT_DEPLOYABLE_NOT_HARDWARE_EVIDENCE"
)


class ContractViolation(RuntimeError):
    """Raised after the model has latched safe zero."""


class FirmwareState(str, Enum):
    BOOT_SAFE = "BOOT_SAFE"
    SELF_TEST = "SELF_TEST"
    CONFIG_VERIFIED = "CONFIG_VERIFIED"
    ARMED_ZERO = "ARMED_ZERO"
    RUNNING = "RUNNING"
    SAFE_ZERO_LATCHED = "SAFE_ZERO_LATCHED"
    POWER_ISOLATED = "POWER_ISOLATED"


class FaultCode(str, Enum):
    INVALID_TRANSITION = "INVALID_TRANSITION"
    SELF_TEST_FAILED = "SELF_TEST_FAILED"
    CONFIG_HASH_MISMATCH = "CONFIG_HASH_MISMATCH"
    RUN_ID_INVALID = "RUN_ID_INVALID"
    TICK_ORDER = "TICK_ORDER"
    TIMING_CONTRACT = "TIMING_CONTRACT"
    DEADLINE_MISS = "DEADLINE_MISS"
    PACKET_SET = "PACKET_SET"
    PACKET_AGE = "PACKET_AGE"
    PACKET_SEQUENCE = "PACKET_SEQUENCE"
    TARGET_MISSING = "TARGET_MISSING"
    TARGET_LEAKAGE = "TARGET_LEAKAGE"
    NONFINITE = "NONFINITE"
    STATE_DOMAIN = "STATE_DOMAIN"
    SAFETY_INPUT = "SAFETY_INPUT"
    LOG_UNAVAILABLE = "LOG_UNAVAILABLE"
    PWM_IDENTITY = "PWM_IDENTITY"
    STOP_REQUESTED = "STOP_REQUESTED"


@dataclass(frozen=True)
class TimingContract:
    sample_period_us: int
    acquire_offset_us: int
    receive_close_offset_us: int
    pwm_deadline_offset_us: int
    log_deadline_offset_us: int
    jitter_abs_max_us: int
    admissible_packet_ages: Tuple[int, ...] = (0, 1)
    max_consecutive_holds: int = 1

    def validate(self) -> None:
        offsets = (
            self.acquire_offset_us,
            self.receive_close_offset_us,
            self.pwm_deadline_offset_us,
            self.log_deadline_offset_us,
        )
        if self.sample_period_us <= 0:
            raise ValueError("sample_period_us must be positive")
        if not (0 <= offsets[0] <= offsets[1] <= offsets[2] <= offsets[3]):
            raise ValueError("tick offsets must be monotone")
        if offsets[3] >= self.sample_period_us:
            raise ValueError("log deadline must precede the next sample")
        if self.jitter_abs_max_us < 0:
            raise ValueError("jitter_abs_max_us must be nonnegative")
        if not self.admissible_packet_ages:
            raise ValueError("at least one packet age is required")
        if any(age < 0 for age in self.admissible_packet_ages):
            raise ValueError("packet ages must be nonnegative")
        if self.max_consecutive_holds < 0:
            raise ValueError("max_consecutive_holds must be nonnegative")


@dataclass(frozen=True)
class ScalarNominalModel:
    a0: float
    b0: float
    c0: float
    ai: float
    bi: float
    ci: float

    def validate(self) -> None:
        values = (self.a0, self.b0, self.c0, self.ai, self.bi, self.ci)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("nominal model values must be finite")
        if self.bi <= 0:
            raise ValueError("follower nominal input gain must be positive")

    def leader_step(self, state: float, applied_input: float) -> float:
        return self.a0 * state + self.b0 * applied_input + self.c0


@dataclass(frozen=True)
class PwmContract:
    signed_min: float
    signed_max: float
    quantization_step: float
    forward_direction_bit: int = 0
    reverse_direction_bit: int = 1
    zero_direction_bit: int = 0

    def validate(self) -> None:
        values = (self.signed_min, self.signed_max, self.quantization_step)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("PWM values must be finite")
        if not self.signed_min < 0 < self.signed_max:
            raise ValueError("signed PWM range must straddle zero")
        if self.quantization_step <= 0:
            raise ValueError("PWM quantization step must be positive")
        for bit in (
            self.forward_direction_bit,
            self.reverse_direction_bit,
            self.zero_direction_bit,
        ):
            if bit not in (0, 1):
                raise ValueError("direction bits must be binary")
        if self.forward_direction_bit == self.reverse_direction_bit:
            raise ValueError("forward and reverse direction bits must differ")


@dataclass(frozen=True)
class FirmwareConfig:
    protocol_id: str
    experiment_id: str
    node_id: int
    role: str
    pin_weight: float
    incoming_state_weights: Tuple[Tuple[int, float], ...]
    gamma: float
    kappa: float
    initial_observer_state: float
    state_abs_max: float
    timing: TimingContract
    model: ScalarNominalModel
    pwm: PwmContract
    arm: str = "observer_loop"
    classification: str = NON_EVIDENTIARY_CLASSIFICATION

    def validate(self) -> None:
        if self.protocol_id != "PHYS-E2E-OCT-R34":
            raise ValueError("wrong protocol_id")
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if self.node_id not in (1, 2):
            raise ValueError("reference model supports follower nodes 1 and 2 only")
        if self.role != "physical_follower":
            raise ValueError("role must be physical_follower")
        if self.arm != "observer_loop":
            raise ValueError("reference model implements observer_loop only")
        if self.classification != NON_EVIDENTIARY_CLASSIFICATION:
            raise ValueError("non-evidentiary classification is immutable")
        if self.node_id == 1 and self.pin_weight <= 0:
            raise ValueError("node 1 must be pinned")
        if self.node_id == 2 and self.pin_weight != 0:
            raise ValueError("node 2 must be unpinned")
        if self.gamma <= 0 or self.kappa <= 0:
            raise ValueError("observer and controller gains must be positive")
        if not math.isfinite(self.initial_observer_state):
            raise ValueError("initial observer state must be finite")
        if not math.isfinite(self.state_abs_max) or self.state_abs_max <= 0:
            raise ValueError("state_abs_max must be positive and finite")
        senders = [sender for sender, _ in self.incoming_state_weights]
        if len(senders) != len(set(senders)):
            raise ValueError("incoming sender IDs must be unique")
        if any(sender not in (1, 2) or sender == self.node_id for sender in senders):
            raise ValueError("incoming sender IDs are invalid")
        if any(not math.isfinite(weight) or weight <= 0 for _, weight in self.incoming_state_weights):
            raise ValueError("incoming weights must be positive and finite")
        if self.node_id == 1 and senders:
            raise ValueError("minimal chain node 1 has no follower predecessor")
        if self.node_id == 2 and senders != [1]:
            raise ValueError("minimal chain node 2 receives only node 1")
        self.timing.validate()
        self.model.validate()
        self.pwm.validate()

    def fingerprint(self) -> str:
        self.validate()
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def incoming_weights(self) -> Dict[int, float]:
        return dict(self.incoming_state_weights)


@dataclass(frozen=True)
class NeighborSample:
    sender_node: int
    sender_tick: int
    sequence: int
    x_q: float
    shat_q: float
    held: bool = False


@dataclass(frozen=True)
class LeaderStateSample:
    sender_tick: int
    sequence: int
    state_q: float


@dataclass(frozen=True)
class LeaderForcingSample:
    sender_tick: int
    sequence: int
    u0_applied: float


@dataclass(frozen=True)
class TickInput:
    tick: int
    acquire_time_us: int
    receive_close_time_us: int
    compute_finish_time_us: int
    pwm_write_time_us: int
    log_commit_time_us: int
    x_local_q: float
    forcing: LeaderForcingSample
    neighbors: Tuple[NeighborSample, ...] = ()
    leader_state: Optional[LeaderStateSample] = None
    heartbeat_valid: bool = True
    emergency_stop: bool = False
    guard_open: bool = False
    driver_fault: bool = False
    log_available: bool = True


@dataclass(frozen=True)
class AppliedPwm:
    v_unclipped: Optional[float]
    v_clipped: float
    quantized_signed: float
    direction_bit: int
    pwm_register: int
    effective_signed_pwm: float
    saturation_flag: bool
    driver_enable: bool
    safety_override: bool

    @classmethod
    def safe_zero(cls, zero_direction_bit: int) -> "AppliedPwm":
        return cls(
            v_unclipped=None,
            v_clipped=0.0,
            quantized_signed=0.0,
            direction_bit=zero_direction_bit,
            pwm_register=0,
            effective_signed_pwm=0.0,
            saturation_flag=False,
            driver_enable=False,
            safety_override=True,
        )


@dataclass(frozen=True)
class TickRecord:
    classification: str
    run_id: Optional[str]
    tick: int
    state_before: FirmwareState
    state_after: FirmwareState
    fault_code: Optional[FaultCode]
    fault_detail: Optional[str]
    shat_k_used_by_controller: Optional[float]
    shat_kplus1_candidate: Optional[float]
    observer_innovation: Optional[float]
    controller_innovation: Optional[float]
    pwm: AppliedPwm
    committed: bool
    scheduled_time_us: Optional[int]


def quantize_half_away_from_zero(value: float, step: float) -> Tuple[int, float]:
    if not math.isfinite(value) or not math.isfinite(step) or step <= 0:
        raise ValueError("quantizer inputs must be finite and step positive")
    scaled = value / step
    units = int(math.copysign(math.floor(abs(scaled) + 0.5), scaled))
    return units, units * step


def apply_pwm_contract(value: float, contract: PwmContract) -> AppliedPwm:
    contract.validate()
    if not math.isfinite(value):
        raise ContractViolation("non-finite PWM candidate")
    clipped = min(max(value, contract.signed_min), contract.signed_max)
    units, quantized = quantize_half_away_from_zero(
        clipped, contract.quantization_step
    )
    quantized = min(max(quantized, contract.signed_min), contract.signed_max)
    units = int(round(quantized / contract.quantization_step))
    if units > 0:
        direction = contract.forward_direction_bit
        sign = 1
    elif units < 0:
        direction = contract.reverse_direction_bit
        sign = -1
    else:
        direction = contract.zero_direction_bit
        sign = 0
    register = abs(units)
    effective = sign * register * contract.quantization_step
    if effective != quantized:
        raise ContractViolation("signed PWM identity failed")
    return AppliedPwm(
        v_unclipped=value,
        v_clipped=clipped,
        quantized_signed=quantized,
        direction_bit=direction,
        pwm_register=register,
        effective_signed_pwm=effective,
        saturation_flag=(clipped != value),
        driver_enable=True,
        safety_override=False,
    )


class FollowerReferenceModel:
    """Fail-closed reference model for one observer-loop follower node."""

    def __init__(self, config: FirmwareConfig):
        config.validate()
        self.config = config
        self.state = FirmwareState.BOOT_SAFE
        self.observer_state = config.initial_observer_state
        self.last_pwm = AppliedPwm.safe_zero(config.pwm.zero_direction_bit)
        self.first_fault: Optional[FaultCode] = None
        self.first_fault_detail: Optional[str] = None
        self.run_id: Optional[str] = None
        self.start_tick: Optional[int] = None
        self.start_time_us: Optional[int] = None
        self.next_tick: Optional[int] = None
        self._neighbor_sequences: Dict[int, int] = {}
        self._neighbor_holds: Dict[int, int] = {}
        self._leader_sequence: Optional[int] = None
        self._forcing_sequence: Optional[int] = None

    @property
    def safe_zero_latched(self) -> bool:
        return self.state in (
            FirmwareState.BOOT_SAFE,
            FirmwareState.SELF_TEST,
            FirmwareState.CONFIG_VERIFIED,
            FirmwareState.ARMED_ZERO,
            FirmwareState.SAFE_ZERO_LATCHED,
            FirmwareState.POWER_ISOLATED,
        )

    def _force_zero(self) -> None:
        self.last_pwm = AppliedPwm.safe_zero(self.config.pwm.zero_direction_bit)

    def _latch(self, code: FaultCode, detail: str) -> None:
        self._force_zero()
        if self.first_fault is None:
            self.first_fault = code
            self.first_fault_detail = detail
        if self.state != FirmwareState.POWER_ISOLATED:
            self.state = FirmwareState.SAFE_ZERO_LATCHED

    def _transition_failure(self, detail: str) -> None:
        self._latch(FaultCode.INVALID_TRANSITION, detail)
        raise ContractViolation(detail)

    def begin_self_test(self) -> None:
        if self.state != FirmwareState.BOOT_SAFE:
            self._transition_failure("self test may begin only from BOOT_SAFE")
        self._force_zero()
        self.state = FirmwareState.SELF_TEST

    def complete_self_test(self, passed: bool) -> None:
        if self.state != FirmwareState.SELF_TEST:
            self._transition_failure("self test completion requires SELF_TEST")
        if not passed:
            self._latch(FaultCode.SELF_TEST_FAILED, "self test reported failure")
            raise ContractViolation("self test failed")
        self._force_zero()

    def verify_configuration(self, expected_fingerprint: str) -> None:
        if self.state != FirmwareState.SELF_TEST:
            self._transition_failure("configuration verification requires SELF_TEST")
        actual = self.config.fingerprint()
        if expected_fingerprint != actual:
            self._latch(
                FaultCode.CONFIG_HASH_MISMATCH,
                "provided configuration fingerprint does not match",
            )
            raise ContractViolation("configuration hash mismatch")
        self.state = FirmwareState.CONFIG_VERIFIED
        self._force_zero()

    def arm_zero(self) -> None:
        if self.state != FirmwareState.CONFIG_VERIFIED:
            self._transition_failure("arm requires CONFIG_VERIFIED")
        self.state = FirmwareState.ARMED_ZERO
        self._force_zero()

    def start(self, run_id: str, start_tick: int, start_time_us: int) -> None:
        if self.state != FirmwareState.ARMED_ZERO:
            self._transition_failure("start requires ARMED_ZERO")
        if not run_id or start_tick < 0 or start_time_us < 0:
            self._latch(FaultCode.RUN_ID_INVALID, "invalid run identity or start")
            raise ContractViolation("invalid run start")
        self.run_id = run_id
        self.start_tick = start_tick
        self.start_time_us = start_time_us
        self.next_tick = start_tick
        self.state = FirmwareState.RUNNING
        self._force_zero()

    def stop(self, reason: str = "operator stop") -> None:
        self._latch(FaultCode.STOP_REQUESTED, reason)

    def power_isolate(self) -> None:
        self._force_zero()
        self.state = FirmwareState.POWER_ISOLATED

    def _scheduled_time(self, tick: int) -> int:
        if self.start_tick is None or self.start_time_us is None:
            raise ContractViolation("run timing is not initialised")
        return self.start_time_us + (
            tick - self.start_tick
        ) * self.config.timing.sample_period_us

    def _validate_safety(self, frame: TickInput) -> None:
        if (
            not frame.heartbeat_valid
            or frame.emergency_stop
            or frame.guard_open
            or frame.driver_fault
        ):
            raise ContractViolation("safety input requires immediate zero")
        if not frame.log_available:
            raise ContractViolation("device log is unavailable")
        if not math.isfinite(frame.x_local_q):
            raise ContractViolation("local state is non-finite")
        if abs(frame.x_local_q) > self.config.state_abs_max:
            raise ContractViolation("local state exceeds certified domain")

    def _validate_timing(self, frame: TickInput, scheduled: int) -> None:
        timing = self.config.timing
        if frame.tick != self.next_tick:
            raise ContractViolation("tick is duplicated, skipped, or reordered")
        expected_acquire = scheduled + timing.acquire_offset_us
        expected_rx_close = scheduled + timing.receive_close_offset_us
        if abs(frame.acquire_time_us - expected_acquire) > timing.jitter_abs_max_us:
            raise ContractViolation("acquisition jitter exceeds bound")
        if (
            abs(frame.receive_close_time_us - expected_rx_close)
            > timing.jitter_abs_max_us
        ):
            raise ContractViolation("receive-close jitter exceeds bound")
        if not (
            frame.acquire_time_us
            <= frame.receive_close_time_us
            <= frame.compute_finish_time_us
            <= frame.pwm_write_time_us
            <= frame.log_commit_time_us
        ):
            raise ContractViolation("tick event times are not monotone")
        if frame.pwm_write_time_us > scheduled + timing.pwm_deadline_offset_us:
            raise ContractViolation("PWM deadline missed")
        if frame.log_commit_time_us > scheduled + timing.log_deadline_offset_us:
            raise ContractViolation("log deadline missed")

    def _packet_age(self, tick: int, sender_tick: int) -> int:
        age = tick - sender_tick
        if age not in self.config.timing.admissible_packet_ages:
            raise ContractViolation(f"packet age {age} is not certified")
        return age

    def _consume_neighbors(
        self, frame: TickInput
    ) -> Dict[int, NeighborSample]:
        expected = self.config.incoming_weights()
        received = {packet.sender_node: packet for packet in frame.neighbors}
        if len(received) != len(frame.neighbors) or set(received) != set(expected):
            raise ContractViolation("neighbor packet set does not match topology")
        for sender, packet in received.items():
            self._packet_age(frame.tick, packet.sender_tick)
            values = (packet.x_q, packet.shat_q)
            if not all(math.isfinite(value) for value in values):
                raise ContractViolation("neighbor packet contains non-finite state")
            previous = self._neighbor_sequences.get(sender)
            if packet.held:
                if previous is None or packet.sequence != previous:
                    raise ContractViolation("held packet does not match prior sequence")
                holds = self._neighbor_holds.get(sender, 0) + 1
                if holds > self.config.timing.max_consecutive_holds:
                    raise ContractViolation("consecutive packet hold limit exceeded")
                self._neighbor_holds[sender] = holds
            else:
                if previous is not None and packet.sequence <= previous:
                    raise ContractViolation("neighbor sequence is not increasing")
                self._neighbor_sequences[sender] = packet.sequence
                self._neighbor_holds[sender] = 0
        return received

    def _consume_forcing(self, frame: TickInput) -> float:
        packet = frame.forcing
        self._packet_age(frame.tick, packet.sender_tick)
        if not math.isfinite(packet.u0_applied):
            raise ContractViolation("leader forcing is non-finite")
        if self._forcing_sequence is not None and packet.sequence <= self._forcing_sequence:
            raise ContractViolation("leader forcing sequence is not increasing")
        self._forcing_sequence = packet.sequence
        return packet.u0_applied

    def _consume_leader_state(self, frame: TickInput) -> Optional[float]:
        packet = frame.leader_state
        if self.config.pin_weight == 0:
            if packet is not None:
                raise ContractViolation("unpinned node received forbidden target state")
            return None
        if packet is None:
            raise ContractViolation("pinned node lacks required target state")
        self._packet_age(frame.tick, packet.sender_tick)
        if not math.isfinite(packet.state_q):
            raise ContractViolation("leader state is non-finite")
        if self._leader_sequence is not None and packet.sequence <= self._leader_sequence:
            raise ContractViolation("leader state sequence is not increasing")
        self._leader_sequence = packet.sequence
        return packet.state_q

    def _fault_code(self, detail: str) -> FaultCode:
        if "tick is" in detail:
            return FaultCode.TICK_ORDER
        if "deadline" in detail:
            return FaultCode.DEADLINE_MISS
        if "jitter" in detail or "event times" in detail:
            return FaultCode.TIMING_CONTRACT
        if "packet set" in detail or "topology" in detail:
            return FaultCode.PACKET_SET
        if "packet age" in detail or "hold" in detail:
            return FaultCode.PACKET_AGE
        if "sequence" in detail:
            return FaultCode.PACKET_SEQUENCE
        if "forbidden target" in detail:
            return FaultCode.TARGET_LEAKAGE
        if "target state" in detail:
            return FaultCode.TARGET_MISSING
        if "non-finite" in detail:
            return FaultCode.NONFINITE
        if "certified domain" in detail:
            return FaultCode.STATE_DOMAIN
        if "device log" in detail:
            return FaultCode.LOG_UNAVAILABLE
        return FaultCode.SAFETY_INPUT

    def _fault_record(
        self,
        tick: int,
        state_before: FirmwareState,
        code: FaultCode,
        detail: str,
        scheduled: Optional[int],
    ) -> TickRecord:
        self._latch(code, detail)
        return TickRecord(
            classification=NON_EVIDENTIARY_CLASSIFICATION,
            run_id=self.run_id,
            tick=tick,
            state_before=state_before,
            state_after=self.state,
            fault_code=code,
            fault_detail=detail,
            shat_k_used_by_controller=None,
            shat_kplus1_candidate=None,
            observer_innovation=None,
            controller_innovation=None,
            pwm=self.last_pwm,
            committed=False,
            scheduled_time_us=scheduled,
        )

    def step(self, frame: TickInput) -> TickRecord:
        state_before = self.state
        if self.state != FirmwareState.RUNNING:
            return self._fault_record(
                frame.tick,
                state_before,
                self.first_fault or FaultCode.INVALID_TRANSITION,
                self.first_fault_detail or "step requested while not RUNNING",
                None,
            )
        scheduled: Optional[int] = None
        try:
            scheduled = self._scheduled_time(frame.tick)
            self._validate_safety(frame)
            self._validate_timing(frame, scheduled)
            neighbors = self._consume_neighbors(frame)
            u0 = self._consume_forcing(frame)
            leader_state = self._consume_leader_state(frame)

            shat_k = self.observer_state
            observer_innovation = sum(
                weight * (shat_k - neighbors[sender].shat_q)
                for sender, weight in self.config.incoming_state_weights
            )
            if self.config.pin_weight > 0:
                if leader_state is None:
                    raise ContractViolation("pinned target state vanished")
                observer_innovation += self.config.pin_weight * (
                    shat_k - leader_state
                )
            shat_next = (
                self.config.model.leader_step(shat_k, u0)
                - self.config.gamma * observer_innovation
            )

            controller_innovation = sum(
                weight * (frame.x_local_q - neighbors[sender].x_q)
                for sender, weight in self.config.incoming_state_weights
            )
            controller_innovation += self.config.pin_weight * (
                frame.x_local_q - shat_k
            )
            v = (
                self.config.model.leader_step(shat_k, u0)
                - self.config.model.ai * frame.x_local_q
                - self.config.model.ci
                - self.config.kappa * controller_innovation
            ) / self.config.model.bi

            if not all(
                math.isfinite(value)
                for value in (
                    shat_k,
                    observer_innovation,
                    shat_next,
                    controller_innovation,
                    v,
                )
            ):
                raise ContractViolation("observer/controller arithmetic is non-finite")
            pwm = apply_pwm_contract(v, self.config.pwm)
            expected_effective = (
                (1 if pwm.quantized_signed > 0 else -1)
                * pwm.pwm_register
                * self.config.pwm.quantization_step
                if pwm.quantized_signed != 0
                else 0.0
            )
            if (
                pwm.quantized_signed != pwm.effective_signed_pwm
                or pwm.effective_signed_pwm != expected_effective
            ):
                raise ContractViolation("post-quantization applied PWM identity failed")

            # Commit order is deliberate: the tick-k command above used shat_k.
            self.last_pwm = pwm
            self.observer_state = shat_next
            if self.next_tick is None:
                raise ContractViolation("next tick is undefined")
            self.next_tick += 1
            return TickRecord(
                classification=NON_EVIDENTIARY_CLASSIFICATION,
                run_id=self.run_id,
                tick=frame.tick,
                state_before=state_before,
                state_after=self.state,
                fault_code=None,
                fault_detail=None,
                shat_k_used_by_controller=shat_k,
                shat_kplus1_candidate=shat_next,
                observer_innovation=observer_innovation,
                controller_innovation=controller_innovation,
                pwm=pwm,
                committed=True,
                scheduled_time_us=scheduled,
            )
        except ContractViolation as exc:
            detail = str(exc)
            code = (
                FaultCode.PWM_IDENTITY
                if "PWM identity" in detail
                else self._fault_code(detail)
            )
            return self._fault_record(
                frame.tick, state_before, code, detail, scheduled
            )
